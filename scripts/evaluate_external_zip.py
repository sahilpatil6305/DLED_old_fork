#!/usr/bin/env python3
"""Evaluate a labeled ZIP using a saved DLED checkpoint and training-only calibration.

Only protocol-referenced images are staged in a temporary directory. The ZIP
and label JSON are read-only; the archive is never extracted wholesale.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile

from compare_archive_protocol import IMAGE_SUFFIXES, clean_path, frame_relative_path


def read_protocol(path, split, real_label, fake_label):
    obj = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("protocol JSON root must be an object")
    roots = [value for value in obj.values() if isinstance(value, dict)]
    if len(roots) != 1:
        raise ValueError("expected exactly one top-level protocol object")
    groups = {"fsgan_Real": {}, "fsgan_Fake": {}}
    empty_videos = 0
    seen_groups = 0
    for group_name, group_data in roots[0].items():
        if not isinstance(group_data, dict):
            raise ValueError(f"protocol group {group_name!r} must be an object")
        if split not in group_data:
            raise ValueError(f"requested split {split!r} is absent from group {group_name!r}")
        videos = group_data[split]
        if not isinstance(videos, dict):
            raise ValueError(f"protocol group {group_name!r} split {split!r} must be an object")
        seen_groups += 1
        for video_name, info in videos.items():
            if not isinstance(info, dict) or not isinstance(info.get("frames"), list):
                raise ValueError(f"{group_name}/{video_name} must contain a frames list")
            frames = info["frames"]
            if not frames:
                empty_videos += 1
                continue
            label = str(info.get("label", ""))
            if label == str(real_label):
                target_group = "fsgan_Real"
            elif label == str(fake_label):
                target_group = "fsgan_Fake"
            else:
                raise ValueError(
                    f"label {label!r} for {group_name}/{video_name} does not match the explicit "
                    f"real/fake labels {real_label!r}/{fake_label!r}"
                )
            output_info = {key: value for key, value in info.items() if key != "frames"}
            output_info.update({"label": target_group, "frames": []})
            output_video_name = f"{group_name}__{video_name}"
            if output_video_name in groups[target_group]:
                raise ValueError(f"duplicate video key after label mapping: {output_video_name}")
            groups[target_group][output_video_name] = output_info
            output_info["frames"] = [str(frame) for frame in frames]
    if not seen_groups:
        raise ValueError(f"no groups contain requested split {split!r}")
    if not groups["fsgan_Real"] or not groups["fsgan_Fake"]:
        raise ValueError("external evaluation requires labeled real and fake videos")
    return groups, empty_videos


def stage_referenced_images(archive_path, protocol_groups, work_dir):
    work_dir = Path(work_dir)
    full_index = {}
    relative_index = {}
    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist()
                 if not info.is_dir() and PurePosixPath(info.filename).suffix.casefold() in IMAGE_SUFFIXES]
        for info in infos:
            path = clean_path(info.filename)
            full_index.setdefault(path, []).append(info)
            suffix = frame_relative_path(path)
            if suffix is not None:
                relative_index.setdefault(suffix, []).append(info)

        extracted = {}
        member_labels = {}
        for group_name, videos in protocol_groups.items():
            for video_name, info in videos.items():
                mapped_frames = []
                for frame in info["frames"]:
                    normalized = clean_path(frame)
                    exact_candidates = full_index.get(normalized, [])
                    suffix = frame_relative_path(normalized)
                    candidates = exact_candidates if exact_candidates else relative_index.get(suffix, [])
                    if not candidates:
                        raise ValueError(f"no ZIP image matches protocol frame: {frame}")
                    if len(candidates) != 1:
                        paths = [candidate.filename for candidate in candidates[:4]]
                        raise ValueError(f"ambiguous ZIP match for {frame}: {paths}")
                    member = candidates[0]
                    prior_label = member_labels.setdefault(member.filename, group_name)
                    if prior_label != group_name:
                        raise ValueError(
                            f"ZIP image {member.filename!r} is referenced with both real and fake labels"
                        )
                    member_path = PurePosixPath(member.filename.replace("\\", "/"))
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError(f"unsafe ZIP member path referenced by protocol: {member.filename}")
                    if member.filename not in extracted:
                        digest = hashlib.sha256(member.filename.encode("utf-8")).hexdigest()
                        target = work_dir / "images" / digest[:2] / f"{digest}{member_path.suffix.lower()}"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted[member.filename] = str(target)
                    mapped_frames.append(extracted[member.filename])
                info["frames"] = mapped_frames

    external_index = {"external_test": {
        group_name: {"test": videos} for group_name, videos in protocol_groups.items()
    }}
    json_dir = work_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / "external_test.json"
    json_path.write_text(json.dumps(external_index, indent=2) + "\n", encoding="utf-8")
    return json_path, {"archive_image_entries": len(infos),
                       "archive_unique_image_paths": len(full_index),
                       "protocol_frame_references": sum(len(v["frames"]) for g in protocol_groups.values() for v in g.values()),
                       "staged_unique_images": len(extracted),
                       "real_videos": len(protocol_groups["fsgan_Real"]),
                       "fake_videos": len(protocol_groups["fsgan_Fake"]),
                       "real_frame_references": sum(len(v["frames"]) for v in protocol_groups["fsgan_Real"].values()),
                       "fake_frame_references": sum(len(v["frames"]) for v in protocol_groups["fsgan_Fake"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="external evaluation ZIP")
    parser.add_argument("--protocol-json", required=True, help="JSON with explicit per-video labels and frame paths")
    parser.add_argument("--split", default="test")
    parser.add_argument("--real-label", required=True, help="exact label string used for real records")
    parser.add_argument("--fake-label", required=True, help="exact label string used for fake records")
    parser.add_argument("--checkpoint", help="best DLED checkpoint")
    parser.add_argument("--train-json-dir", help="directory containing fsgan_pilot_train.json for threshold calibration")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--dataset-root", help="training-data root for threshold calibration (or DF40_DATA_ROOT)")
    parser.add_argument("--old-root", default="deepfakes_detection_datasets")
    parser.add_argument("--dry-run", action="store_true", help="stage and validate paths without loading a model")
    args = parser.parse_args()

    try:
        archive_path = Path(args.archive).expanduser().resolve(strict=True)
        protocol_path = Path(args.protocol_json).expanduser().resolve(strict=True)
        output_dir = Path(args.output_dir).expanduser().resolve()
        groups, empty_videos = read_protocol(protocol_path, args.split, args.real_label, args.fake_label)
        with tempfile.TemporaryDirectory(prefix="dled_external_eval_") as temp_dir:
            work_dir = Path(temp_dir)
            staged_json, summary = stage_referenced_images(archive_path, groups, work_dir)
            summary["empty_protocol_videos_skipped"] = empty_videos
            if args.dry_run:
                print(json.dumps(summary, indent=2))
                return 0
            if not args.checkpoint or not args.train_json_dir:
                raise ValueError("--checkpoint and --train-json-dir are required unless --dry-run is used")
            checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
            train_json = Path(args.train_json_dir).expanduser() / "fsgan_pilot_train.json"
            if not train_json.is_file():
                raise ValueError(f"missing training-only calibration manifest: {train_json}")

            json_dir = staged_json.parent
            shutil.copyfile(train_json, json_dir / train_json.name)
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = output_dir / "external_test_metrics.json"
            log_dir = output_dir / "logs"
            project_root = Path(__file__).resolve().parents[1]
            python_bin = Path(args.python).expanduser()
            if not python_bin.is_absolute():
                python_bin = project_root / python_bin
            test_command = [
                str(python_bin), "training/test.py",
                "--model_path", str(checkpoint),
                "--train_dataset", "fsgan_pilot_train",
                "--test_dataset", "external_test",
                "--dataset_json_folder", str(json_dir),
                "--log_dir", str(log_dir),
                "--num_classes", "2",
                "--training_mode", "single",
                "--metrics_output", str(metrics_path),
                "--experiment_type", "external_test",
                "--output_root", str(project_root / "results"),
            ]
            invocation = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
            (output_dir / "command.json").write_text(json.dumps(invocation, indent=2) + "\n", encoding="utf-8")
            (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            environment = None
            if args.dataset_root:
                environment = os.environ.copy()
                environment["DF40_DATA_ROOT"] = str(Path(args.dataset_root).expanduser().resolve())
                environment["DF40_OLD_ROOT"] = args.old_root
            run_parent = project_root / "results" / "external_test"
            before_runs = set(run_parent.glob("run_*")) if run_parent.exists() else set()
            subprocess.run(test_command, check=True, cwd=project_root, env=environment)
            tracked_runs = sorted(
                [path / "predictions" / "external_test_metrics.json"
                 for path in run_parent.glob("run_*") if path not in before_runs
                 and (path / "predictions" / "external_test_metrics.json").is_file()],
                key=lambda path: path.stat().st_mtime,
            )
            if not tracked_runs:
                raise ValueError("tracked DLED evaluator completed without a metrics JSON")
            metrics_path = tracked_runs[-1]
            print(f"External evaluation metrics: {metrics_path}")
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
