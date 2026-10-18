#!/usr/bin/env python3
"""Build a video-disjoint E4S -> FSGAN pilot index and stage only its PNGs.

The original ZIP archives and DF40 protocol JSONs are read-only.  E4S fake
training frames are indexed from e4s.zip; genuine FF++ frame labels and splits
come from the retained NOEFS/NOFE protocols; FSGAN fake test frames come from
the FF partition in fsgan.zip.  All staged files are decoded and CRC-checked
before the dataset-root directory is atomically replaced.
"""

import argparse
import collections
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
import tempfile

import cv2
import numpy as np
import yaml
import zipfile


OLD_ROOT = "deepfakes_detection_datasets"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def normalized(path):
    return str(PurePosixPath(str(path).replace("\\", "/").lstrip("./")))


def frames_suffix(path):
    parts = normalized(path).split("/")
    try:
        index = len(parts) - 1 - parts[::-1].index("frames")
    except ValueError:
        raise ValueError(f"path has no frames component: {path}")
    return "/".join(parts[index:])


def relative_under_dataset_root(path):
    path = normalized(path)
    prefix = OLD_ROOT + "/"
    if not path.startswith(prefix):
        raise ValueError(f"protocol frame does not start with {OLD_ROOT}/: {path}")
    return path[len(prefix):]


def read_protocol(path, key):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get(key), dict):
        raise ValueError(f"{path} must contain top-level object {key!r}")
    return obj[key]


def zip_images(archive, required_top=None):
    by_suffix = collections.defaultdict(list)
    all_images = []
    for info in archive.infolist():
        member = normalized(info.filename)
        if info.is_dir() or PurePosixPath(member).suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        if PurePosixPath(member).name.startswith("._") or member.startswith("__MACOSX/"):
            continue
        if required_top and not member.startswith(required_top + "/"):
            continue
        if PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts:
            raise ValueError(f"unsafe archive member: {info.filename}")
        all_images.append(info)
        by_suffix[frames_suffix(member)].append(info)
    return all_images, by_suffix


def class_records(groups, group_name, split):
    if group_name not in groups or split not in groups[group_name]:
        raise ValueError(f"protocol has no {group_name}/{split}")
    records = groups[group_name][split]
    if not isinstance(records, dict):
        raise ValueError(f"{group_name}/{split} must be an object")
    return records


def require_unique_match(by_suffix, protocol_path, archive_name):
    suffix = frames_suffix(protocol_path)
    matches = by_suffix.get(suffix, [])
    if len(matches) != 1:
        raise ValueError(
            f"expected one {archive_name} match for {protocol_path}, found {len(matches)}"
        )
    return matches[0]


def build_plan(args):
    train_protocol = read_protocol(args.train_protocol, "DF40_FF_train_NOEFS")
    test_protocol = read_protocol(args.test_protocol, "DF40_FF_test_NOFE_all")
    train_real = class_records(train_protocol, "fsgan_Real", "train")
    test_real = class_records(test_protocol, "fsgan_Real", "test")
    fsgan_test_fake = class_records(test_protocol, "fsgan_Fake", "test")

    train_real_ids = {video.split("_")[0] for video in train_real}
    test_real_ids = set(test_real)
    if train_real_ids & test_real_ids:
        raise ValueError("FF++ real source-video IDs overlap between training and FSGAN test")
    fsgan_fake_ids = {part for video in fsgan_test_fake for part in video.split("_")}
    if fsgan_fake_ids != test_real_ids:
        raise ValueError(
            "FSGAN test fake video endpoints do not exactly match the held-out real video IDs"
        )
    if train_real_ids & fsgan_fake_ids:
        raise ValueError("an FSGAN test source video also appears in E4S training")

    with zipfile.ZipFile(args.e4s_archive) as e4s_zip, \
            zipfile.ZipFile(args.real_archive) as real_zip, \
            zipfile.ZipFile(args.fsgan_archive) as fsgan_zip:
        e4s_images, e4s_suffixes = zip_images(e4s_zip, "e4s")
        real_images, real_suffixes = zip_images(real_zip, "FaceForensics++")
        fsgan_images, fsgan_suffixes = zip_images(fsgan_zip, "ff")

        e4s_frame_infos = [info for info in e4s_images
                           if normalized(info.filename).startswith("e4s/frames/")]
        if not e4s_frame_infos:
            raise ValueError("E4S archive contains no e4s/frames images")
        e4s_by_video = collections.defaultdict(list)
        for info in e4s_frame_infos:
            parts = normalized(info.filename).split("/")
            if len(parts) != 4:
                raise ValueError(f"unexpected E4S frame path: {info.filename}")
            video = parts[2]
            if not re.fullmatch(r"\d+_\d+", video):
                raise ValueError(f"cannot parse E4S source-video pair: {video}")
            e4s_by_video[video].append(info)
        for video in e4s_by_video:
            e4s_by_video[video].sort(key=lambda info: int(PurePosixPath(info.filename).stem))

        pair_components = UnionFind()
        for video in e4s_by_video:
            first, second = video.split("_")
            pair_components.union(first, second)
        for source_id in train_real_ids:
            pair_components.find(source_id)

        e4s_source_ids = set(pair_components.parent)
        if e4s_source_ids & test_real_ids:
            raise ValueError("E4S archive source IDs overlap the held-out FSGAN test videos")
        component_roots = sorted({pair_components.find(source_id) for source_id in e4s_source_ids})
        if len(component_roots) < 2:
            raise ValueError("not enough independent source-video components for validation")
        random.Random(args.seed).shuffle(component_roots)
        val_count = max(1, min(len(component_roots) - 1,
                               round(len(component_roots) * args.val_fraction)))
        val_roots = set(component_roots[:val_count])

        split_records = {
            "train": {"e4s_Real": {}, "e4s_Fake": {}},
            "val": {"e4s_Real": {}, "e4s_Fake": {}},
        }
        split_sources = {"train": set(), "val": set()}
        for source_id, info in train_real.items():
            video_id = source_id.split("_")[0]
            target_split = "val" if pair_components.find(video_id) in val_roots else "train"
            frames = [normalized(path) for path in info.get("frames", [])]
            split_records[target_split]["e4s_Real"][video_id + "_e4s"] = {
                "label": "e4s_Real", "frames": frames,
            }
            split_sources[target_split].add(video_id)

        for video, infos in e4s_by_video.items():
            first, second = video.split("_")
            target_split = "val" if pair_components.find(first) in val_roots else "train"
            frame_paths = [f"{OLD_ROOT}/DF40_train/e4s/frames/{video}/"
                           f"{PurePosixPath(info.filename).name}" for info in infos]
            split_records[target_split]["e4s_Fake"][video + "_e4s"] = {
                "label": "e4s_Fake", "frames": frame_paths,
            }
            split_sources[target_split].update((first, second))

        if split_sources["train"] & split_sources["val"]:
            raise ValueError("source-video leakage detected between E4S train and validation")
        for split in ("train", "val"):
            if not split_records[split]["e4s_Real"] or not split_records[split]["e4s_Fake"]:
                raise ValueError(f"{split} split must contain both real and fake videos")

        test_records = {
            "fsgan_Real": {}, "fsgan_Fake": {},
        }
        for group, records in (("fsgan_Real", test_real), ("fsgan_Fake", fsgan_test_fake)):
            for video, info in records.items():
                test_records[group][video] = {
                    "label": group,
                    "frames": [normalized(path) for path in info.get("frames", [])],
                }

        yaml_config = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8"))
        labels = yaml_config.get("label_dict", {})
        if labels.get("e4s_Real") != 0 or labels.get("e4s_Fake") != 1:
            raise ValueError("train_config.yaml must map e4s_Real=0 and e4s_Fake=1")
        if labels.get("fsgan_Real") != 0 or labels.get("fsgan_Fake") != 1:
            raise ValueError("train_config.yaml must map fsgan_Real=0 and fsgan_Fake=1")

        planned = {}

        def plan_frame(protocol_path, archive_info, zip_path, root_relative):
            target = normalized(root_relative)
            if target in planned and planned[target][1].filename != archive_info.filename:
                raise ValueError(f"two archive members map to {target}")
            planned[target] = (zip_path, archive_info)

        for records in split_records.values():
            for video, info in records["e4s_Real"].items():
                for frame in info["frames"]:
                    member = require_unique_match(real_suffixes, frame, "FF++ real archive")
                    plan_frame(frame, member, args.real_archive,
                               relative_under_dataset_root(frame))
            for video, info in records["e4s_Fake"].items():
                pair = video.removesuffix("_e4s")
                for frame in info["frames"]:
                    suffix = frames_suffix(frame)
                    member = require_unique_match(e4s_suffixes, suffix, "E4S archive")
                    plan_frame(frame, member, args.e4s_archive,
                               relative_under_dataset_root(frame))

        for group, records in test_records.items():
            for video, info in records.items():
                for frame in info["frames"]:
                    if group == "fsgan_Real":
                        member = require_unique_match(real_suffixes, frame, "FF++ real archive")
                    else:
                        member = require_unique_match(fsgan_suffixes, frame, "FSGAN FF archive")
                    plan_frame(frame, member, args.real_archive if group == "fsgan_Real" else args.fsgan_archive,
                               relative_under_dataset_root(frame))

        if set(planned).intersection({".DS_Store"}):
            raise ValueError("unexpected metadata file in extraction plan")

        def describe(groups, split):
            result = {}
            for group, records in groups.items():
                vids = records[split] if split in records else records
                result[group] = {
                    "videos": len(vids),
                    "frame_references": sum(len(info["frames"]) for info in vids.values()),
                }
            return result

        split_roots = {
            name: sorted(values) for name, values in split_sources.items()
        }
        real_source_sets = {
            "e4s_train": sorted(split_sources["train"]),
            "e4s_validation": sorted(split_sources["val"]),
            "fsgan_test": sorted(test_real_ids),
        }
        if (set(real_source_sets["e4s_train"]) & set(real_source_sets["fsgan_test"]) or
                set(real_source_sets["e4s_validation"]) & set(real_source_sets["fsgan_test"])):
            raise ValueError("FF++ real video IDs overlap between E4S and FSGAN test partitions")

        total_uncompressed = sum(info.file_size for _, info in planned.values())
        summary = {
            "experiment": "E4S-to-FSGAN limited cross-method pilot",
            "seed": args.seed,
            "validation_fraction_by_source_component": args.val_fraction,
            "e4s_source_components": len(component_roots),
            "validation_components": val_count,
            "e4s_source_video_counts": {k: len(v) for k, v in split_sources.items()},
            "source_video_id_disjoint": True,
            "fsgan_test_source_ids_match_heldout_ffpp_real_ids": True,
            "real_source_id_overlap_train_val": 0,
            "real_source_id_overlap_train_test": 0,
            "real_source_id_overlap_val_test": 0,
            "e4s_train": describe(split_records["train"], "train"),
            "e4s_validation": describe(split_records["val"], "val"),
            "fsgan_test": {
                group: {"videos": len(records),
                        "frame_references": sum(len(x["frames"]) for x in records.values())}
                for group, records in test_records.items()
            },
            "unique_staged_pngs": len(planned),
            "selected_uncompressed_bytes": total_uncompressed,
            "archive_image_entries": {
                "e4s": len(e4s_images), "ffpp_real": len(real_images),
                "fsgan_ff": len(fsgan_images),
            },
            "excluded": ["FSGAN CDF images", "all .npy landmarks", "unreferenced PNGs"],
            "labels_from_config": {k: labels[k] for k in (
                "e4s_Real", "e4s_Fake", "fsgan_Real", "fsgan_Fake")},
        }

        plans = {
            "files": planned,
            "e4s_train": {"e4s_to_fsgan_train": {
                group: {"train": records} for group, records in split_records["train"].items()
            }},
            "e4s_val": {"e4s_to_fsgan_val": {
                group: {"test": records} for group, records in split_records["val"].items()
            }},
            "fsgan_test": {"fsgan_to_e4s_test": {
                group: {"test": records} for group, records in test_records.items()
            }},
            "preflight_train": {"DF40_FF_train_NOEFS": {
                group: {"train": records} for group, records in split_records["train"].items()
            }},
            "preflight_val": {"DF40_FF_train_NOEFS": {
                group: {"train": records} for group, records in split_records["val"].items()
            }},
            "preflight_test": {"DF40_FF_test_NOFE_all": {
                group: {"test": records} for group, records in test_records.items()
            }},
            "summary": summary,
        }
        return plans


def stage(plans, args):
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty; refusing to overwrite: {output_dir}")
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise ValueError(f"dataset root is not empty; refusing to overwrite existing data: {dataset_root}")
    if not dataset_root.parent.is_dir():
        raise ValueError(f"dataset root parent does not exist: {dataset_root.parent}")

    archive_handles = {}
    try:
        for zip_path, _ in plans["files"].values():
            archive_key = str(zip_path)
            if archive_key not in archive_handles:
                archive_handles[archive_key] = zipfile.ZipFile(zip_path)
        with tempfile.TemporaryDirectory(prefix=".dled_e4s_fsgan_stage_",
                                         dir=str(dataset_root.parent)) as temp_dir:
            stage_root = Path(temp_dir) / "dataset"
            stage_root.mkdir()
            decoded = 0
            for target_relative, (zip_path, info) in sorted(plans["files"].items()):
                archive = archive_handles[str(zip_path)]
                data = archive.read(info)
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (256, 256) or image.ndim != 3:
                    raise ValueError(f"image decode/shape failed for {info.filename}")
                target = stage_root / target_relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                decoded += 1

            if decoded != len(plans["files"]):
                raise ValueError(f"decoded {decoded} images, expected {len(plans['files'])}")
            if dataset_root.exists():
                dataset_root.rmdir()
            os.replace(stage_root, dataset_root)

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = output_dir / "manifests"
        manifest_dir.mkdir(exist_ok=True)
        manifest_files = {
            "e4s_to_fsgan_train.json": plans["e4s_train"],
            "e4s_to_fsgan_val.json": plans["e4s_val"],
            "fsgan_to_e4s_test.json": plans["fsgan_test"],
            "preflight_train.json": plans["preflight_train"],
            "preflight_val.json": plans["preflight_val"],
            "preflight_test.json": plans["preflight_test"],
        }
        for filename, data in manifest_files.items():
            (manifest_dir / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        summary = plans["summary"]
        summary["dataset_root"] = str(dataset_root)
        summary["output_dir"] = str(output_dir)
        summary["decoded_images"] = decoded
        summary["archive_paths"] = {
            "e4s": str(Path(args.e4s_archive).resolve()),
            "ffpp_real": str(Path(args.real_archive).resolve()),
            "fsgan": str(Path(args.fsgan_archive).resolve()),
        }
        (output_dir / "data_preflight.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        detector = yaml.safe_load(Path(args.detector_config).read_text(encoding="utf-8"))
        train_config = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8"))
        experiment = {
            "name": "E4S-to-FSGAN limited cross-method pilot",
            "seed": args.seed,
            "epochs": 5,
            "max_samples_per_class": 256,
            "num_classes": 2,
            "training_mode": "single",
            "checkpoint_selection": "highest E4S validation fused accuracy",
            "model": detector,
            "train_config": train_config,
            "data_preflight": summary,
        }
        (output_dir / "experiment_config.json").write_text(
            json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Staged and decoded {decoded} unique PNGs under {dataset_root}")
        print(f"Experiment manifests/config saved under {output_dir}")
    finally:
        for archive in archive_handles.values():
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e4s-archive", default="/Volumes/pendrive/DF40/incoming/e4s.zip")
    parser.add_argument("--real-archive", default="/Volumes/pendrive/DF40/incoming/FaceForensics++_real_data_for_DF40.zip")
    parser.add_argument("--fsgan-archive", default="/Volumes/pendrive/DF40/incoming/fsgan.zip")
    parser.add_argument("--train-protocol", default="datasets/DF40/json_file/DF40_FF_train_NOEFS.json")
    parser.add_argument("--test-protocol", default="datasets/DF40/json_file/DF40_FF_test_NOFE_all.json")
    parser.add_argument("--train-config", default="training/config/train_config.yaml")
    parser.add_argument("--detector-config", default="training/config/detector/dled.yaml")
    parser.add_argument("--dataset-root", default="/Volumes/pendrive/DF40/deepfakes_detection_datasets")
    parser.add_argument("--output-dir", default="results/e4s_to_fsgan_pilot")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="validate mapping and report the plan without writing")
    parser.add_argument("--stage", action="store_true", help="extract only planned PNGs and write isolated experiment manifests")
    args = parser.parse_args()
    if args.dry_run == args.stage:
        parser.error("choose exactly one of --dry-run or --stage")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    try:
        plans = build_plan(args)
        if args.dry_run:
            print(json.dumps(plans["summary"], indent=2))
            return 0
        stage(plans, args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
