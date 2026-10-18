#!/usr/bin/env python3
"""Validate the staged E4S train/validation and FSGAN held-out test manifests."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import yaml


def load(path, key):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get(key), dict):
        raise ValueError(f"{path} must contain {key!r}")
    return obj[key]


def source_ids(group, video):
    if group.endswith("_Real"):
        match = re.match(r"^(\d+)(?:_|$)", video)
        return {match.group(1)} if match else set()
    if group.endswith("_Fake"):
        match = re.match(r"^(\d+)_(\d+)(?:_|$)", video)
        return set(match.groups()) if match else set()
    raise ValueError(f"unexpected group name {group}")


def inspect_manifest(path, key, split, data_root, expected_labels):
    groups = load(path, key)
    report = {"videos": 0, "frame_references": 0, "missing_paths": 0,
              "missing_examples": [], "class_counts": {}, "video_counts": {}}
    sources = set()
    for group, split_map in groups.items():
        if split not in split_map or not isinstance(split_map[split], dict):
            raise ValueError(f"{path}: {group} missing {split} video object")
        for video, info in split_map[split].items():
            label = info.get("label")
            if label not in expected_labels:
                raise ValueError(f"{path}: invalid label {label!r} for {group}/{video}")
            expected_class = 0 if group.endswith("_Real") else 1
            if expected_labels[label] != expected_class:
                raise ValueError(f"{path}: {label} has inconsistent binary class mapping")
            frames = info.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"{path}: {group}/{video} has no frame references")
            report["videos"] += 1
            report["frame_references"] += len(frames)
            report["class_counts"]["real" if expected_class == 0 else "fake"] = (
                report["class_counts"].get("real" if expected_class == 0 else "fake", 0) + len(frames)
            )
            class_name = "real" if expected_class == 0 else "fake"
            report["video_counts"][class_name] = report["video_counts"].get(class_name, 0) + 1
            sources.update(source_ids(group, video))
            for frame in frames:
                resolved = str(frame).replace(os.getenv("DF40_OLD_ROOT", "deepfakes_detection_datasets"),
                                              str(data_root))
                if not Path(resolved).is_file():
                    report["missing_paths"] += 1
                    if len(report["missing_examples"]) < 20:
                        report["missing_examples"].append(resolved)
    if set(report["class_counts"]) != {"real", "fake"}:
        raise ValueError(f"{path} must contain both real and fake labels")
    return report, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="results/e4s_to_fsgan_pilot/manifests")
    parser.add_argument("--dataset-root", default="/Volumes/pendrive/DF40/deepfakes_detection_datasets")
    parser.add_argument("--train-config", default="training/config/train_config.yaml")
    parser.add_argument("--output", default="results/e4s_to_fsgan_pilot/pilot_preflight.json")
    args = parser.parse_args()
    try:
        cfg = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8"))
        labels = cfg["label_dict"]
        base = Path(args.manifest_dir)
        train, train_sources = inspect_manifest(
            base / "e4s_to_fsgan_train.json", "e4s_to_fsgan_train", "train",
            args.dataset_root, labels)
        val, val_sources = inspect_manifest(
            base / "e4s_to_fsgan_val.json", "e4s_to_fsgan_val", "test",
            args.dataset_root, labels)
        test, test_sources = inspect_manifest(
            base / "fsgan_to_e4s_test.json", "fsgan_to_e4s_test", "test",
            args.dataset_root, labels)
        if train_sources & val_sources or train_sources & test_sources or val_sources & test_sources:
            raise ValueError("source-video ID leakage between train, validation, and test")
        missing = sum(row["missing_paths"] for row in (train, val, test))
        report = {
            "train": train, "validation": val, "test": test,
            "source_video_id_counts": {
                "train": len(train_sources), "validation": len(val_sources), "test": len(test_sources),
            },
            "source_video_id_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
            "missing_paths": missing,
            "passed": missing == 0,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if missing:
            raise ValueError("one or more referenced files are missing; see missing_path_examples")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
