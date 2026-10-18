#!/usr/bin/env python3
"""Create a source-video-disjoint FSGAN pilot train/validation index.

The script reads the existing fsgan_ff manifest, uses only its official
training partition, verifies every referenced frame exists, then writes two
new indices. The source manifest and images are never modified.
"""

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys


REAL_GROUP = "fsgan_Real"
FAKE_GROUP = "fsgan_Fake"


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
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def source_ids(group, video):
    if group == REAL_GROUP:
        match = re.fullmatch(r"(\d+)_fsgan", video, re.IGNORECASE)
        if match:
            return (match.group(1),)
    elif group == FAKE_GROUP:
        match = re.fullmatch(r"(\d+)_(\d+)_fsgan", video, re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
    raise ValueError(f"unrecognized {group} video key {video!r}; cannot split safely by source video")


def resolve_frame(frame, data_root, old_root):
    resolved = str(frame)
    if data_root and old_root:
        resolved = resolved.replace(old_root, str(data_root))
    return Path(resolved).expanduser()


def make_split(groups, val_fraction, seed):
    union_find = UnionFind()
    identities = {}

    for group in (REAL_GROUP, FAKE_GROUP):
        for video, info in groups[group]["train"].items():
            ids = source_ids(group, video)
            identities[(group, video)] = ids
            for source_id in ids:
                union_find.find(source_id)
            if group == FAKE_GROUP:
                union_find.union(*ids)

    component_ids = sorted({union_find.find(source_id) for source_id in union_find.parent})
    if len(component_ids) < 2:
        raise ValueError("at least two independent source-video groups are required for validation")
    random.Random(seed).shuffle(component_ids)
    val_count = max(1, min(len(component_ids) - 1, round(len(component_ids) * val_fraction)))
    val_components = set(component_ids[:val_count])

    split_groups = {
        "train": {REAL_GROUP: {}, FAKE_GROUP: {}},
        "val": {REAL_GROUP: {}, FAKE_GROUP: {}},
    }
    split_sources = {"train": set(), "val": set()}
    for (group, video), ids in identities.items():
        root = union_find.find(ids[0])
        split_name = "val" if root in val_components else "train"
        split_groups[split_name][group][video] = groups[group]["train"][video]
        split_sources[split_name].update(ids)

    if split_sources["train"] & split_sources["val"]:
        raise ValueError("source-video leakage detected between train and validation")
    for split_name, split_data in split_groups.items():
        if any(not split_data[group] for group in (REAL_GROUP, FAKE_GROUP)):
            raise ValueError(f"{split_name} split does not contain both real and fake videos")
    return split_groups, split_sources


def count_split(split_data):
    result = {}
    for group, videos in split_data.items():
        result[group] = {
            "videos": len(videos),
            "frame_references": sum(len(info.get("frames", [])) for info in videos.values()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", default="dataset_json/fsgan_ff.json")
    parser.add_argument("--dataset-root", default=os.getenv("DF40_DATA_ROOT"),
                        help="extracted data root (defaults to DF40_DATA_ROOT)")
    parser.add_argument("--old-root", default=os.getenv("DF40_OLD_ROOT", "deepfakes_detection_datasets"),
                        help="path prefix embedded in the source JSON")
    parser.add_argument("--output-dir", default="outputs/fsgan_pilot/manifests")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--missing-limit", type=int, default=10)
    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if not args.dataset_root:
        parser.error("set DF40_DATA_ROOT or pass --dataset-root")

    try:
        source_path = Path(args.source_json).expanduser()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(source, dict) or not isinstance(source.get("fsgan_ff"), dict):
            raise ValueError("source JSON must contain top-level object 'fsgan_ff'")
        groups = source["fsgan_ff"]
        if any(group not in groups or "train" not in groups[group] for group in (REAL_GROUP, FAKE_GROUP)):
            raise ValueError("source JSON must include fsgan_Real/train and fsgan_Fake/train")

        split_groups, split_sources = make_split(groups, args.val_fraction, args.seed)
        missing = []
        missing_count = 0
        for split_data in split_groups.values():
            for videos in split_data.values():
                for info in videos.values():
                    for frame in info.get("frames", []):
                        if not resolve_frame(frame, args.dataset_root, args.old_root).is_file():
                            missing_count += 1
                            if len(missing) < args.missing_limit:
                                missing.append(str(resolve_frame(frame, args.dataset_root, args.old_root)))
        if missing_count:
            print(f"PILOT PREPARATION BLOCKED: {missing_count} training frame references are missing.", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            print("No split manifests were written. Install the authorized FSGAN FF training frames and FF++ originals, then rerun.", file=sys.stderr)
            return 1

        summary = {
            "source_json": str(source_path),
            "dataset_root": str(Path(args.dataset_root).expanduser()),
            "seed": args.seed,
            "validation_fraction_target": args.val_fraction,
            "source_video_disjoint": True,
            "frame_counts_are_references_not_decoded_unique_images": True,
            "train": count_split(split_groups["train"]),
            "validation": count_split(split_groups["val"]),
            "source_video_ids": {key: sorted(value) for key, value in split_sources.items()},
        }
        output_dir = Path(args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        train_index = {"fsgan_pilot_train": {
            group: {"train": split_groups["train"][group]} for group in (REAL_GROUP, FAKE_GROUP)
        }}
        val_index = {"fsgan_pilot_val": {
            group: {"test": split_groups["val"][group]} for group in (REAL_GROUP, FAKE_GROUP)
        }}
        (output_dir / "fsgan_pilot_train.json").write_text(json.dumps(train_index, indent=2) + "\n", encoding="utf-8")
        (output_dir / "fsgan_pilot_val.json").write_text(json.dumps(val_index, indent=2) + "\n", encoding="utf-8")
        (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({key: summary[key] for key in ("seed", "validation_fraction_target", "train", "validation")}, indent=2))
    print(f"Wrote verified split manifests to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
