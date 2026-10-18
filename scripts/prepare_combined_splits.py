#!/usr/bin/env python3
"""Create source-video-disjoint 70/15/15 combined E4S/FSGAN/FF++ manifests."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import sys


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_splits(manifest, name, groups, split_names):
    result = []
    for split_name in split_names:
        for group in groups:
            rows = manifest[name][group][split_name]
            for video, info in rows.items():
                result.append((group, video, info))
    return result


def source_ids(group, video):
    if group.endswith("_Real"):
        match = re.match(r"^(\d+)(?:_|$)", video)
        if not match:
            raise ValueError(f"cannot derive source-video ID from {group}/{video}")
        return {match.group(1)}
    if group == "e4s_Fake":
        match = re.match(r"^(\d+)_(\d+)_e4s$", video)
    elif group == "fsgan_Fake":
        match = re.match(r"^(\d+)_(\d+)(?:_|$)", video)
    else:
        raise ValueError(f"unsupported class group: {group}")
    if not match:
        raise ValueError(f"cannot parse paired source-video IDs from {group}/{video}")
    return set(match.groups())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e4s-manifest", default="results/e4s_to_fsgan_pilot/manifests/e4s_to_fsgan_train.json")
    parser.add_argument("--validation-manifest", default="results/e4s_to_fsgan_pilot/manifests/e4s_to_fsgan_val.json")
    parser.add_argument("--fsgan-manifest", default="results/e4s_to_fsgan_pilot/manifests/fsgan_to_e4s_test.json")
    parser.add_argument("--dataset-root", default="/Volumes/pendrive/DF40/deepfakes_detection_datasets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    args = parser.parse_args()
    try:
        if abs(sum(args.fractions) - 1.0) > 1e-8 or any(x <= 0 for x in args.fractions):
            raise ValueError("split fractions must be positive and sum to 1")
        e4s = read(args.e4s_manifest)
        e4s_validation = read(args.validation_manifest)
        fsgan = read(args.fsgan_manifest)
        rows = get_splits(e4s, "e4s_to_fsgan_train", ["e4s_Real", "e4s_Fake"], ["train"])
        rows.extend(get_splits(e4s_validation, "e4s_to_fsgan_val", ["e4s_Real", "e4s_Fake"], ["test"]))
        rows.extend(get_splits(fsgan, "fsgan_to_e4s_test", ["fsgan_Real", "fsgan_Fake"], ["test"]))

        uf = UnionFind()
        row_sources = {}
        for group, video, info in rows:
            ids = source_ids(group, video)
            row_sources[(group, video)] = ids
            for source_id in ids:
                uf.find(source_id)
            if group.endswith("_Fake"):
                left, right = sorted(ids)
                uf.union(left, right)

        components = {}
        for source_id in uf.parent:
            components.setdefault(uf.find(source_id), set()).add(source_id)
        component_rows = {root: [] for root in components}
        for row in rows:
            group, video, _ = row
            component_rows[uf.find(next(iter(row_sources[(group, video)])))].append(row)

        names = ("train", "validation", "test")
        fractions = dict(zip(names, args.fractions))
        totals = {name: {"real": 0, "fake": 0, "rows": 0} for name in names}
        ordered = list(components)
        random.Random(args.seed).shuffle(ordered)
        ordered.sort(key=lambda root: -len(component_rows[root]))
        assigned = {name: [] for name in names}
        target_totals = {
            cls: sum(1 for group, _, _ in rows if group.endswith("_Real") == (cls == "real"))
            for cls in ("real", "fake")
        }
        for root in ordered:
            members = component_rows[root]
            class_load = {
                cls: sum(1 for group, _, _ in members if group.endswith("_Real") == (cls == "real"))
                for cls in ("real", "fake")
            }
            candidates = []
            for split in names:
                fill = []
                for cls in ("real", "fake"):
                    denom = max(1, target_totals[cls] * fractions[split])
                    projected = totals[split][cls] + class_load[cls]
                    fill.append(projected / denom)
                # Allocate components in proportion to each target partition,
                # while mildly penalizing class imbalance within a partition.
                score = max(fill) + 0.25 * abs(fill[0] - fill[1])
                candidates.append((score, split))
            split = min(candidates)[1]
            assigned[split].extend(members)
            totals[split]["rows"] += len(members)
            totals[split]["real"] += class_load["real"]
            totals[split]["fake"] += class_load["fake"]

        groups = ("e4s_Real", "e4s_Fake", "fsgan_Real", "fsgan_Fake")
        split_records = {split: {group: {} for group in groups} for split in names}
        split_sources = {split: set() for split in names}
        all_paths = set()
        path_splits = {}
        missing = []
        for split in names:
            for group, video, info in assigned[split]:
                split_records[split][group][video] = info
                split_sources[split].update(row_sources[(group, video)])
                for frame in info.get("frames", []):
                    if frame in all_paths:
                        if path_splits[frame] != split:
                            raise ValueError(f"frame path crosses split boundary: {frame}")
                    all_paths.add(frame)
                    path_splits[frame] = split
                    resolved = Path(frame.replace("deepfakes_detection_datasets", args.dataset_root))
                    if not resolved.is_file():
                        missing.append(str(resolved))

        overlaps = {
            "train_validation": sorted(split_sources["train"] & split_sources["validation"]),
            "train_test": sorted(split_sources["train"] & split_sources["test"]),
            "validation_test": sorted(split_sources["validation"] & split_sources["test"]),
        }
        if any(overlaps.values()):
            raise ValueError(f"source-video groups leak across splits: {overlaps}")
        for split in names:
            class_rows = {"real": 0, "fake": 0}
            class_frames = {"real": 0, "fake": 0}
            for group_name, videos in split_records[split].items():
                cls = "real" if group_name.endswith("_Real") else "fake"
                class_rows[cls] += len(videos)
                class_frames[cls] += sum(len(info.get("frames", [])) for info in videos.values())
            if not all(class_rows.values()):
                raise ValueError(f"{split} has no examples of both classes: {class_rows}")
            totals[split]["class_video_rows"] = class_rows
            totals[split]["class_frame_references"] = class_frames
        if missing:
            raise ValueError(f"{len(missing)} referenced files are missing, examples: {missing[:10]}")

        out = Path(args.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=False)
        train_json = {"combined_A_train": {group: {"train": split_records["train"][group]} for group in groups}}
        validation_json = {"combined_A_validation": {group: {"test": split_records["validation"][group]} for group in groups}}
        test_json = {"combined_A_test": {group: {"test": split_records["test"][group]} for group in groups}}
        for filename, obj in (("combined_A_train.json", train_json),
                              ("combined_A_validation.json", validation_json),
                              ("combined_A_test.json", test_json)):
            (out / filename).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
        summary = {
            "experiment": "Experiment A: combined seen-method evaluation",
            "seed": args.seed,
            "fractions_target": dict(zip(names, args.fractions)),
            "source_video_group_counts": {s: len(split_sources[s]) for s in names},
            "connected_component_counts": {s: len({uf.find(x) for x in split_sources[s]}) for s in names},
            "source_video_overlaps": {k: len(v) for k, v in overlaps.items()},
            "overlap_examples": overlaps,
            "split_video_row_counts": {s: totals[s] for s in names},
            "method_counts": {
                split: {
                    group: {
                        "videos": len(split_records[split][group]),
                        "frame_references": sum(len(info.get("frames", []))
                                                 for info in split_records[split][group].values()),
                    }
                    for group in groups
                }
                for split in names
            },
            "unique_frame_references": len(all_paths),
            "missing_paths": 0,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "input_manifest_sha256": {str(Path(p).resolve()): sha256(p) for p in (args.e4s_manifest, args.validation_manifest, args.fsgan_manifest)},
        }
        (out / "split_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"SPLIT PREPARATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
