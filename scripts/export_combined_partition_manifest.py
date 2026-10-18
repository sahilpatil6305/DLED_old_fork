#!/usr/bin/env python3
"""Validate combined split manifests and export per-frame provenance as JSONL."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import re
import sys


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def source_ids(group, video):
    if group.endswith("_Real"):
        match = re.match(r"^(\d+)(?:_|$)", video)
    elif group == "e4s_Fake":
        match = re.match(r"^(\d+)_(\d+)_e4s$", video)
    elif group == "fsgan_Fake":
        match = re.match(r"^(\d+)_(\d+)(?:_|$)", video)
    else:
        raise ValueError(f"Unexpected protocol group: {group}")
    if not match:
        raise ValueError(f"Cannot parse source IDs from {group}/{video}")
    return set(match.groups())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--dataset-root", default=os.environ.get(
        "DF40_DATA_ROOT", "/Volumes/pendrive/DF40/deepfakes_detection_datasets"))
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()
    split_dir = Path(args.split_dir).expanduser().resolve(strict=True)
    root = Path(args.dataset_root).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    partitions = ("train", "validation", "test")
    filenames = {"train": "combined_A_train.json",
                 "validation": "combined_A_validation.json",
                 "test": "combined_A_test.json"}
    parsed = {}
    uf = UnionFind()
    rows = []
    for partition in partitions:
        payload = json.loads((split_dir / filenames[partition]).read_text(encoding="utf-8"))
        if len(payload) != 1:
            raise ValueError(f"Expected one protocol root in {filenames[partition]}")
        groups = next(iter(payload.values()))
        parsed[partition] = groups
        for group, split_obj in groups.items():
            if set(split_obj) != {"train" if partition == "train" else "test"}:
                raise ValueError(f"Unexpected split key for {partition}/{group}")
            videos = next(iter(split_obj.values()))
            for video, info in videos.items():
                ids = source_ids(group, video)
                for source_id in ids:
                    uf.find(source_id)
                if group.endswith("_Fake"):
                    left, right = sorted(ids)
                    uf.union(left, right)
                rows.append((partition, group, video, info, ids))

    summaries = {p: {"frames": 0, "videos_by_method": defaultdict(set),
                     "frames_by_method": defaultdict(int), "groups": set(),
                     "class_frames": defaultdict(int)} for p in partitions}
    seen_paths = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for partition, group, video, info, ids in rows:
            label = 0 if group.endswith("_Real") else 1
            method = "real" if label == 0 else group.removesuffix("_Fake")
            source_dataset = "FaceForensics++" if label == 0 else "DF40 " + method
            group_id = "source_group:" + "+".join(sorted(
                {uf.find(source_id) for source_id in ids}))
            summary = summaries[partition]
            summary["groups"].add(group_id)
            summary["videos_by_method"][method].add(video)
            summary["class_frames"]["real" if label == 0 else "fake"] += len(info["frames"])
            summary["frames_by_method"][method] += len(info["frames"])
            for path in info["frames"]:
                if path in seen_paths and seen_paths[path] != partition:
                    raise ValueError(f"Frame reference crosses partitions: {path}")
                seen_paths[path] = partition
                resolved = Path(path.replace("deepfakes_detection_datasets", str(root)))
                if not resolved.is_file():
                    raise ValueError(f"Missing frame: {resolved}")
                record = {"path": path, "resolved_path": str(resolved), "label": label,
                          "source_dataset": source_dataset, "method": method,
                          "source_video_ids": sorted(ids), "source_video_group_id": group_id,
                          "partition": partition, "protocol_group": group,
                          "protocol_video_id": video}
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                summary["frames"] += 1

    compact = {}
    for partition in partitions:
        s = summaries[partition]
        compact[partition] = {
            "source_video_groups": len(s["groups"]), "video_rows_by_method": {
                method: len(videos) for method, videos in sorted(s["videos_by_method"].items())},
            "frame_references": s["frames"], "frames_by_method": dict(s["frames_by_method"]),
            "class_frames": dict(s["class_frames"]),
        }
    source_group_sets = {p: summaries[p]["groups"] for p in partitions}
    overlaps = {"train_validation": len(source_group_sets["train"] & source_group_sets["validation"]),
                "train_test": len(source_group_sets["train"] & source_group_sets["test"]),
                "validation_test": len(source_group_sets["validation"] & source_group_sets["test"])}
    if any(overlaps.values()):
        raise ValueError(f"Source group leakage detected: {overlaps}")
    report = {"split_dir": str(split_dir), "dataset_root": str(root),
              "unique_frame_references": len(seen_paths), "missing_paths": 0,
              "source_group_overlaps": overlaps, "partitions": compact,
              "proportions_by_source_group": {
                  p: len(summaries[p]["groups"]) / sum(len(x["groups"]) for x in summaries.values())
                  for p in partitions},
              "proportions_by_frame_reference": {
                  p: summaries[p]["frames"] / sum(x["frames"] for x in summaries.values())
                  for p in partitions}, "per_frame_manifest": str(output)}
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Per-frame manifest: {output}")
    print(f"Summary: {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"PARTITION AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
