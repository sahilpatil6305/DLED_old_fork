#!/usr/bin/env python3
"""Compare image paths in a ZIP archive with one method in a DLED protocol JSON."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def clean_path(value):
    """Normalize separators and harmless leading ./ without changing path roots."""
    path = str(value).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return re.sub(r"/{2,}", "/", path)


def frame_relative_path(value):
    """Return the path after the last frames/ component, if present."""
    parts = PurePosixPath(clean_path(value)).parts
    frame_positions = [i for i, part in enumerate(parts[:-1]) if part.casefold() == "frames"]
    if not frame_positions:
        return None
    return "/".join(parts[frame_positions[-1] + 1 :])


def load_protocol_refs(path, method, split):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("protocol JSON root must be an object")
    roots = [value for value in obj.values() if isinstance(value, dict)]
    if len(roots) != 1:
        raise ValueError("expected exactly one top-level protocol object")

    refs = []
    matched_groups = []
    method = method.casefold().strip()
    if not method:
        raise ValueError("method name must not be empty")
    method_pattern = re.compile(rf"(?<![a-z0-9]){re.escape(method)}(?![a-z0-9])")
    missing_split_groups = []
    for group, group_data in roots[0].items():
        if not method_pattern.search(str(group).casefold()):
            continue
        matched_groups.append(str(group))
        if not isinstance(group_data, dict):
            raise ValueError(f"protocol group {group!r} must be an object")
        if split and split not in group_data:
            missing_split_groups.append(str(group))
            continue
        for split_name, videos in group_data.items():
            if split and split_name != split:
                continue
            if not isinstance(videos, dict):
                raise ValueError(f"protocol group {group!r} split {split_name!r} must be an object")
            for video, info in videos.items():
                if not isinstance(info, dict):
                    raise ValueError(f"protocol video {group}/{split_name}/{video} must be an object")
                frames = info.get("frames", [])
                if not isinstance(frames, list):
                    raise ValueError(f"{group}/{split_name}/{video}.frames must be a list")
                refs.extend((str(frame), str(group), str(video)) for frame in frames)
    return refs, matched_groups, missing_split_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", help="ZIP archive to inspect (never extracted)")
    parser.add_argument("method", help="method name used in protocol group names, e.g. fsgan")
    parser.add_argument("protocol_json", help="protocol JSON to compare")
    parser.add_argument("--split", help="optional protocol split to inspect, e.g. train or test")
    parser.add_argument("--examples", type=int, default=5, help="unmatched examples to show (default: 5)")
    args = parser.parse_args()

    try:
        archive_path = Path(args.zip_path).expanduser()
        protocol_path = Path(args.protocol_json).expanduser()
        with zipfile.ZipFile(archive_path) as archive:
            archive_image_entries = [
                clean_path(info.filename)
                for info in archive.infolist()
                if not info.is_dir() and PurePosixPath(info.filename).suffix.casefold() in IMAGE_SUFFIXES
            ]
        archive_images = set(archive_image_entries)
        refs, groups, missing_split_groups = load_protocol_refs(protocol_path, args.method, args.split)
        if not groups:
            raise ValueError(f"no protocol groups matched method {args.method!r}")
        if missing_split_groups:
            raise ValueError(f"requested split {args.split!r} is absent from method groups: "
                             + ", ".join(missing_split_groups))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    archive_relative = {key: path for path in archive_images if (key := frame_relative_path(path)) is not None}
    protocol_paths = {clean_path(path) for path, _, _ in refs}
    protocol_path_relative = {path: frame_relative_path(path) for path in protocol_paths}
    protocol_relative = {key: path for path, key in protocol_path_relative.items() if key is not None}
    exact = archive_images & protocol_paths
    relative_matches = set(archive_relative) & set(protocol_relative)
    prefix_only = [key for key in sorted(relative_matches)
                   if archive_relative[key] != protocol_relative[key]]
    suffix_matched_protocol = {
        path for path, key in protocol_path_relative.items() if key is not None and key in archive_relative
    }
    suffix_matched_archive = {
        path for path in archive_images
        if (key := frame_relative_path(path)) is not None and key in protocol_relative
    }
    unmatched_protocol = sorted(protocol_paths - exact - suffix_matched_protocol)
    unmatched_archive = sorted(archive_images - exact - suffix_matched_archive)
    archive_basenames = {PurePosixPath(path).name.casefold() for path in archive_images}

    print(f"Archive: {archive_path}")
    print(f"Protocol: {protocol_path}")
    print(f"Method groups: {', '.join(groups) if groups else '(none)'}")
    print(f"Split filter: {args.split or 'all'}")
    duplicate_entries = len(archive_image_entries) - len(archive_images)
    print(f"Archive image entries: {len(archive_image_entries)} ({len(archive_images)} unique paths; "
          f"{duplicate_entries} duplicate entries)")
    print(f"Protocol frame references: {len(refs)} ({len(protocol_paths)} unique paths)")
    print(f"Exact full-path matches: {len(exact)} unique paths")
    print(f"Same path after frames/ (prefix/layout-only difference): {len(relative_matches)} unique paths")
    no_frames_protocol = sum(key is None for key in protocol_path_relative.values())
    no_frames_archive = sum(frame_relative_path(path) is None for path in archive_images)
    print(f"Paths without a frames/ component (not suffix-comparable): protocol={no_frames_protocol}, "
          f"archive={no_frames_archive}")
    print(f"Protocol paths unmatched by archive: {len(unmatched_protocol)} unique paths")
    same_name = sum(frame_relative_path(path) is not None and
                    PurePosixPath(path).name.casefold() in archive_basenames
                    for path in unmatched_protocol)
    print(f"  same frame filename exists elsewhere in archive (different parent/video path): {same_name}")
    print(f"  frame filename absent from archive: {len(unmatched_protocol) - same_name}")
    print(f"Archive paths unmatched by protocol: {len(unmatched_archive)} unique paths")

    if prefix_only:
        print("Prefix/layout-only examples (archive path => protocol path):")
        for key in prefix_only[:args.examples]:
            print(f"  {archive_relative[key]} => {protocol_relative[key]}")
    if unmatched_protocol:
        print("Unmatched protocol examples:")
        for path in unmatched_protocol[:args.examples]:
            if frame_relative_path(path) is None:
                kind = "no frames/ component"
            else:
                kind = "same filename elsewhere" if PurePosixPath(path).name.casefold() in archive_basenames else "filename absent"
            print(f"  [{kind}] {path}")
    if unmatched_archive:
        print("Unmatched archive examples:")
        for path in unmatched_archive[:args.examples]:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
