#!/usr/bin/env python3
"""Validate the selected DLED protocol JSON and referenced image files."""
import argparse
import json
import os
from pathlib import Path
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=['NOEFS', 'NOFS', 'NOFR', 'open'], default='NOEFS')
    parser.add_argument('--dataset-json-folder', default=os.getenv('DLED_DATASET_JSON_DIR', './datasets/DF40/json_file'))
    parser.add_argument('--train-json', default=None, help='Override train protocol JSON path')
    parser.add_argument('--test-json', default=None, help='Override test protocol JSON path')
    parser.add_argument('--sample-limit', type=int, default=20, help='Maximum missing paths shown')
    args = parser.parse_args()
    names = {
        'NOEFS': ('DF40_FF_train_NOEFS', 'DF40_FF_test_NOFE_all'),
        'NOFS': ('DF40_FF_train_NOFS', 'DF40_FF_test_NOFE_all'),
        'NOFR': ('DF40_FF_train_NOFR', 'DF40_FF_test_NOFE_all'),
        'open': ('DF40_FF_train', 'DF40_open_test'),
    }
    train_name, test_name = names[args.protocol]
    folder = Path(args.dataset_json_folder).expanduser()
    paths = {
        train_name: Path(args.train_json).expanduser() if args.train_json else folder / f'{train_name}.json',
        test_name: Path(args.test_json).expanduser() if args.test_json else folder / f'{test_name}.json',
    }
    config_path = Path(__file__).resolve().parent / 'config/train_config.yaml'
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    mapping = cfg.get('label_dict', {})
    errors, missing, summaries = [], [], []
    for dataset_name, path in paths.items():
        if not path.is_file():
            errors.append(f"Missing {dataset_name} protocol JSON: {path}")
            continue
        try:
            obj = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            errors.append(f"Invalid JSON {path}: {exc}")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get(dataset_name), dict):
            errors.append(f"{path} must contain a top-level object key '{dataset_name}'")
            continue
        split = 'train' if dataset_name == train_name else 'test'
        videos = frames = empty_videos = 0
        labels = set()
        for group, group_data in obj[dataset_name].items():
            if not isinstance(group_data, dict) or not isinstance(group_data.get(split), dict):
                errors.append(f"{path}: '{group}' must contain object '{split}'")
                continue
            data = group_data[split]
            # This protocol stores direct video entries, unlike FF++ compression subtrees.
            for video, info in data.items():
                if not isinstance(info, dict):
                    errors.append(f"{path}: {group}/{video} must be a video object")
                    continue
                label = info.get('label')
                frame_list = info.get('frames')
                if label not in mapping:
                    errors.append(f"{path}: label {label!r} for {group}/{video} is absent from label_dict")
                else:
                    labels.add(mapping[label])
                if not isinstance(frame_list, list):
                    errors.append(f"{path}: {group}/{video}.frames must be a list")
                    continue
                if not frame_list:
                    empty_videos += 1
                    continue
                videos += 1
                frames += len(frame_list)
                old_root, new_root = os.getenv('DF40_OLD_ROOT'), os.getenv('DF40_DATA_ROOT')
                for frame in frame_list:
                    resolved = str(frame).replace(old_root, new_root) if old_root and new_root else str(frame)
                    if not Path(resolved).expanduser().is_file():
                        missing.append(resolved)
        summaries.append(
            f"{dataset_name}: {videos} usable videos, {frames} frame paths, "
            f"{empty_videos} empty videos skipped, mapped labels={sorted(labels)}"
        )
        if videos == 0 or frames == 0:
            errors.append(f"{path}: no usable {split} samples found")
    print(f"Protocol: {args.protocol} (full protocol definition retained)")
    for line in summaries:
        print(line)
    if errors or missing:
        print('\nPREFLIGHT FAILED', file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        if missing:
            print(f"- {len(missing)} referenced frame paths are missing; first {args.sample_limit}:", file=sys.stderr)
            for path in missing[:args.sample_limit]:
                print(f"  {path}", file=sys.stderr)
            print("Set DF40_DATA_ROOT (and, when needed, DF40_OLD_ROOT) or correct the JSON paths.", file=sys.stderr)
        sys.exit(1)
    print('PREFLIGHT PASSED: JSON schema, sample presence, labels, and frame paths are valid.')


if __name__ == '__main__':
    main()
