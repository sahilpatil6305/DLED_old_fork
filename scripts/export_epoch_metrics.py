#!/usr/bin/env python3
"""Join DLED's training log and per-epoch validation JSON into CSV."""

import argparse
import ast
import csv
import json
from pathlib import Path
import re


def parse_training_log(path):
    result = {}
    pattern = re.compile(r"Epoch\[(\d+)\] training (loss|metric) means: (\{.*\})")
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        epoch, kind, payload = match.groups()
        try:
            # Python logging formats unavailable metrics as nan/inf, which
            # ast.literal_eval rejects. Preserve finite values such as train
            # accuracy and store those non-finite metrics as empty cells.
            safe_payload = re.sub(
                r"(?<![\w.])[+-]?(?:nan|inf)(?![\w.])",
                "None",
                payload,
                flags=re.IGNORECASE,
            )
            values = ast.literal_eval(safe_payload)
        except (ValueError, SyntaxError):
            continue
        target = result.setdefault(int(epoch), {})
        for key, value in values.items():
            if isinstance(value, (int, float)):
                target[f"train_{kind}_{key}"] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = parse_training_log(args.training_log)
    val_dir = Path(args.validation_dir)
    for path in sorted(val_dir.glob("metrics_epoch_*.json")):
        try:
            epoch = int(path.stem.rsplit("_", 1)[1])
            report = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            continue
        row = rows.setdefault(epoch, {})
        row["epoch"] = epoch
        for key, value in report.items():
            if isinstance(value, (int, float)) and key != "epoch":
                row[f"val_{key}"] = value
            elif key == "losses" and isinstance(value, dict):
                for loss_name, loss_value in value.items():
                    if isinstance(loss_value, (int, float)):
                        row[f"val_loss_{loss_name}"] = loss_value

    if not rows:
        raise SystemExit("No epoch training logs or validation metric JSONs were found.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch"] + sorted({key for row in rows.values() for key in row if key != "epoch"})
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in sorted(rows):
            row = dict(rows[epoch])
            row["epoch"] = epoch
            writer.writerow(row)
    print(f"Wrote {len(rows)} epoch rows to {output}")


if __name__ == "__main__":
    main()
