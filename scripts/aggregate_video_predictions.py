#!/usr/bin/env python3
"""Aggregate saved DLED frame predictions by manifest video directory."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        paths = data["image_path"].astype(str)
        labels = data["binary_labels"].astype(int)
        scores = data["fused_fake_probability"].astype(float)
    videos = {}
    for path, label, score in zip(paths, labels, scores):
        p = Path(path)
        parts = p.parts
        if "frames" not in parts:
            raise ValueError(f"prediction path has no frames directory: {path}")
        idx = len(parts) - 1 - parts[::-1].index("frames")
        parent = parts[idx + 1]
        if "DF40_train" in parts and "e4s" in parts:
            group = "e4s_Fake"
            method = "e4s"
        elif "DF40" in parts and "fsgan" in parts:
            group = "fsgan_Fake"
            method = "fsgan"
        else:
            group = "real"
            method = "real"
        key = f"{group}:{parent}"
        row = videos.setdefault(key, {"label": int(label), "method": method, "scores": [], "frames": 0})
        if row["label"] != int(label):
            raise ValueError(f"inconsistent labels for video group {key}")
        row["scores"].append(float(score))
        row["frames"] += 1

    if not videos:
        raise ValueError("no video predictions were found")
    y_true = np.asarray([row["label"] for row in videos.values()], dtype=int)
    y_score = np.asarray([np.mean(row["scores"]) for row in videos.values()], dtype=float)
    y_pred = (y_score >= 0.5).astype(int)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    report = {
        "aggregation": "mean frame fake probability per video, decision threshold 0.5",
        "videos": len(videos),
        "frame_count": int(sum(row["frames"] for row in videos.values())),
        "video_counts": {"real": int(np.sum(y_true == 0)), "fake": int(np.sum(y_true == 1))},
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision),
        "recall_fake": float(recall),
        "f1_fake": float(f1),
        "roc_auc_fake_score": (float(metrics.roc_auc_score(y_true, y_score))
                                if len(np.unique(y_true)) == 2 else None),
        "confusion_matrix_real_fake": metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "per_method_video_counts": {
            method: {"videos": sum(row["method"] == method for row in videos.values()),
                     "frames": sum(row["frames"] for row in videos.values() if row["method"] == method)}
            for method in sorted({row["method"] for row in videos.values()})
        },
        "per_video_predictions": [
            {"group": key, "method": row["method"], "label": row["label"],
             "mean_fake_probability": float(np.mean(row["scores"])), "frames": row["frames"]}
            for key, row in sorted(videos.items())
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_video_predictions"}, indent=2))


if __name__ == "__main__":
    main()
