"""Small append-only run tracking shared by DLED training and evaluation."""
from __future__ import annotations

import csv
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import contextlib
import hashlib
import sys
import time
import uuid

import torch


INDEX_FIELDS = [
    "run_id", "timestamp", "event", "experiment_type", "dataset_combination",
    "config_id", "status", "completed_epochs", "best_validation_metric",
    "test_accuracy", "test_roc_auc", "runtime_seconds", "artifact_dir",
]


def create_run(experiment_type, dataset_combination, command, output_root="results",
               config_id="", metadata=None):
    """Create a unique run directory and append its initial RUNNING event."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    base = Path(output_root).expanduser().resolve() / experiment_type / run_id
    base.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "checkpoints", "predictions"):
        (base / name).mkdir()
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    record = {
        "run_id": run_id,
        "timestamp": now,
        "experiment_type": experiment_type,
        "dataset_combination": dataset_combination,
        "config_id": config_id,
        "status": "RUNNING",
        "completed_epochs": 0,
        "best_validation_metric": "",
        "test_accuracy": "",
        "test_roc_auc": "",
        "runtime_seconds": "",
        "artifact_dir": str(base),
        "start_time": now,
        "end_time": None,
        "command": command,
        "parent_run_id": None,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dependencies": {},
    }
    for package in ("numpy", "Pillow", "opencv-python", "scikit-learn", "PyYAML", "torchvision"):
        try:
            record["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    if metadata:
        record.update(metadata)
    (base / "run_metadata.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    append_event(base.parent.parent / "experiment_index.csv", record, "STARTED")
    return RunRecord(base, record)


class RunRecord:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.metadata = metadata
        self._started = time.monotonic()

    def update(self, **values):
        self.metadata.update(values)
        self._save_event("UPDATED")

    def finish(self, status, **values):
        self.metadata.update(values)
        self.metadata["status"] = status
        end_time = dt.datetime.now(dt.timezone.utc)
        self.metadata["end_time"] = end_time.isoformat()
        self.metadata["runtime_seconds"] = round(time.monotonic() - self._started, 3)
        try:
            start_time = dt.datetime.fromisoformat(self.metadata["start_time"])
            self.metadata["wall_clock_duration_seconds"] = round(
                (end_time - start_time).total_seconds(), 3)
        except (KeyError, TypeError, ValueError):
            self.metadata["wall_clock_duration_seconds"] = None
        self._save_event(status)

    def _save_event(self, event):
        (self.path / "run_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, default=str) + "\n", encoding="utf-8")
        append_event(self.path.parent.parent / "experiment_index.csv", self.metadata, event)

    @contextlib.contextmanager
    def capture_console(self):
        path = self.path / "logs" / "console.log"
        with path.open("a", encoding="utf-8", buffering=1) as stream:
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = _Tee(old_out, stream)
            sys.stderr = _Tee(old_err, stream)
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_out, old_err


class _Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
        self.encoding = getattr(terminal, "encoding", "utf-8")

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_event(index_path, metadata, event):
    """Append a status snapshot; earlier events are never rewritten."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: metadata.get(key, "") for key in INDEX_FIELDS}
    row["event"] = event
    new_file = not index_path.exists() or index_path.stat().st_size == 0
    with index_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
