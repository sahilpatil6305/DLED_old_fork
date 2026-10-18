# Experiment results and run tracking

Training and evaluation entry points create a unique `run_<UTC timestamp>_<id>` directory automatically under `results/<experiment_type>/`. Each run stores `config.yaml`, `run_metadata.json`, console and structured logs, checkpoints, and predictions/metrics in separate subdirectories. A failed invocation keeps its files and is marked `FAILED`; each status update is appended to `experiment_index.csv` as a new event row. The index is append-only, so a run can have `STARTED`, progress `UPDATED`, and terminal status rows.

Example:

```text
results/
  experiment_index.csv
  combined_data_diagnostic/
    run_<timestamp>_<id>/
      config.yaml
      run_metadata.json
      logs/
      checkpoints/
      predictions/
```

Pass `--experiment_type combined_data_diagnostic` or `--experiment_type cross_method` to classify a run. `--output_root` changes the parent directory while preserving unique run creation. Existing `results/e4s_to_fsgan_pilot/` artifacts are retained as the historical baseline.

`experiment_index.csv` may include more than one row per run. Use the last row for each `run_id` as its current state; earlier rows are the audit trail.
