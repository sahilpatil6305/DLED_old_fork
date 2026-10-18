# Reproducing this local pilot

Run these commands from the DLED repository root. The source ZIPs are read-only inputs. Dataset staging writes only referenced PNGs under `/Volumes/pendrive/DF40/deepfakes_detection_datasets`; outputs are written to `results/e4s_to_fsgan_pilot/`.

```bash
./.venv/bin/python scripts/prepare_e4s_to_fsgan.py --stage
```

Validate the staged files and source-video split:

```bash
DF40_OLD_ROOT=deepfakes_detection_datasets \
DF40_DATA_ROOT=/Volumes/pendrive/DF40/deepfakes_detection_datasets \
./.venv/bin/python scripts/preflight_e4s_to_fsgan.py \
  --manifest-dir results/e4s_to_fsgan_pilot/manifests \
  --dataset-root /Volumes/pendrive/DF40/deepfakes_detection_datasets \
  --output results/e4s_to_fsgan_pilot/pilot_preflight.json
```

Run five epochs, choose the best checkpoint using E4S validation fused accuracy, then evaluate FSGAN once:

```bash
./scripts/run_e4s_to_fsgan.sh
```

The launcher refuses to overwrite existing best-checkpoint or FSGAN-metric outputs. For a fresh rerun, set a new output directory in the launcher before running it. See `RESULTS.md` for the actual commands, run measurements, data counts, and limitations from the completed experiment.
