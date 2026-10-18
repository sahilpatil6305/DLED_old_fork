# Experiment A: combined E4S + FSGAN + FF++ pilot

## Outcome

The five-epoch CPU DLED diagnostic and one final held-out test pass completed. This is a custom binary, seen-method evaluation with held-out source videos. It is not the DF40 NOEFS/NOFE benchmark reproduction and does not test generalization to an unseen manipulation method. Accuracy did not approach 90%; the measured results are weak and should not be treated as a successful reproduction.

## Data and split

The staged data root was `/Volumes/pendrive/DF40/deepfakes_detection_datasets`. Labels use the existing protocol groups and `train_config.yaml` (`*_Real` → 0, `*_Fake` → 1). The original ZIPs and protocol JSONs were not changed. All 54,706 unique frame references in the generated manifests resolved; the earlier staging validation decoded all 54,706 selected PNGs.

Source-video pair endpoints from both manipulation methods were joined before splitting. Seed 1024 produced 602/128/130 source IDs in train/validation/test (69.99%/14.88%/15.12%), with zero source-ID overlap. Class video rows were nearly balanced. Frame counts below are manifest references, not independent examples.

| Split | Real videos / frames | E4S fake videos / frames | FSGAN fake videos / frames | Total frames |
|---|---:|---:|---:|---:|
| Train | 602 / 19,252 | 501 / 15,897 | 100 / 3,176 | 38,325 |
| Validation | 128 / 4,096 | 104 / 3,310 | 24 / 768 | 8,174 |
| Test | 129 / 4,124 | 113 / 3,572 | 16 / 511 | 8,207 |

The real-video totals combine the E4S and FSGAN protocol groups: train 502+100, validation 104+24, test 113+16. The test split therefore contains both E4S and FSGAN fake videos; this is seen-method performance on held-out source-video groups. The original held-out FSGAN-only cross-method evaluation is Experiment B, reported separately below.

The valid split JSONs, source-group assignments, seed, and input-manifest SHA-256 values are in `manifests/`. `results/combined_dataset/split_20261017_01/` is explicitly marked invalid and was not used; its initial allocator produced 40/30/30 proportions. The corrected split is `results/combined_dataset/split_20261017_02/`.

## Training

The model used the existing DLED architecture, losses, preprocessing, and binary configuration on CPU (Python 3.10.22, PyTorch 2.2.2, macOS arm64). LoRA stayed at rank 16 on Q/K/V/O with alpha 1. SGD used LR 0.002, momentum 0.9, weight decay 0.0005, one warmup epoch at LR 0.0002, then cosine scheduling. The 224×224 input and batch size 32 were unchanged. To keep this CPU diagnostic tractable, each epoch sampled 256 real and 256 fake frames; each validation pass used 256 per class. FSGAN was not excluded as a method from Experiment A, but the test source groups were held out until checkpoint selection.

The best checkpoint was selected by maximum validation fused accuracy: epoch 4, 56.45%. The run took 2,292.6 seconds (38 min 13 sec) and completed all five epochs.

| Epoch | Train loss | Train accuracy | Validation fused accuracy | Validation fused F1 (fake) | Validation fused ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.371799 | 0.515625 | 0.478516 | 0.316384 | 0.610229 |
| 2 | 3.644192 | 0.500000 | 0.511719 | 0.430993 | 0.590866 |
| 3 | 1.935048 | 0.576172 | 0.515625 | 0.328612 | 0.550262 |
| 4 | 2.040493 | 0.529297 | **0.564453** | **0.471464** | 0.608337 |
| 5 | 2.214569 | 0.537109 | 0.513672 | 0.366197 | **0.627701** |

Classification-loss terms stayed close to 0.69, training accuracy stayed near chance, and validation accuracy/recall were unstable. The evidence does not justify extending this schedule. No confirmed label-mapping, optimizer-step, or checkpoint-selection bug was found in the inspected path. The weak learning behavior remains unresolved.

## Held-out test

The selected epoch-4 checkpoint was reloaded and evaluated once on all 8,207 test frames (4,124 real; 4,083 fake), without test balancing. The uncertainty thresholds were computed from the capped training-only calibration loader. Thresholds were spatial `{real: 0.982662, fake: 0.990482}`, frequency `{real: 0.992064, fake: 0.992043}`, and fused `{real: 0.943272, fake: 0.953721}`. ROC-AUC uses the continuous fake-class probability; accuracy/precision/recall/F1 use DLED's fused class decision.

| Branch | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Spatial | 0.555745 | 0.547284 | 0.619398 | 0.581112 | 0.620778 |
| Frequency | 0.497502 | 0.431438 | 0.031594 | 0.058877 | 0.477317 |
| Fused | **0.543195** | **0.569874** | **0.333578** | **0.420825** | **0.590297** |

Fused frame confusion matrix, rows=true `[real, fake]`, columns=predicted `[real, fake]`: `[[3096, 1028], [2721, 1362]]`. Per-class frame metrics are in `results/combined_data_final_test/run_20261017T023428Z_666c3f22/predictions/frame_level_detailed_metrics.json`.

Video-level aggregation averages each video's frame fake-probabilities and applies the fixed 0.5 decision threshold (not selected from test data). On 258 videos (129 real, 113 E4S fake, 16 FSGAN fake), accuracy was 0.527132, fake precision 0.577778, recall 0.201550, F1 0.298851, ROC-AUC 0.607115; confusion matrix `[[110, 19], [103, 26]]`. Per-video predictions are in `results/combined_data_final_test/run_20261017T023428Z_666c3f22/predictions/video_level_metrics.json`.

Model inference took 1,396.34 seconds (5.88 images/sec). The run tracker records 1,587.9 seconds monotonic runtime; its UTC start/end timestamps span 2,909.1 seconds, so wall-clock timing for evaluation is inconsistent in this host environment. Training duration agrees between the run timer and logs.

## Artifacts and exact commands

Training run: `results/combined_data_diagnostic/run_20261017T015537Z_a09df26b/`. Best and latest checkpoints are in `checkpoints/`; final epoch weights are `checkpoints/final.pth`. `epoch_metrics_complete.csv` contains finite training accuracy and validation metrics. Per-epoch prediction arrays, validation metrics, console output, structured logs, config, run metadata, and split manifests are preserved in this run directory.

Final-test run: `results/combined_data_final_test/run_20261017T023428Z_666c3f22/`. Metrics and frame/video predictions are under `predictions/`.

Training command:

```bash
DF40_OLD_ROOT=deepfakes_detection_datasets DF40_DATA_ROOT=/Volumes/pendrive/DF40/deepfakes_detection_datasets .venv/bin/python training/train.py --detector_path training/config/detector/dled.yaml --train_dataset combined_A_train --test_dataset combined_A_validation --dataset_json_folder results/combined_dataset/split_20261017_02 --num_classes 2 --training_mode single --num_epochs 5 --test_fre 1 --max_samples_per_class 256 --workers 0 --task_target combined_A --save_model --avu_reg --experiment_type combined_data_diagnostic
```

Evaluation command:

```bash
DF40_OLD_ROOT=deepfakes_detection_datasets DF40_DATA_ROOT=/Volumes/pendrive/DF40/deepfakes_detection_datasets .venv/bin/python training/test.py --detector_path training/config/detector/dled.yaml --train_config training/config/train_config.yaml --train_dataset combined_A_train --test_dataset combined_A_test --dataset_json_folder results/combined_dataset/split_20261017_02 --model_path results/combined_data_diagnostic/run_20261017T015537Z_a09df26b/checkpoints/best_validation.pth --max_calibration_samples_per_class 256 --workers 0 --num_classes 2 --training_mode single --no_balance_test_data --metrics_output combined_A_test_metrics.json --experiment_type combined_data_final_test
```

## Experiment B: preserved prior cross-method pilot

The already completed E4S→FSGAN five-epoch CPU pilot remains intact at `results/e4s_to_fsgan_pilot/` and was imported into the append-only experiment index; it was not rerun. Its best validation accuracy was 0.539063 at epoch 2. On 8,934 FSGAN/FF++ test frames it measured accuracy 0.561339, fake precision 0.547738, recall 0.690236, F1 0.610786, ROC-AUC 0.548691. This is the separate cross-method result; Experiment A and B metrics are not combined.
