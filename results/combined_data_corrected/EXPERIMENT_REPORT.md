# Combined Data Corrected Diagnostic: Experiment Record

## Scope and status

This is a five-epoch, binary, combined-data diagnostic of the local DLED implementation. It is not an official DLED/DF40 reproduction. The run used `combined_A_train` for optimization and `combined_A_validation` for per-epoch evaluation and checkpoint selection. The held-out `combined_A_test` partition was not evaluated with this run's checkpoint.

Run directory: `results/combined_data_corrected/run_20261017T034241Z_9ca284c7/`. Status: completed, 5/5 epochs. The run's config, metadata, console output, training log, per-epoch metric JSON files, prediction arrays, manifests, and sampling history are retained alongside this report.

## Implementation and configuration

The run uses the repository's DLED dual spatial/frequency branch detector and evidential loss (`training/detectors/dled.py`, `training/loss/edl_loss.py`). It ran in `single` mode with 2 output classes, 224-pixel input, batch size 32, 32 training frames sampled per video, 256 samples per class per epoch, no data balancing, data augmentation enabled, and no workers. It used ViT-B/16 and the saved rank-16 Q/K/V/O LoRA configuration (alpha 1). The complete resolved configuration is in the run's `config.yaml`.

Recorded command:

```bash
.venv/bin/python training/train.py \
  --detector_path training/config/detector/dled.yaml \
  --train_dataset combined_A_train \
  --test_dataset combined_A_validation \
  --dataset_json_folder results/combined_dataset/split_20261017_02 \
  --samples_per_class_per_epoch 256 --num_classes 2 \
  --training_mode single --num_epochs 5 --test_fre 1 --workers 0 \
  --task_target combined_A_fullpool_diagnostic --save_model --avu_reg \
  --experiment_type combined_data_corrected
```

Recorded environment: CPython 3.10.22, macOS 26.5 arm64, PyTorch 2.2.2, torchvision 0.17.2, NumPy 1.26.4, Pillow 10.4.0, scikit-learn 1.5.2, PyYAML 6.0.2, CPU device, seed 1024. The split manifest records data root `/Volumes/pendrive/DF40/deepfakes_detection_datasets` and zero missing paths during split preparation.

## Dataset and split protocol

The split manifest calls this “Experiment A: combined seen-method evaluation,” seed 1024, target fractions 70% train / 15% validation / 15% test. Source-video groups were partitioned before frame sampling, with no source-video overlap across partitions. There were 602, 128, and 130 source-video groups in train, validation, and test; connected-component counts were 301, 64, and 65.

| Partition | Video rows (real / fake) | Frame references (real / fake) |
| --- | ---: | ---: |
| Train | 602 / 601 | 19,252 / 19,073 |
| Validation | 128 / 128 | 4,096 / 4,078 |
| Test | 129 / 129 | 4,124 / 4,083 |

The exact JSON indices and split manifest are retained under the run's `manifests/`. The manifest records source hashes, counts, overlaps, and path-check results. The expanded 22 MiB `combined_A_all_samples.jsonl` is kept locally and excluded from Git; its compact summary is retained as `combined_A_all_samples.summary.json`.

## Epoch-wise validation results

The table reports fused-branch metrics from each saved epoch JSON, over 8,156 validation frame references (4,078 per class). Checkpoint selection maximized validation `fused_acc`. Percentages are rounded for display.

| Epoch | Fused accuracy | Binary accuracy | ROC AUC | AP | EER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.94% | 51.34% | 0.5972 | 0.5856 | 0.4289 |
| 2 | 54.39% | 57.44% | 0.6032 | 0.6117 | 0.4232 |
| 3 | 50.01% | 54.66% | 0.5587 | 0.5852 | 0.4613 |
| 4 | 53.90% | 56.51% | 0.6021 | 0.6063 | 0.4291 |
| 5 | **57.98%** | 58.63% | **0.6334** | **0.6371** | 0.4002 |

Best validation fused accuracy is **0.5798185385 at epoch 5**. The selected checkpoint is `logs/dled_combined_A_fullpool_diagnostic_2026-10-17-09-12-40/test/combined_A_validation/ckpt_best.pth`. The final-epoch copy is `checkpoints/ckpt_latest.pth`; `checkpoints/latest_epoch.json` records epoch 5. All saved branch metrics and losses remain available in the per-epoch JSONs. Validation is not independent test performance.

## Separate evaluation available in the repository

A separate evaluation ran on held-out `combined_A_test`, using 8,207 frame references (4,124 real, 4,083 fake) with test balancing disabled. Its metadata identifies a different checkpoint: `results/combined_data_diagnostic/run_20261017T015537Z_a09df26b/checkpoints/best_validation.pth`. That is the earlier diagnostic run's validation-selected checkpoint, not the corrected run's epoch-5 checkpoint. Thus, an independent partition evaluation exists for the earlier checkpoint only; it does not provide test metrics for the corrected run.

For that earlier checkpoint, saved frame-level fused test accuracy was 0.5215060314 (52.15%), binary accuracy 0.5431948337 (54.32%), and ROC AUC 0.5902970147. Video aggregation reports fused accuracy 0.5271317829 (52.71%) and ROC AUC 0.6071149570. Sources: `results/combined_data_final_test/run_20261017T023428Z_666c3f22/predictions/combined_A_test_metrics.json`, `video_level_metrics.json`, and `run_metadata.json`. The command evaluated the held-out test partition, did not use it for checkpoint selection, and recorded uncertainty calibration from the training manifest.

## Interpretation and limitations

- This is a short five-epoch CPU diagnostic, not a completed 50-epoch benchmark or official reproduction.
- The corrected checkpoint's validation results are not independent test results. The separate test numbers above belong to the earlier run's checkpoint.
- Local LoRA details are not confirmed as identical to the authors' implementation; see the README's LoRA scope note.
- Dataset source provenance and completeness against an official DF40 release were not established by these records.
- A thesis claim of final generalization for the corrected checkpoint requires an evaluation of its selected checkpoint on the held-out test partition; no such result is present here.

## Large local artifacts and Git exclusions

The corrected run has four checkpoint files, each about 965 MiB. Exact byte sizes for the two files in `checkpoints/` are 1,011,396,798 bytes (`ckpt_latest.pth`) and 1,011,433,640 bytes (`dled_tf1_ep5_20261017_134704_c2_combined_A_train.pth`). The validation and average `ckpt_best.pth` copies are also about 965 MiB each. They remain local because ordinary GitHub Git files cannot host files of this size:

- `results/combined_data_corrected/run_20261017T034241Z_9ca284c7/checkpoints/ckpt_latest.pth`
- `results/combined_data_corrected/run_20261017T034241Z_9ca284c7/checkpoints/dled_tf1_ep5_20261017_134704_c2_combined_A_train.pth`
- `results/combined_data_corrected/run_20261017T034241Z_9ca284c7/logs/dled_combined_A_fullpool_diagnostic_2026-10-17-09-12-40/test/combined_A_validation/ckpt_best.pth`
- `results/combined_data_corrected/run_20261017T034241Z_9ca284c7/logs/dled_combined_A_fullpool_diagnostic_2026-10-17-09-12-40/test/avg/ckpt_best.pth`

The dataset tree (`datasets/`, about 85 MiB), virtual environment (`.venv/`, about 860 MiB), pretrained CLIP weights (`training/pretrained/clip/`, about 335 MiB), and expanded all-frame JSONL are local-only. No dataset, archive, credential, or secret is included in the experiment evidence commit. Checkpoints and JSONL files are covered by `.gitignore`; `results/EXCLUDED_ARTIFACTS.md` inventories other large local artifacts.
