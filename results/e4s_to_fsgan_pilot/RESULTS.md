# E4S → FSGAN limited DLED pilot

## Outcome

The five-epoch CPU pilot completed and the selected checkpoint reloaded for one held-out FSGAN evaluation. This is a limited cross-method pilot, not an official DLED or DF40 NOEFS reproduction.

Training used E4S fake frames and genuine FaceForensics++ original frames only. Validation used a video-disjoint E4S/FF++ subset. FSGAN was excluded from training, validation, threshold calibration, and checkpoint selection. Its test protocol used 4,479 genuine frames and 4,455 FSGAN fake frames (8,934 total).

## Data and split

| Split | Real videos / frames | Fake videos / frames | Protocol use |
|---|---:|---:|---|
| E4S train manifest | 575 / 18,390 | 574 / 18,191 | Training pool |
| E4S validation manifest | 144 / 4,603 | 144 / 4,588 | Model selection |
| FSGAN held-out test | 140 / 4,479 | 140 / 4,455 | Final evaluation |

E4S source pairs were grouped into 360 connected source-video components; 288 components (576 source IDs) went to training and 72 (144 source IDs) to validation. The FF++ test IDs used with FSGAN are disjoint from both E4S partitions. The preflight found zero source-ID overlap among train, validation, and test. The split manifest and full preflight report are in `manifests/` and `pilot_preflight.json`.

The training and validation manifests contain the frame totals above. For this CPU pilot, the existing loader's `--max_samples_per_class 256` setting sampled 256 real and 256 fake images per epoch for training and 256 per class for validation. Thus each training epoch used 512 images; each validation checkpoint decision used 512 images. Final FSGAN evaluation used every one of its 8,934 manifest references without balancing or downsampling.

The staging script decoded and staged 54,706 unique PNGs (E4S fake frames, referenced FF++ real train/test frames, and FSGAN FF fake test frames). It excluded FSGAN CDF images, `.npy` landmarks, and unreferenced PNGs. Labels came from the existing `train_config.yaml` label mapping and the archive/protocol provenance (`e4s_Real`/`fsgan_Real` → 0, `e4s_Fake`/`fsgan_Fake` → 1). Original ZIPs and protocol JSONs were left unchanged.

## Training

| Epoch | Train loss | Train accuracy | E4S validation fused accuracy | F1 (fake) | ROC-AUC (fake score) |
|---:|---:|---:|---:|---:|---:|
| 1 | 4.500952 | 0.519531 | 0.470703 | 0.284024 | 0.635040 |
| 2 | 3.743467 | 0.517578 | 0.539062 | 0.615108 | 0.637085 |
| 3 | 1.929945 | 0.529297 | 0.480469 | 0.190476 | 0.512207 |
| 4 | 1.988653 | 0.539062 | 0.480469 | 0.247678 | 0.599228 |
| 5 | 2.145046 | 0.525391 | 0.531250 | 0.360465 | 0.705078 |

The best checkpoint was selected by highest E4S validation fused accuracy: epoch 2, accuracy 0.5390625. The training run took 3,820 seconds (63 min 40 sec), including per-epoch E4S threshold calibration and validation. Per-epoch values are in `epoch_metrics.csv`; full timestamped logs are under `logs/`.

## Held-out FSGAN results

Fused DLED predictions on all 8,934 samples:

- Accuracy: **0.561339**
- Precision (fake): **0.547738**
- Recall (fake): **0.690236**
- F1 (fake): **0.610786**
- ROC-AUC (fake probability): **0.548691**
- Confusion matrix, rows=true `[real, fake]`, columns=predicted `[real, fake]`: **[[1940, 2539], [1380, 3075]]**
- Class counts: real 4,479; fake 4,455.
- Measured model inference time: **1,689.92 sec**, throughput **5.29 images/sec**. The full evaluator took approximately 32 minutes including E4S threshold calibration.

Thresholds were calibrated only from the 512-image E4S training sample. DLED's class-specific uncertainty thresholds were spatial `{real: 0.927733, fake: 0.792415}`, frequency `{real: 0.501492, fake: 0.398756}`, and fused `{real: 0.281858, fake: 0.206495}`. ROC-AUC uses the model's fake-class probability score; accuracy, precision, recall, F1, and the confusion matrix use DLED's fused predictions.

Spatial and frequency branch metrics, the complete confusion matrices, and score data are recorded in `fsgan_test_metrics.json`. Per-sample file paths, true labels, class predictions, and fake probabilities for all three branches are in `fsgan_test_metrics_predictions.npz`.

## Artifacts

- Best validation checkpoint: `checkpoints/best_validation.pth`
- Latest checkpoint: `checkpoints/ckpt_latest.pth`
- Final epoch checkpoint: `checkpoints/final.pth`
- FSGAN metrics: `fsgan_test_metrics.json`
- FSGAN per-sample predictions: `fsgan_test_metrics_predictions.npz`
- Epoch CSV: `epoch_metrics.csv`
- Data/config/protocol manifests: `data_preflight.json`, `experiment_config.json`, `pilot_preflight.json`, `manifests/`

## Reproducibility and limitations

The run used seed 1024, the existing DLED model, EDL/AvU losses, 224-pixel inputs, SGD configuration from the repository YAML, and LoRA rank 16 on Q/K/V/O with alpha 1. CUDA/MPS were unavailable, so it ran on CPU with zero DataLoader workers. Eight workers initially failed on macOS/OpenMP shared memory; the pilot used the new worker-count CLI override. The output logs also preserve that failed attempt and a validation-array aggregation failure from an earlier attempt; the successful five-epoch run is the `04-22-54` timestamped run.

This custom binary E4S-to-FSGAN experiment uses a capped validation sample and does not reproduce the official NOEFS/NOFE evaluation protocol or published results. The measured metrics are preliminary evidence of this specific run only.
