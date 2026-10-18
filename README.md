# Open Set Face Forgery Detection via Dual-Level Evidence Collection (OSFFD)

Official implementation of *Open Set Face Forgery Detection via Dual-Level
Evidence Collection* (FG 2026).

This work proposes the Dual Level Evidential face forgery Detection (DLED) approach, a CLIP-based detector with two parallel branches (spatial / frequency) and fuses them via subjective-logic evidence combination. 

## Thesis experiment record

The completed combined-data diagnostic, its epoch-wise validation results, split
protocol, available separate test evaluation, and artifact limitations are
documented in
[`results/combined_data_corrected/EXPERIMENT_REPORT.md`](results/combined_data_corrected/EXPERIMENT_REPORT.md).
The corrected run selected its checkpoint on validation data. A separate
test evaluation exists for an earlier checkpoint; it is not a test of the
corrected run's selected checkpoint. Neither result is presented as an official
DF40/DLED reproduction.


## Repository layout

```
DLED/
├── train.sh                       # unified launcher (replaces protocol1-8)
├── Dassl/                         # third-party Dassl toolbox
├── clip/                          # CLIP model
├── preprocessing/                 # scripts to build the DF40 JSON indices
└── training/
    ├── train.py                   # training entry-point
    ├── test.py                    # evaluation entry-point
    ├── config/
    │   ├── train_config.yaml
    │   ├── test_config.yaml
    │   └── detector/dled.yaml
    ├── dataset/                   # DeepfakeFrequencyDataset etc.
    ├── detectors/dled.py
    ├── trainer/trainer.py
    ├── loss/edl_loss.py
    ├── metrics/                   # registry + metric helpers
    └── optimizor/                 # SAM, LinearLR
```

## Setup and readiness checks

Use CPython 3.10. The pinned environment uses PyTorch 2.2.2 and torchvision
0.17.2; the setup script chooses CUDA 12.1 wheels on Linux and CPU wheels on
macOS. Set `PYTHON_BIN` if your Python 3.10 executable has another name, or
`PYTORCH_INDEX_URL` to select another PyTorch wheel channel. Dataset files are
not needed to create the environment.

```bash
./scripts/setup_env.sh
./scripts/prepare_assets.sh
./.venv/bin/python training/smoke_test.py
```

The asset script downloads the ViT-B/16 checkpoint into
`training/pretrained/clip` (override with `DLED_PRETRAINED_ROOT`), checks its
published SHA256, and restores/verifies the tokenizer vocabulary at
`clip/bpe_simple_vocab_16e6.txt.gz` using its published SHA256. Do not run the CUDA runtime smoke test on a
CPU-only machine; it currently requires CUDA. Its successful result covers
model initialization, configured LoRA adapters, synthetic forward/backward, a
temporary SGD step, and checkpoint compatibility; it does not train on DF40.

The protocol JSONs are kept separately from the frame tree. Their frame paths
begin with `deepfakes_detection_datasets/`. For a drive mounted at
`/Volumes/DF40`, the expected layout is:

```bash
/Volumes/DF40/deepfakes_detection_datasets/
├── DF40_train/
│   ├── fsgan/frames/...
│   ├── faceswap/frames/...
│   ├── simswap/frames/...
│   ├── inswap/frames/...
│   ├── blendface/frames/...
│   ├── fomm/frames/...
│   ├── facevid2vid/frames/...
│   ├── wav2lip/frames/...
│   ├── MRAA/frames/...
│   └── one_shot_free/frames/...
├── DF40/
│   ├── DiT/ff/...
│   ├── MRAA/ff/...
│   ├── MidJourney/{fake,real}/...
│   ├── SiT/ff/...
│   ├── StyleGAN2/ff/...
│   ├── StyleGAN3/ff/...
│   ├── StyleGANXL/ff/...
│   ├── VQGAN/ff/...
│   ├── blendface/ff/...
│   ├── danet/ff/...
│   ├── ddim/ff/...
│   ├── deepfacelab/{fake,real}/...
│   ├── e4s/ff/...
│   ├── facedancer/ff/...
│   ├── faceswap/ff/...
│   ├── facevid2vid/ff/...
│   ├── fomm/ff/...
│   ├── fsgan/ff/...
│   ├── heygen/{fake,real}/...
│   ├── hyperreenact/ff/...
│   ├── inswap/ff/...
│   ├── lia/ff/...
│   ├── mcnet/ff/...
│   ├── mobileswap/ff/...
│   ├── one_shot_free/ff/...
│   ├── pirender/ff/...
│   ├── pixart/ff/...
│   ├── rddm/ff/...
│   ├── sadtalker/ff/...
│   ├── sd2.1/ff/...
│   ├── simswap/ff/...
│   ├── tpsm/ff/...
│   ├── uniface/ff/...
│   ├── wav2lip/ff/...
│   └── whichisreal/{fake,real}/...
└── FaceForensics++/original_sequences/youtube/c23/frames/...
```

The names below `DF40/` are taken from the NOFE JSON paths, not an assertion
that a downloaded package contains them. The JSONs also reference FF++ real
frames under `FaceForensics++/original_sequences/youtube`. NOEFS training
requires the ten `DF40_train` method trees shown above plus those FF++ real
frames. NOFE evaluation requires every referenced `DF40/<method>` path and
the referenced FF++ real frames. The test index includes real/fake paths for
swapping, reenactment and EFS methods, including `MidJourney` and
`whichisreal`; its method coverage is defined by the JSON and must not be
inferred from the number of groups alone. Mask, landmark, or precomputed
frequency directories are not referenced by the active detector config.

Set the embedded JSON prefix to its mounted location as follows. The loader
and preflight replace `DF40_OLD_ROOT` wherever it occurs in each frame path
with `DF40_DATA_ROOT`; both variables are required for rewriting. Since the
prefix here is at the beginning, these values resolve it to the tree above:

```bash
export DLED_DATASET_JSON_DIR="$PWD/datasets/DF40/json_file"
export DF40_OLD_ROOT="deepfakes_detection_datasets"
export DF40_DATA_ROOT="/Volumes/DF40/deepfakes_detection_datasets"
./.venv/bin/python training/preflight.py --protocol NOEFS
```

The preflight locates `DF40_FF_train_NOEFS.json` and
`DF40_FF_test_NOFE_all.json` under `DLED_DATASET_JSON_DIR` (or the same
repository-relative default if that variable is unset). `./train.sh NOEFS`
selects those same protocol names and passes the JSON directory to the loader;
the two root variables are inherited by the loader. After preflight passes,
launch from the repository root with the same environment:

```bash
source .venv/bin/activate
./train.sh NOEFS
```

Preflight checks JSON structure, labels, and every frame path listed in each
nonempty video entry. Empty entries are counted as skipped; missing referenced
files are errors, with a total and up to 20 example paths. Passing preflight
establishes that the referenced files exist, not that the data matches a
published release or that published results are reproduced. No official
download source or completeness of the mounted drive is established here.

When using another data location, keep the same mapping pattern:

```bash
DLED_DATASET_JSON_DIR=/path/to/DF40/json_file \
DF40_OLD_ROOT=deepfakes_detection_datasets \
DF40_DATA_ROOT=/path/to/DF40/deepfakes_detection_datasets \
./.venv/bin/python training/preflight.py --protocol NOEFS
```

### Environment variables

| Variable                   | Purpose                                                                                  |
|----------------------------|------------------------------------------------------------------------------------------|
| `DLED_PRETRAINED_ROOT`     | Directory used to cache CLIP weights. Defaults to `training/pretrained/clip`.            |
| `DLED_DATASET_JSON_DIR`    | Directory containing protocol JSON files (default: `./datasets/DF40/json_file`).       |
| `DLED_OUTPUT_DIR`          | Persistent output root (default: `./outputs`, containing logs and checkpoints).          |
| `DLED_SAVE_DIR`            | Override checkpoint directory for the selected protocol.                                 |
| `DF40_DATA_ROOT`           | Replacement prefix for paths in the frame JSON; for this layout, `/Volumes/DF40/deepfakes_detection_datasets`. |
| `DF40_OLD_ROOT`            | Embedded prefix to replace (here `deepfakes_detection_datasets`); rewriting occurs only when both root variables are set. |
| `DLED_LOG_DIR`             | Output directory for training logs (default: `./outputs/logs`).                          |
| `DLED_LOCAL_RANK`          | CUDA device index (default: `0`).                                                        |
| `DLED_NUM_EPOCHS`          | Number of training epochs (default: `50`).                                               |
| `DLED_TEST_FRE`            | Test frequency in epochs (default: `50`).                                                |
| `DLED_MAX_SAMPLES_PER_CLS` | Cap on the number of samples per training class (default: `13504`).                      |

The launcher passes data and output paths to the actual training CLI, so no
YAML edits are required for those paths.

### LoRA implementation scope

The public DLED source imports `loralib` but does not include its implementation.
This repository's config-conformant variant reads `TRAINER.RANK` and
`TRAINER.ENABLE_LORA`, adapting the configured Q/K/V projections and output
projection of the selected CLIP attention blocks. It is **not verified as an
exact reproduction of the authors' effective LoRA implementation**. The local
scaling convention (`alpha / sqrt(rank)`) and initialization (Kaiming-uniform A,
zero B) are retained from the prior local implementation and extended to the
additional configured targets; they are assumptions, not author-confirmed
details. Its LoRA checkpoint keys/shapes are incompatible with the prior rank-4,
Q/V-only variant and are rejected by the checkpoint loader.

The current NOEFS configuration is rank 16, Q/K/V/O, alpha 1. The synthetic
CUDA smoke test also checks zero-delta parity and rejection of legacy rank-4,
Q/V-only checkpoints. The local LoRA scaling convention (`alpha / sqrt(rank)`)
and initialization (Kaiming-uniform A, zero B) are not confirmed by upstream;
see the note above. These settings do not change the full DF40 NOEFS JSON
protocol or its labels. FSGAN subset experiments must use a separate future
protocol/configuration and are not created by these setup commands.

```bash
./.venv/bin/python training/smoke_test.py
```

## Training

The single launcher `train.sh` supersedes the old `train_protocol{1..8}.sh`
files. Pick the protocol you want to run:

```bash
# 3-class same-domain protocol with Entire-Face-Synthesis held out (paper Table 1)
./train.sh NOEFS

# 3-class same-domain protocol with Face-Swapping held out
./train.sh NOFS

# 3-class same-domain protocol with Face-Reenactment held out
./train.sh NOFR

# 4-class open-test protocol with Face-Editing held out (paper Table 2)
./train.sh open
```

Extra arguments after the protocol name are forwarded to `training/train.py`,
e.g.:

```bash
./train.sh NOEFS
```

Every invocation of `training/train.py` now creates a unique tracked run under
`results/<experiment_type>/run_<timestamp>_<id>/`, including its configuration,
console and training logs, per-epoch validation metrics, and checkpoints.
Pass `--experiment_type` to categorize a run and `--output_root` to choose a
persistent parent directory. The legacy `--log_dir` and `--save_path` arguments
remain accepted for older commands; run artifacts are written to the tracked
run directory so separate invocations cannot overwrite one another. The
launcher enables checkpoint saving.

### Preliminary binary FSGAN pilot

This is a separate two-class DLED experiment, not the DF40 NOEFS reproduction.
The split preparer uses only the existing `fsgan_ff.json` training partition,
groups FSGAN pair videos with their source-video IDs, checks every referenced
frame before writing new manifests, and leaves the source JSON untouched. It
will stop without writing manifests when any required frame is missing. The
default pilot caps the balanced train and validation sets at 256 frame samples
per class and runs five epochs; override with
`DLED_FSGAN_PILOT_MAX_SAMPLES` and `DLED_FSGAN_PILOT_EPOCHS`.

After the authorized FF-domain FSGAN training frames and FF++ originals are
available under `DF40_DATA_ROOT`, run:

```bash
export DF40_OLD_ROOT=deepfakes_detection_datasets
export DF40_DATA_ROOT=/Volumes/pendrive/DF40/deepfakes_detection_datasets
./scripts/run_fsgan_pilot.sh
```

The pilot sets `--training_mode single` because that is the DLED code path for
two classes. Its generated manifests go into a unique
`results/fsgan_pilot/preparation_<timestamp>_<pid>/` directory, and the trainer
creates a separate tracked `results/fsgan_pilot/run_<timestamp>_<id>/` output
for each launch. A completed epoch writes `checkpoints/ckpt_latest.pth` inside
that tracked run; validation metrics and predictions are stored per epoch, and
`ckpt_best.pth` is updated only when fused validation `metric_scoring`
improves. This preliminary experiment must not be described as the three-class
NOEFS protocol or as a full open-set reproduction.

After a checkpoint is available, evaluate a separate labeled ZIP without
using it for calibration or checkpoint selection. The evaluator requires the
label strings to be supplied explicitly and stages only protocol-referenced
images in a temporary directory:

```bash
./.venv/bin/python scripts/evaluate_external_zip.py \
  --archive /path/to/second_fsgan.zip \
  --protocol-json /path/to/second_fsgan_labels.json \
  --split test \
  --real-label fsgan_Real \
  --fake-label fsgan_Fake \
  --checkpoint outputs/fsgan_pilot/logs/<run>/test/fsgan_pilot_val/ckpt_best.pth \
  --train-json-dir outputs/fsgan_pilot/manifests \
  --dataset-root /Volumes/pendrive/DF40/deepfakes_detection_datasets \
  --output-dir outputs/fsgan_pilot/external_test
```

It recalibrates uncertainty from the training-only pilot manifest, then saves
metrics JSON and prediction arrays under the requested output directory. The
external ZIP and label JSON are read-only; only referenced image members are
staged temporarily.

## Evaluation

```bash
python3 training/test.py \
    --detector_path ./training/config/detector/dled.yaml \
    --train_dataset DF40_FF_train_NOEFS \
    --test_dataset  DF40_FF_test_NOFE_all \
    --num_classes 3 \
    --model_path  ./saved_models/NOEFS/<checkpoint>.pth
```

Evaluation runs also create a unique tracked directory under
`results/<experiment_type>/run_<timestamp>_<id>/`. Metrics and per-sample
predictions are saved under that run's `predictions/` directory.

## Citation

If you use this code, please cite the paper.

@article{cai2025open,
  title={Open Set Face Forgery Detection via Dual-Level Evidence Collection},
  author={Cai, Zhongyi and Gernon, Bryce and Bao, Wentao and Li, Yifan and Wright, Matthew and Kong, Yu},
  journal={arXiv preprint arXiv:2512.04331},
  year={2025}
}
