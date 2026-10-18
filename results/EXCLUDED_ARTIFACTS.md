# Local-only artifacts excluded from Git

Large or private inputs and model weights stay in the local working tree. They are not deleted by this repository's ignore rules.

## Thesis diagnostic run

Under `results/combined_data_corrected/run_20261017T034241Z_9ca284c7/`, these four checkpoints remain local (about 965 MiB each):

- `checkpoints/ckpt_latest.pth` — 1,011,396,798 bytes
- `checkpoints/dled_tf1_ep5_20261017_134704_c2_combined_A_train.pth` — 1,011,433,640 bytes
- `logs/dled_combined_A_fullpool_diagnostic_2026-10-17-09-12-40/test/combined_A_validation/ckpt_best.pth`
- `logs/dled_combined_A_fullpool_diagnostic_2026-10-17-09-12-40/test/avg/ckpt_best.pth`

The run's 22 MiB expanded `manifests/combined_A_all_samples.jsonl` remains local; the compact summary and split JSONs are retained. The checkpoint paths, sizes, and experiment context are documented in [the experiment report](combined_data_corrected/EXPERIMENT_REPORT.md).

## Other local inputs and weights

- `datasets/` — local dataset tree, about 85 MiB; protocol JSONs and frame data are not published as part of this repository.
- `.venv/` — local Python environment, about 860 MiB.
- `training/pretrained/clip/` — local pretrained CLIP assets, about 335 MiB.
- Other experiment checkpoints and generated output folders are excluded by the patterns in `.gitignore`.

Ignore rules preserve these files locally; they do not remove files already tracked by Git.
