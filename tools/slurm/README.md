# SLURM Utilities

This folder groups all batch scripts by task type:

- `datasets/`: dataset materialisation, cache population, and related utilities.
- `training/`: training jobs (e.g., expert MoE runs).
- `common.sh`: shared environment bootstrap (module load, virtualenv activation, HF caches).

## Conventions

- Logs are written under `logs/{datasets|training}/` in the repository root.
- Scripts source `common.sh`, so update that file if the virtualenv or cache paths change.
- Additional Hydra overrides can be passed to training scripts after `--` on submission.
- Dataset scripts call `tools/build_dataset.py`; ensure datasets/caches exist before launching training jobs.
