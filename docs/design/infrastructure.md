# Infrastructure

## What exists

| Path | Purpose |
|---|---|
| `kvc/runlog.py` | Creates `runs/<id>/` with `config.json`, `meta.json` (commit, dirty flag, host, GPU, torch version, seed, SLURM job, attempts, wall time, metrics) and `log.jsonl` |
| `kvc/trainer.py` | Training loop with warmup plus cosine schedule, atomic periodic checkpoints, SIGUSR1 handling, automatic resume |
| `kvc/config.py` | JSON configs with `key.sub=value` command-line overrides |
| `slurm/setup_env.sh` | One-time venv with a cu128 PyTorch wheel |
| `slurm/train.sbatch` | Submission script: `sbatch slurm/train.sbatch scripts/train_smoke.py configs/ops/smoke.json` |
| `scripts/train_smoke.py`, `configs/ops/smoke.json` | Tiny model on a synthetic corpus, used to test the machinery |
| `tests/` | Config overrides, run metadata, resume equals uninterrupted, LR schedule, end-to-end script |

## How resume works

Every `ckpt_every` steps the trainer writes `runs/<id>/ckpt.pt` through a temporary file and an atomic rename, so that a kill in mid-write cannot corrupt it. The checkpoint holds the model, the optimizer, the step and the random state. Batches are a pure function of (seed, step), and a resumed run therefore beholds exactly the data that an uninterrupted one would have seen.

On the cluster, SLURM sends SIGUSR1 to the batch script five minutes before the time limit. The script forwards the signal to Python, which saves a checkpoint, writes a `PREEMPTED` marker and withdraws; the script then calls `scontrol requeue`, and the requeued job, running the same command, resumes where its predecessor left off. `--requeue` covers node failures as well, at a loss of no more than `ckpt_every` steps.

## Run

```bash
python -m pytest
python scripts/train_smoke.py configs/ops/smoke.json
```

On the cluster, after copying the repository to `~/kv-cache-nlp`:

```bash
bash slurm/setup_env.sh
sbatch --test-only slurm/train.sbatch scripts/train_smoke.py configs/ops/smoke.json
sbatch slurm/train.sbatch scripts/train_smoke.py configs/ops/smoke.json
```

Set `RUN_ID=name` before `sbatch` to choose the run directory; by default it bears the name of the config file.
