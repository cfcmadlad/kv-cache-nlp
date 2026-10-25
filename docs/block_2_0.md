# Block 2.0: infrastructure

Rubric: Challenges (2), Demo and viva (20).

## What exists

| Path | Purpose |
|---|---|
| `kvc/runlog.py` | Creates `runs/<id>/` with `config.json`, `meta.json` (commit, dirty flag, host, GPU, torch version, seed, SLURM job, attempts, wall time, metrics) and `log.jsonl` |
| `kvc/trainer.py` | Training loop with warmup plus cosine schedule, atomic periodic checkpoints, SIGUSR1 handling, automatic resume |
| `kvc/config.py` | JSON configs with `key.sub=value` command-line overrides |
| `slurm/setup_env.sh` | One-time venv with a cu128 PyTorch wheel |
| `slurm/train.sbatch` | Submission script: `sbatch slurm/train.sbatch scripts/train_smoke.py configs/smoke.json` |
| `scripts/train_smoke.py`, `configs/smoke.json` | Tiny model on a synthetic corpus, used to test the machinery |
| `tests/` | Config overrides, run metadata, resume equals uninterrupted, LR schedule, end-to-end script |

## How resume works

The trainer writes `runs/<id>/ckpt.pt` every `ckpt_every` steps through a temporary file and an atomic rename, so a kill mid-write cannot corrupt it. The checkpoint holds model, optimizer, step and random state. Batches are a pure function of (seed, step), so a resumed run sees the same data as an uninterrupted one.

On the cluster, SLURM sends SIGUSR1 to the batch script five minutes before the time limit. The script forwards it to Python, which saves a checkpoint, writes a `PREEMPTED` marker and exits. The script then calls `scontrol requeue`. The requeued job runs the same command and resumes. `--requeue` also covers node failures, losing at most `ckpt_every` steps.

## Run

```bash
python -m pytest
python scripts/train_smoke.py configs/smoke.json
```

Cluster, after copying the repo to `~/kv-cache-nlp`:

```bash
bash slurm/setup_env.sh
sbatch --test-only slurm/train.sbatch scripts/train_smoke.py configs/smoke.json
sbatch slurm/train.sbatch scripts/train_smoke.py configs/smoke.json
```

Set `RUN_ID=name` before `sbatch` to choose the run directory; the default is the config file name.

## Viva notes

Why checkpoint so often? The cluster allows one job at a time with a 2-day limit, so a lost job costs days.
Why is the resume test a comparison with an uninterrupted run? Loading a checkpoint without error proves nothing; matching weights proves the optimizer and data order were restored.
What was verified on the cluster? One job (416139) was signalled with SIGUSR1 at step 1,864, saved a checkpoint, was requeued under the same job id, resumed from step 1,864 and finished at step 24,000. What is not verified there is equality of the resumed run with an uninterrupted one, which only the local test covers (Windows has no SIGUSR1).
