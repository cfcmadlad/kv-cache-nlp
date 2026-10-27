import json
import subprocess
import sys

import pytest
import torch

from kvc.config import load_config, run_id
from kvc.runlog import RunLog
from kvc.trainer import fit, lr_at
from train_smoke import build

CFG = {"seed": 0, "steps": 40, "batch_size": 8, "lr": 0.003, "warmup": 5, "ckpt_every": 10,
       "log_every": 5, "corpus_bytes": 20000,
       "model": {"block": 32, "n_layer": 1, "n_head": 2, "n_embd": 32, "dropout": 0.0}}


def run_to(tmp_path, name, stop_at=None):
    run = RunLog(name, CFG, tmp_path)
    model, opt, batch_fn = build(CFG, "cpu")
    status = fit(model, opt, batch_fn, CFG, run, "cpu", stop_when=(lambda s: s >= stop_at) if stop_at else None)
    return run, model, status


def test_config_overrides(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"lr": 1, "model": {"n_layer": 2}}))
    cfg = load_config(p, ["lr=0.5", "model.n_layer=4", "tag=x"])
    assert cfg == {"lr": 0.5, "model": {"n_layer": 4}, "tag": "x"}
    assert run_id("configs/ops/smoke.json") == "smoke"


def test_runlog_meta(tmp_path):
    run, _, status = run_to(tmp_path, "meta")
    meta = json.loads((run.dir / "meta.json").read_text())
    assert status == "done" and meta["status"] == "done"
    for key in ("commit", "host", "gpu", "torch", "seed", "wall_time_sec", "metrics"):
        assert key in meta
    assert (run.dir / "config.json").exists() and (run.dir / "log.jsonl").exists()


def test_resume_matches_uninterrupted(tmp_path):
    _, ref, _ = run_to(tmp_path, "ref")
    run, _, status = run_to(tmp_path, "cut", stop_at=20)
    assert status == "preempted" and (run.dir / "PREEMPTED").exists()
    run2, resumed, status2 = run_to(tmp_path, "cut")
    meta = json.loads((run2.dir / "meta.json").read_text())
    assert status2 == "done"
    assert meta["attempts"][1]["resumed_from_step"] == 20 and len(meta["attempts"]) == 2
    for a, b in zip(ref.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)


def test_lr_schedule():
    cfg = {"lr": 1.0, "warmup": 4, "steps": 100, "min_lr_frac": 0.1}
    assert lr_at(0, cfg) == 0.25 and lr_at(3, cfg) == 1.0
    assert lr_at(99, cfg) == pytest.approx(0.1, abs=1e-3)


def test_smoke_script_end_to_end(tmp_path):
    cfg = tmp_path / "e2e.json"
    cfg.write_text(json.dumps({**CFG, "steps": 20}))
    out = subprocess.run([sys.executable, "scripts/train_smoke.py", str(cfg)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("done"), out.stderr
    meta = json.loads((__import__("pathlib").Path("runs") / "e2e" / "meta.json").read_text())
    assert meta["status"] == "done" and "final_loss" in meta["metrics"]
