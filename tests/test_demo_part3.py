import json

import torch

import demo_part3
from models.blt import BLT
from models.entropy import EntropyLM
from test_blt import SMALL

TEXTS = ["the cat sat on the mat. " * 12, "dogs bark at night. " * 14] * 4

def test_live_training_reduces_loss_and_evaluation_rows(monkeypatch):
    monkeypatch.setattr(demo_part3, "TINY", SMALL)
    logs = []
    model = demo_part3.train_live(TEXTS, 60, "cpu", block=64, batch=4, log=logs.append)
    first, last = (float(l.split("loss ")[1].split()[0]) for l in (logs[1], logs[-1]))
    assert last < first
    rows = demo_part3.evaluate_live(model, TEXTS, 12, "cpu")
    assert [r[0] for r in rows][0] == "full cache" and len(rows) == 4
    assert all(r[1] > 0 for r in rows) and rows[1][2] < rows[0][2]

def test_stream_compare_bounds_cache_and_matches_full_at_large_budget(tmp_path, monkeypatch):
    import train_blt
    (tmp_path / "runs" / "e").mkdir(parents=True)
    (tmp_path / "runs" / "e" / "patching.json").write_text(json.dumps({"threshold": 0.01}))
    monkeypatch.setattr(train_blt, "REPO_ROOT", tmp_path)
    cfg = {"patching": {"mode": "entropy", "entropy_run": "e", "max_len": 8, "stride": 4}, "model": SMALL}
    torch.manual_seed(0)
    model, entropy = BLT(**SMALL).eval(), EntropyLM(1, 2, 32, 16).eval()
    text = "the cat sat on the mat. " * 12
    small = demo_part3.stream_compare(text, model, entropy, cfg, 10, "cpu")
    large = demo_part3.stream_compare(text, model, entropy, cfg, 10_000, "cpu")
    assert small["rows"][1][2] < small["rows"][0][2]
    assert abs(large["rows"][1][1] - large["rows"][0][1]) < 1e-3
