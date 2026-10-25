import json

import numpy as np
import torch

from kvc.bytedata import write_split
from kvc.stats import bootstrap_ci, bootstrap_indices, bpb, paired_diff_ci
from models.blt import BLT

DOCS = ["the cat sat on the mat. " * 30, "dogs bark loudly at night. " * 25] * 4


def test_policy_eval_end_to_end(tmp_path, monkeypatch):
    import eval_policies
    root = tmp_path
    (root / "data" / "bytes" / "toy" / "entropy_e").mkdir(parents=True)
    write_split(DOCS, root / "data" / "bytes" / "toy", "val")
    n = sum(len(d) for d in DOCS)
    np.save(root / "data" / "bytes" / "toy" / "entropy_e" / "val.npy", np.random.default_rng(0).random(n).astype(np.float16) * 3)
    cfg = {"dataset": "toy", "block": 256, "patching": {"mode": "fixed", "entropy_run": "e", "max_len": 8, "stride": 4},
           "model": dict(local_dim=32, local_heads=2, local_window=16, enc_layers=1, dec_layers=1, global_dim=48, global_heads=2, global_layers=2)}
    monkeypatch.setattr(eval_policies, "REPO_ROOT", root)
    torch.manual_seed(0)
    model = BLT(**cfg["model"]).eval()
    args = type("A", (), dict(run="x", split="val", max_docs=None, policies=["recency", "random", "entropy", "h2o", "snap"], budgets=[8, 24, 10_000],
                              sink=1, recent=2, agg="mean", seed=0, random_seeds=2, entropy_run=None, signal="entropy_model", dataset=None, attn_iters=2, snap_window=4, online=False, block=None, min_len=0))
    out = eval_policies.run(model, cfg, args, "cpu")
    v = out["variants"]
    assert out["docs"] == len(DOCS) and set(v) >= {"full|0", "recency|8", "random|24", "entropy|10000"}
    assert np.allclose(v["entropy|10000"]["nats"], v["full|0"]["nats"])
    assert not np.allclose(v["recency|8"]["nats"], v["full|0"]["nats"])
    assert v["h2o|8"]["mean_entries"] <= 8 and v["snap|24"]["mean_entries"] <= 24
    assert np.allclose(v["h2o|10000"]["nats"], v["full|0"]["nats"])
    assert v["recency|8"]["mean_entries"] <= 8 < v["full|0"]["mean_entries"]
    json.dumps(out)
    args.online = "all"
    online = eval_policies.run(model, cfg, args, "cpu")["variants"]
    for key in ("recency|8", "random|24", "entropy|8"):
        assert np.allclose(online[key]["nats"], v[key]["nats"], atol=1e-2)
        assert np.isclose(online[key]["mean_entries"], v[key]["mean_entries"])
    assert online["snap|8"]["mean_entries"] <= 8
    assert np.allclose(online["h2o|10000"]["nats"], v["full|0"]["nats"])


def test_combo_eval_matches_policy_eval_and_quant_reduces_bytes(tmp_path, monkeypatch):
    import eval_combo
    import eval_policies
    (tmp_path / "data" / "bytes" / "toy" / "entropy_e").mkdir(parents=True)
    write_split(DOCS, tmp_path / "data" / "bytes" / "toy", "val")
    n = sum(len(d) for d in DOCS)
    np.save(tmp_path / "data" / "bytes" / "toy" / "entropy_e" / "val.npy", np.random.default_rng(0).random(n).astype(np.float16) * 3)
    cfg = {"dataset": "toy", "block": 256, "patching": {"mode": "fixed", "entropy_run": "e", "max_len": 8, "stride": 4},
           "model": dict(local_dim=32, local_heads=2, local_window=16, enc_layers=1, dec_layers=1, global_dim=48, global_heads=2, global_layers=2)}
    monkeypatch.setattr(eval_combo, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(eval_policies, "REPO_ROOT", tmp_path)
    torch.manual_seed(0)
    model = BLT(**cfg["model"]).eval()
    args = type("A", (), dict(run="x", split="val", max_docs=None, policies=["recency", "snap"], budgets=[8], bits=[16, 4], sink=1, recent=2,
                              snap_window=4, random_seeds=1, seed=0))
    v = eval_combo.run(model, cfg, args, "cpu")["variants"]
    ref_args = type("B", (), dict(run="x", split="val", max_docs=None, policies=["recency"], budgets=[8], sink=1, recent=2, agg="mean", seed=0,
                                  random_seeds=1, entropy_run=None, signal="entropy_model", dataset=None, attn_iters=2, snap_window=4,
                                  online=False, block=None, min_len=0))
    ref = eval_policies.run(model, cfg, ref_args, "cpu")["variants"]
    assert np.allclose(v["recency|8|16"]["nats"], ref["recency|8"]["nats"], atol=1e-3)
    assert v["recency|8|4"]["mean_cache_bytes"] < v["recency|8|16"]["mean_cache_bytes"] < v["full|0|16"]["mean_cache_bytes"]
    assert not np.allclose(v["recency|8|4"]["nats"], v["recency|8|16"]["nats"])


def test_bootstrap_stats():
    rng = np.random.default_rng(0)
    nbytes = rng.integers(500, 1500, 200).astype(float)
    a = nbytes * 0.7 * np.log(2) + rng.normal(0, 20, 200)
    b = nbytes * 0.9 * np.log(2) + rng.normal(0, 20, 200)
    idx = bootstrap_indices(200, 500, 1)
    lo, hi = bootstrap_ci(a, nbytes, idx)
    assert lo < bpb(a, nbytes) < hi
    mean, dlo, dhi = paired_diff_ci(a, b, nbytes, idx)
    assert dhi < 0 and abs(mean + 0.2) < 0.02
