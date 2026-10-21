import json

import torch

from models.blt import BLT
from models.entropy import EntropyLM
from test_blt import SMALL


def test_demo_analysis_rows(tmp_path, monkeypatch):
    import cache_demo
    import train_blt
    (tmp_path / "runs" / "e").mkdir(parents=True)
    (tmp_path / "runs" / "e" / "patching.json").write_text(json.dumps({"threshold": 0.01}))
    monkeypatch.setattr(train_blt, "REPO_ROOT", tmp_path)
    cfg = {"patching": {"mode": "entropy", "entropy_run": "e", "max_len": 8, "stride": 4}, "model": SMALL}
    torch.manual_seed(0)
    model, entropy = BLT(**SMALL).eval(), EntropyLM(1, 2, 32, 16).eval()
    out = cache_demo.analyse("the cat sat on the mat. " * 8, model, entropy, cfg, 12, "cpu")
    names = [r[0] for r in out["rows"]]
    assert names[0] == "full cache (fp16)" and len(names) == 7
    assert out["marked"].replace("|", "") == ("the cat sat on the mat. " * 8)[:out["bytes"]]
    assert all(r[1] > 0 and r[2] > 0 for r in out["rows"])
    assert out["rows"][1][2] < out["rows"][0][2]
