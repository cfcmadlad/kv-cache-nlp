import numpy as np
import torch

from kvc.bytedata import open_split, write_split
from kvc.patching import patch_flags
from models.blt import BLT
from models.entropy import BOS
from train_blt import batcher

DOCS = ["abcdefghij" * 5, "klmnop" * 9, "q" * 40, "hello world " * 6] * 20


def cfg(mode):
    return {"seed": 0, "batch_size": 3, "block": 64,
            "patching": {"mode": mode, "entropy_run": "none", "max_len": 6, "stride": 4}}


def test_fixed_stride_flags_and_bos_alignment(tmp_path):
    write_split(DOCS, tmp_path, "train")
    data, offsets = open_split(tmp_path, "train")
    (x, flags), y = batcher(data, offsets, None, cfg("fixed"))(0)
    assert x.shape == y.shape == flags.shape == (3, 64)
    assert flags[:, 0].all()
    for r in range(3):
        bos = (x[r] == BOS).nonzero().flatten()
        for t in bos.tolist():
            assert flags[r, t] and (t == 0 or flags[r, t - 1])
    assert (x[:, 1:][x[:, 1:] != BOS] == y[:, :-1][x[:, 1:] != BOS]).all()


def test_entropy_flags_respect_threshold_and_cap():
    h = np.zeros(40, dtype=np.float32)
    h[[5, 9]] = 3.0
    rel = np.arange(40)
    flags = patch_flags(h, rel, np.zeros(40, dtype=bool), "entropy", 1.0, 6, 4)
    starts = np.flatnonzero(flags)
    assert {0, 5, 9} <= set(starts) and np.diff(np.append(starts, 40)).max() <= 6


def test_blt_trains_on_batcher_output(tmp_path):
    write_split(DOCS, tmp_path, "train")
    data, offsets = open_split(tmp_path, "train")
    model = BLT(local_dim=32, local_heads=2, local_window=16, global_dim=48, global_heads=2, global_layers=2)
    batch = batcher(data, offsets, None, cfg("fixed"))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(25):
        x, y = batch(step)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.3
