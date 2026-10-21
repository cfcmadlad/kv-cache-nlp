import json
import sys

import numpy as np
import torch

from data_utils import REPO_ROOT
from kvc.bytedata import open_split
from kvc.config import load_config, run_id
from kvc.patching import patch_flags
from kvc.runlog import RunLog
from kvc.trainer import fit
from models.blt import BLT
from models.entropy import BOS


def patch_settings(cfg: dict) -> dict:
    p = cfg["patching"]
    theta = None
    if p["mode"] == "entropy":
        theta = json.loads((REPO_ROOT / "runs" / p["entropy_run"] / "patching.json").read_text())["threshold"]
    return {"mode": p["mode"], "theta": theta, "max_len": p.get("max_len", 16), "stride": p.get("stride", 4)}


def batcher(data, offsets, entropy, cfg: dict, mix=None):
    n, block, batch, seed = len(data), cfg["block"], cfg["batch_size"], cfg["seed"]
    ps = patch_settings(cfg)

    def get(step: int):
        rng = np.random.default_rng([seed, step])
        pos = rng.integers(1, n - block, size=batch)
        use = rng.random(batch) < cfg["mix"]["frac"] if mix is not None else np.zeros(batch, dtype=bool)
        if use.any():
            pos = np.where(use, rng.integers(1, len(mix[0]) - block, size=batch), pos)
        x = np.empty((batch, block), dtype=np.int64)
        y = np.empty((batch, block), dtype=np.int64)
        flags = np.empty((batch, block), dtype=bool)
        for r, p in enumerate(pos):
            src, offs, ent = mix if use[r] else (data, offsets, entropy)
            q = np.arange(p, p + block)
            doc = np.searchsorted(offs, q, side="right") - 1
            rel = q - offs[doc]
            x[r], y[r] = src[p - 1:p + block - 1], src[p:p + block]
            x[r, rel == 0] = BOS
            forced = rel == 0
            forced[:-1] |= rel[1:] == 0
            h = ent[p:p + block].astype(np.float32) if ent is not None else None
            flags[r] = patch_flags(h, rel, forced, ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
        return (torch.from_numpy(x), torch.from_numpy(flags)), torch.from_numpy(y)
    return get


def build_model(cfg: dict) -> BLT:
    return BLT(**cfg["model"])


def main():
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path, sys.argv[2:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, "train")
    entropy = None
    if cfg["patching"]["mode"] == "entropy":
        entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / "train.npy", mmap_mode="r")
    mix = None
    if "mix" in cfg:
        mbase = REPO_ROOT / "data" / "bytes" / cfg["mix"]["dataset"]
        mdata, moffsets = open_split(mbase, "train")
        ment = np.load(mbase / f"entropy_{cfg['patching']['entropy_run']}" / "train.npy", mmap_mode="r") if entropy is not None else None
        mix = (mdata, moffsets, ment)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    run = RunLog(run_id(cfg_path), cfg, REPO_ROOT / "runs")
    (run.dir / "PREEMPTED").unlink(missing_ok=True)
    print(f"params {model.num_params():,}", flush=True)
    status = fit(model, opt, batcher(data, offsets, entropy, cfg, mix), cfg, run, device)
    if status == "done":
        torch.save({"config": cfg, "state_dict": model.state_dict()}, run.dir / "model.pt")
        run.close("done", cfg["steps"], {"params": model.num_params()})
    print(status, flush=True)


if __name__ == "__main__":
    main()
