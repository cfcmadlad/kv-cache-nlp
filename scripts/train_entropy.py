import sys

import numpy as np
import torch

from data_utils import REPO_ROOT
from kvc.bytedata import open_split
from kvc.config import load_config, run_id
from kvc.runlog import RunLog
from kvc.trainer import fit
from models.entropy import BOS, EntropyLM


def batcher(data, offsets, block: int, batch: int, seed: int, mix=None, frac: float = 0.0):
    n = len(data)

    def get(step: int):
        rng = np.random.default_rng([seed, step])
        pos = rng.integers(1, n - block, size=batch)
        use = rng.random(batch) < frac if mix is not None else np.zeros(batch, dtype=bool)
        if use.any():
            pos = np.where(use, rng.integers(1, len(mix[0]) - block, size=batch), pos)
        x = np.empty((batch, block), dtype=np.int64)
        y = np.empty((batch, block), dtype=np.int64)
        for r, p in enumerate(pos):
            src, offs = mix if use[r] else (data, offsets)
            x[r] = src[p - 1:p + block - 1]
            y[r] = src[p:p + block]
            lo, hi = np.searchsorted(offs, [p, p + block])
            x[r, offs[lo:hi] - p] = BOS
        return torch.from_numpy(x), torch.from_numpy(y)
    return get


def build_model(cfg: dict) -> EntropyLM:
    m = cfg["model"]
    return EntropyLM(m["n_layer"], m["n_head"], m["dim"], m["window"])


def main():
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path, sys.argv[2:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    data, offsets = open_split(REPO_ROOT / "data" / "bytes" / cfg["dataset"], "train")
    mix = open_split(REPO_ROOT / "data" / "bytes" / cfg["mix"]["dataset"], "train") if "mix" in cfg else None
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    run = RunLog(run_id(cfg_path), cfg, REPO_ROOT / "runs")
    (run.dir / "PREEMPTED").unlink(missing_ok=True)
    print(f"params {model.num_params():,}", flush=True)
    status = fit(model, opt, batcher(data, offsets, cfg["block"], cfg["batch_size"], cfg["seed"], mix, cfg.get("mix", {}).get("frac", 0.0)), cfg, run, device)
    if status == "done":
        torch.save({"config": cfg, "state_dict": model.state_dict()}, run.dir / "model.pt")
        run.close("done", cfg["steps"], {"params": model.num_params()})
    print(status, flush=True)


if __name__ == "__main__":
    main()
