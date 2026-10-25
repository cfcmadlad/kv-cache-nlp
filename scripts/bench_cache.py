import argparse
import json
import time

import numpy as np
import torch

from data_utils import REPO_ROOT
from eval_blt import load_blt
from kvc.bytedata import open_split
from kvc.cache import Policy
from kvc.patching import patch_flags
from models.entropy import BOS
from train_blt import patch_settings


def cache_elems(state):
    return sum(c["k"].numel() + c["v"].numel() for c in state["glob"]), sum(c["k"].numel() + c["v"].numel() for c in state["enc"] + state["dec"])


@torch.no_grad()
def decode(model, tokens, flags, policy, device, probe):
    state = model.init_state()
    torch.cuda.synchronize() if device == "cuda" else None
    t0, marks = time.perf_counter(), {}
    for i in range(len(tokens)):
        model.step(state, torch.tensor([tokens[i]], device=device), bool(flags[i]), 0.0, policy)
        if i + 1 in probe:
            torch.cuda.synchronize() if device == "cuda" else None
            marks[i + 1] = (time.perf_counter() - t0, *cache_elems(state), state["patches"])
    return marks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--lengths", nargs="+", type=int, default=[2048, 8192, 32768])
    p.add_argument("--budgets", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--out", default="bench.json")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, "val")
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / "val.npy", mmap_mode="r")
    n = max(args.lengths)
    h = entropy[:n].astype(np.float32)
    tokens = np.concatenate([[BOS], np.asarray(data[:n - 1])]).astype(np.int64)
    flags = patch_flags(h, np.arange(n), np.zeros(n, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
    probe = set(args.lengths)
    res = {"run": args.run, "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu", "rows": []}
    decode(model, tokens[:256], flags[:256], None, device, set())
    for name, pol in [("full", None)] + [(f"recency|{b}", Policy("recency", b, 1, 8)) for b in args.budgets]:
        marks = decode(model, tokens[:n], flags[:n], pol, device, probe)
        prev = (0.0, 0)
        for L in sorted(marks):
            t, g, loc, pat = marks[L]
            res["rows"].append({"policy": name, "bytes": L, "patches": pat, "seconds_total": t, "global_elems": g, "local_elems": loc,
                                "global_bytes_fp16": g * 2, "local_bytes_fp16": loc * 2,
                                "ms_per_byte_since_last": 1000 * (t - prev[0]) / (L - prev[1])})
            prev = (t, L)
        print(name, [(L, round(marks[L][0], 2), marks[L][1] * 2) for L in sorted(marks)], flush=True)
    out = REPO_ROOT / "runs" / args.run / args.out
    out.write_text(json.dumps(res), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
