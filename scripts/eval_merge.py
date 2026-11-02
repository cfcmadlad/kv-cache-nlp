import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from kvc.bytedata import open_split
from kvc.cache import patch_entropy_scores
from kvc.patching import patch_flags
from models.entropy import BOS
from train_blt import patch_settings

MODES = ("evict", "weak", "random", "fixed")


@torch.no_grad()
def run(model, cfg, args, device):
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, args.split)
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    count = min(args.max_docs, len(offsets) - 1)
    variants = [("full", 0)] + [(m, args.budget) for m in MODES]
    nats = {v: np.zeros(count) for v in variants}
    sizes = np.zeros(count, dtype=np.int64)
    block = cfg["block"]
    for di in range(count):
        s, e = offsets[di], offsets[di + 1]
        doc, h = np.asarray(data[s:e]), entropy[s:e].astype(np.float32)
        sizes[di] = len(doc)
        for a in range(0, len(doc), block):
            b = min(a + block, len(doc))
            x = np.empty(b - a, dtype=np.int64)
            x[0] = BOS if a == 0 else doc[a - 1]
            x[1:] = doc[a:b - 1]
            flags = patch_flags(h[a:b], np.arange(a, b), np.zeros(b - a, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
            cs = np.cumsum(flags)
            patch_in = cs - flags
            P = int(cs[-1]) + 1
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            tgt = torch.from_numpy(doc[a:b].astype(np.int64)).to(device)
            lengths = np.bincount(patch_in, minlength=P)
            margins = patch_entropy_scores(patch_in, h[a:b].astype(np.float64), P, "first") - ps["theta"]
            for v in variants:
                name, budget = v
                if name == "full" or budget >= P:
                    logits = model(inputs)[0][0]
                else:
                    rng = np.random.default_rng([args.seed, di, a])
                    logits = model.forward_merge(inputs, budget, name, margins, lengths, 1, 8, rng)[0][0]
                nats[v][di] += F.cross_entropy(logits.float(), tgt, reduction="sum").item()
    return {"run": args.run, "dataset": cfg["dataset"], "split": args.split, "docs": count, "bytes_per_doc": sizes.tolist(),
            "variants": {f"{n}|{b}": {"nats": nats[(n, b)].tolist(), "mean_entries": float(args.budget if n != "full" else 0)} for n, b in variants}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    result = run(model, cfg, args, device)
    out = REPO_ROOT / "runs" / args.run / f"merge_{args.split}_{result['docs']}.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
