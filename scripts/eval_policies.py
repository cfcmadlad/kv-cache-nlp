import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import GLOBAL_CACHE_BYTES_PER_PATCH, load_blt
from kvc.bytedata import open_split
from kvc.cache import Policy, mean_cache_entries, patch_entropy_scores, patch_scores, visible_mask
from kvc.patching import patch_flags
from models.entropy import BOS
from train_blt import patch_settings


@torch.no_grad()
def run(model, cfg, args, device):
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, args.split)
    entropy_run = args.entropy_run or cfg["patching"]["entropy_run"]
    entropy = np.load(base / f"entropy_{entropy_run}" / f"{args.split}.npy", mmap_mode="r")
    count = len(offsets) - 1 if args.max_docs is None else min(args.max_docs, len(offsets) - 1)
    docs = [i for i in range(count) if offsets[i + 1] > offsets[i]]
    variants = [("full", 0)] + [(m, b) for m in args.policies for b in args.budgets]
    nats = {v: np.zeros(len(docs)) for v in variants}
    entries = {v: [] for v in variants}
    sizes = np.zeros(len(docs), dtype=np.int64)
    block = cfg["block"]
    for di, i in enumerate(docs):
        s, e = offsets[i], offsets[i + 1]
        doc, h = np.asarray(data[s:e]), entropy[s:e].astype(np.float32)
        sizes[di] = len(doc)
        for a in range(0, len(doc), block):
            b = min(a + block, len(doc))
            x = np.empty(b - a, dtype=np.int64)
            x[0] = BOS if a == 0 else doc[a - 1]
            x[1:] = doc[a:b - 1]
            flags = patch_flags(h[a:b] if ps["mode"] == "entropy" else None, np.arange(a, b), np.zeros(b - a, dtype=bool),
                                ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
            cs = np.cumsum(flags)
            patch_in = cs - flags
            P = int(cs[-1]) + 1
            ent = patch_entropy_scores(patch_in, h[a:b].astype(np.float64), P, args.agg)
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            target = torch.from_numpy(doc[a:b].astype(np.int64)).to(device)
            chunk_full = None
            for v in variants:
                mode, budget = v
                if mode == "full" or budget >= P:
                    if chunk_full is None:
                        logits = model(inputs)[0][0]
                        chunk_full = F.cross_entropy(logits.float(), target, reduction="sum").item()
                    nats[v][di] += chunk_full
                    entries[v].append(P)
                    continue
                pol = Policy(mode, budget, args.sink, args.recent, args.agg, args.seed)
                reps = args.random_seeds if mode == "random" else 1
                total = 0.0
                for r in range(reps):
                    rng = np.random.default_rng([args.seed, r, di, a])
                    mask = visible_mask(P, patch_scores(pol, P, ent, rng), pol)
                    logits = model((*inputs, torch.from_numpy(mask)[None].to(device)))[0][0]
                    total += F.cross_entropy(logits.float(), target, reduction="sum").item()
                nats[v][di] += total / reps
                entries[v].append(mean_cache_entries(mask))
    per_patch = GLOBAL_CACHE_BYTES_PER_PATCH(cfg["model"])
    return {"run": args.run, "dataset": cfg["dataset"], "split": args.split, "docs": len(docs),
            "bytes_per_doc": sizes.tolist(), "bytes_per_patch": per_patch,
            "settings": {"sink": args.sink, "recent": args.recent, "agg": args.agg, "seed": args.seed,
                         "random_seeds": args.random_seeds, "entropy_run": entropy_run},
            "variants": {f"{m}|{b}": {"nats": nats[(m, b)].tolist(), "mean_entries": float(np.mean(entries[(m, b)])),
                                      "mean_cache_bytes": float(np.mean(entries[(m, b)])) * per_patch}
                         for m, b in variants}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--policies", nargs="+", default=["recency", "random", "entropy", "entropy_inv"])
    p.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512])
    p.add_argument("--sink", type=int, default=1)
    p.add_argument("--recent", type=int, default=8)
    p.add_argument("--agg", default="mean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-seeds", type=int, default=3)
    p.add_argument("--entropy-run", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    result = run(model, cfg, args, device)
    out = REPO_ROOT / "runs" / args.run / (args.out or f"policies_{args.split}_{result['docs']}.json")
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
