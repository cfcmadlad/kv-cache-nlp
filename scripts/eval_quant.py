import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from kvc.bytedata import open_split
from kvc.cache import QuantPolicy, mean_entry_bytes, patch_entropy_scores, tier_mask
from kvc.patching import patch_flags
from kvc.quant import entry_bytes
from models.entropy import BOS
from train_blt import patch_settings


@torch.no_grad()
def run(model, cfg, args, device):
    ps = patch_settings(cfg)
    m = cfg["model"]
    per_bits = lambda b: entry_bytes(m["global_layers"], m["global_dim"], m["global_heads"], b)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, args.split)
    entropy_run = args.entropy_run or cfg["patching"]["entropy_run"]
    entropy = np.load(base / f"entropy_{entropy_run}" / f"{args.split}.npy", mmap_mode="r")
    count = len(offsets) - 1 if args.max_docs is None else min(args.max_docs, len(offsets) - 1)
    docs = [i for i in range(count) if offsets[i + 1] > offsets[i]]
    variants = [("full", "16")] + [("uniform", str(b)) for b in args.uniform_bits] + \
        [(mode, str(f)) for mode in args.policies for f in args.fracs]
    nats = {v: np.zeros(len(docs)) for v in variants}
    cache_bytes = {v: [] for v in variants}
    sizes = np.zeros(len(docs), dtype=np.int64)
    for di, i in enumerate(docs):
        s, e = offsets[i], offsets[i + 1]
        doc, h = np.asarray(data[s:e]), entropy[s:e].astype(np.float32)
        sizes[di] = len(doc)
        for a in range(0, len(doc), cfg["block"]):
            b = min(a + cfg["block"], len(doc))
            x = np.empty(b - a, dtype=np.int64)
            x[0] = BOS if a == 0 else doc[a - 1]
            x[1:] = doc[a:b - 1]
            flags = patch_flags(h[a:b] if ps["mode"] == "entropy" else None, np.arange(a, b), np.zeros(b - a, dtype=bool),
                                ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
            cs = np.cumsum(flags)
            P = int(cs[-1]) + 1
            ent = patch_entropy_scores(cs - flags, h[a:b].astype(np.float64), P, args.agg)
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            target = torch.from_numpy(doc[a:b].astype(np.int64)).to(device)

            def score(mask, hb, lb):
                extra = () if mask is None else (None, (torch.from_numpy(mask)[None].to(device), hb, lb))
                logits = model((*inputs, *extra))[0][0]
                return F.cross_entropy(logits.float(), target, reduction="sum").item()

            for v in variants:
                mode, param = v
                if mode == "full":
                    nats[v][di] += score(None, 16, 16)
                    cache_bytes[v].append(float(np.mean(np.arange(1, P + 1)) * per_bits(16)))
                elif mode == "uniform":
                    nats[v][di] += score(np.zeros((P, P), dtype=bool), 16, int(param))
                    cache_bytes[v].append(float(np.mean(np.arange(1, P + 1)) * per_bits(int(param))))
                else:
                    qp = QuantPolicy(mode, float(param), args.hi_bits, args.lo_bits, math.ceil(float(param) * P), 1, args.agg, args.seed)
                    reps = args.random_seeds if mode == "random" else 1
                    total = 0.0
                    for r in range(reps):
                        mask = tier_mask(qp, P, ent, np.random.default_rng([args.seed, r, di, a]))
                        total += score(mask, args.hi_bits, args.lo_bits)
                    nats[v][di] += total / reps
                    cache_bytes[v].append(mean_entry_bytes(qp, mask, per_bits))
    return {"run": args.run, "dataset": cfg["dataset"], "split": args.split, "docs": len(docs),
            "bytes_per_doc": sizes.tolist(),
            "settings": {"hi_bits": args.hi_bits, "lo_bits": args.lo_bits, "agg": args.agg, "seed": args.seed,
                         "random_seeds": args.random_seeds, "entropy_run": entropy_run},
            "variants": {f"{m_}|{p_}": {"nats": nats[(m_, p_)].tolist(), "mean_cache_bytes": float(np.mean(cache_bytes[(m_, p_)]))}
                         for m_, p_ in variants}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--policies", nargs="+", default=["recency", "random", "entropy", "entropy_inv"])
    p.add_argument("--fracs", nargs="+", type=float, default=[0.125, 0.25, 0.5])
    p.add_argument("--uniform-bits", nargs="+", type=int, default=[8, 4, 3, 2])
    p.add_argument("--hi-bits", type=int, default=8)
    p.add_argument("--lo-bits", type=int, default=2)
    p.add_argument("--agg", default="mean")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-seeds", type=int, default=3)
    p.add_argument("--entropy-run", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    result = run(model, cfg, args, device)
    out = REPO_ROOT / "runs" / args.run / (args.out or f"quant_{args.split}_{result['docs']}.json")
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
