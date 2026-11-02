import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_indices, bpb, paired_diff_ci

SETS = {"tinystories": (500, 32, "blt_tinystories_{}"), "wikitext103": (60, 64, "blt_wikitext103_{}")}
REGIMES = ["none", "recency", "random", "entropy", "mixed"]
POLICIES = ["recency", "random", "entropy", "snap"]
SUFFIXES = ["", "_s1", "_s2"]


def load(run: str, docs: int, split: str = "val"):
    path = REPO_ROOT / "runs" / run / f"policies_{split}_{docs}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"nbytes": np.asarray(r["bytes_per_doc"], dtype=float), "v": {k: np.asarray(d["nats"]) for k, d in r["variants"].items()}}


def run_name(ds: str, regime: str, suffix: str) -> str:
    base = SETS[ds][2]
    return base.format("entropy" + suffix) if regime == "none" else base.format(f"eat_{regime}{suffix}")


def table(resamples: int, seed: int, split: str = "val") -> str:
    lines = ["Eviction-aware training: models trained with a policy applied to the cache (regimes) and evaluated under every policy, " + ("validation only, " if split == "val" else "test split, evaluated once, ") +
             "mean over the available seeds. 'none' is the standard model trained with a full cache. "
             f"Paired differences use {resamples} document bootstrap resamples, 95% intervals, per seed.", ""]
    for ds, (docs, budget, _) in SETS.items():
        if split == "test":
            if ds != "wikitext103":
                continue
            docs = 62
        lines += [f"## {ds} (budget {budget} patches)", "", "| Training regime | Seeds | full | recency | random | entropy | snap | entropy - recency | entropy - random |", "|---|---|---|---|---|---|---|---|---|"]
        for reg in REGIMES:
            rs = [r for r in (load(run_name(ds, reg, s), docs, split) for s in SUFFIXES) if r is not None]
            if not rs:
                lines.append(f"| {reg} | 0 | not run | | | | | | |")
                continue
            cells = [f"{np.mean([bpb(r['v']['full|0'], r['nbytes']) for r in rs]):.4f}"]
            for pol in POLICIES:
                vals = [bpb(r["v"][f"{pol}|{budget}"], r["nbytes"]) for r in rs if f"{pol}|{budget}" in r["v"]]
                cells.append(f"{np.mean(vals):.4f}" if vals else "n/a")
            diffs = []
            for ref in ("recency", "random"):
                ds_ = []
                for r in rs:
                    idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
                    m, lo, hi = paired_diff_ci(r["v"][f"entropy|{budget}"], r["v"][f"{ref}|{budget}"], r["nbytes"], idx)
                    ds_.append((m, lo, hi))
                worse = sum(lo > 0 for _, lo, _ in ds_)
                better = sum(hi < 0 for _, _, hi in ds_)
                diffs.append(f"{np.mean([m for m, _, _ in ds_]):+.4f} (worse {worse}, better {better} of {len(rs)})")
            lines.append(f"| {reg} | {len(rs)} | " + " | ".join(cells) + " | " + " | ".join(diffs) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="val")
    args = p.parse_args()
    text = table(args.resamples, args.seed, args.split)
    if args.out:
        (REPO_ROOT / args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
