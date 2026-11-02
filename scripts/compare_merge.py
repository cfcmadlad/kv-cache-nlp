import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_indices, bpb, paired_diff_ci

SETS = {"tinystories": (500, 32, ["blt_tinystories_entropy", "blt_tinystories_entropy_s1", "blt_tinystories_entropy_s2"]),
        "wikitext103": (60, 64, ["blt_wikitext103_entropy", "blt_wikitext103_entropy_s1", "blt_wikitext103_entropy_s2"])}
MODES = ["weak", "random", "fixed"]


def load(run: str, docs: int):
    path = REPO_ROOT / "runs" / run / f"merge_val_{docs}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"nbytes": np.asarray(r["bytes_per_doc"], dtype=float), "v": {k: np.asarray(d["nats"]) for k, d in r["variants"].items()}}


def table(resamples: int, seed: int) -> str:
    lines = ["Merging adjacent cache entries instead of evicting them, at equal numbers of entries (training-free). When the cache is full, the two adjacent old entries "
             "are replaced by their byte-count-weighted average (keys and values, every layer): `weak` merges the pair whose shared boundary had the smallest entropy margin "
             "above the patching threshold (the weakest boundary is undone), `random` a random adjacent pair, `fixed` the oldest adjacent pair. `evict` drops the oldest entry (recency). "
             "The first patch and the 8 newest are never touched. Validation only; paired 95% document bootstrap intervals "
             f"({resamples} resamples); negative means the merge rule is better than the comparison.", ""]
    for ds, (docs, budget, runs) in SETS.items():
        runs = [r for r in runs if load(r, docs) is not None]
        if not runs:
            continue
        data = [load(r, docs) for r in runs]
        idxs = [bootstrap_indices(len(d["nbytes"]), resamples, seed) for d in data]
        lines += [f"## {ds} ({docs} documents, budget {budget} entries)", "", "| Rule | Comparison | " + " | ".join(f"seed {i}" for i in range(len(runs))) + " | Better in |", "|---|---|" + "---|" * (len(runs) + 1)]
        for mode in MODES:
            for comp in ("evict", "random") if mode != "random" else ("evict",):
                cells, wins = [], 0
                for d, idx in zip(data, idxs):
                    m, lo, hi = paired_diff_ci(d["v"][f"{mode}|{budget}"], d["v"][f"{comp}|{budget}"], d["nbytes"], idx)
                    cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                    wins += hi < 0
                lines.append(f"| {mode} | {comp} | " + " | ".join(cells) + f" | {wins}/{len(runs)} |")
        lines += ["", "| Rule | " + " | ".join(f"BPB seed {i}" for i in range(len(runs))) + " |", "|---|" + "---|" * len(runs)]
        for name in ["full|0"] + [f"{m}|{budget}" for m in ["evict"] + MODES]:
            lines.append(f"| {name} | " + " | ".join(f"{bpb(d['v'][name], d['nbytes']):.4f}" for d in data) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    text = table(args.resamples, args.seed)
    if args.out:
        (REPO_ROOT / args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
