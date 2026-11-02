import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_indices, bpb, paired_diff_ci

SETS = {"tinystories": ("val", 500, 32, ["blt_tinystories_entropy", "blt_tinystories_entropy_s1", "blt_tinystories_entropy_s2"]),
        "wikitext103": ("val", 60, 64, ["blt_wikitext103_entropy", "blt_wikitext103_entropy_s1", "blt_wikitext103_entropy_s2"])}
VARIANTS = ["entropy", "floor3", "floor4", "snap", "hybrid0.25", "hybrid0.5", "dynamic"]


def load(run: str, split: str, docs: int) -> dict:
    r = json.loads((REPO_ROOT / "runs" / run / f"variants_{split}_{docs}.json").read_text(encoding="utf-8"))
    return {"nbytes": np.asarray(r["bytes_per_doc"], dtype=float), "v": {k: (np.asarray(d["nats"]), d["mean_entries"]) for k, d in r["variants"].items()}}


def matched_recency(r: dict, entries: float) -> np.ndarray:
    pts = sorted((d[1], d[0]) for k, d in r["v"].items() if k.startswith("recency|"))
    xs = [x for x, _ in pts]
    if entries <= xs[0]:
        return pts[0][1]
    if entries >= xs[-1]:
        return pts[-1][1]
    j = int(np.searchsorted(xs, entries))
    w = (entries - xs[j - 1]) / (xs[j] - xs[j - 1])
    return (1 - w) * pts[j - 1][1] + w * pts[j][1]


def table(resamples: int, seed: int) -> str:
    lines = ["Variants of entropy-based and attention-based cache policies on the entropy-patched models (three seeds), validation only. "
             f"Paired BPB difference against the comparison policy, mean and 95% document bootstrap interval ({resamples} resamples); negative means the variant is better. "
             "The first patch and the 8 newest are always kept. The dynamic variant lets the entropy of the newest patch set the budget (1.5 times the base budget above the median entropy, 0.5 times below); "
             "its mean cache size is lower than the base budget and its peak is 1.5 times the budget, so it is compared with recency linearly interpolated, per document, to the same mean number of entries.", ""]
    for ds, (split, docs, budget, runs) in SETS.items():
        runs = [r for r in runs if (REPO_ROOT / "runs" / r / f"variants_{split}_{docs}.json").exists()]
        if not runs:
            continue
        lines += [f"## {ds} (budget {budget} patches)", "", "| Variant | Comparison | " + " | ".join(r.rsplit("_", 1)[-1] if r.endswith(("_s1", "_s2")) else "s0" for r in runs) + " | Better in |", "|---|---|" + "---|" * (len(runs) + 1)]
        data = [load(r, split, docs) for r in runs]
        idxs = [bootstrap_indices(len(d["nbytes"]), resamples, seed) for d in data]
        for var in VARIANTS:
            for comp in ("recency", "entropy") if var != "entropy" else ("recency",):
                cells, wins = [], 0
                for d, idx in zip(data, idxs):
                    key = f"{var}|{budget}"
                    ref = matched_recency(d, d["v"][key][1]) if comp == "recency" and var == "dynamic" else d["v"][f"{comp}|{budget}"][0]
                    m, lo, hi = paired_diff_ci(d["v"][key][0], ref, d["nbytes"], idx)
                    cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                    wins += hi < 0
                name = f"{comp} (matched size)" if comp == "recency" and var == "dynamic" else comp
                lines.append(f"| {var} | {name} | " + " | ".join(cells) + f" | {wins}/{len(runs)} |")
        lines += ["", "Mean BPB and mean resident cache entries, seed 0:", "", "| Variant | BPB | Entries |", "|---|---|---|"]
        d0 = data[0]
        for key in ["full|0"] + [f"{v}|{budget}" for v in ["recency"] + VARIANTS] + sorted(k for k in d0["v"] if k.startswith("recency|") and k != f"recency|{budget}"):
            lines.append(f"| {key} | {bpb(d0['v'][key][0], d0['nbytes']):.4f} | {d0['v'][key][1]:.1f} |")
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
