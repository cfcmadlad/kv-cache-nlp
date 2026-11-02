import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_indices, bpb, paired_diff_ci

SETS = {"tinystories": (500, 32, ["blt_tinystories_entropy", "blt_tinystories_entropy_s1", "blt_tinystories_entropy_s2"]),
        "wikitext103": (60, 64, ["blt_wikitext103_entropy", "blt_wikitext103_entropy_s1", "blt_wikitext103_entropy_s2"])}
PAIRS = [("snap_pyramid", "snap_uniform"), ("snap_inverse", "snap_uniform"), ("snap_pyramid", "snap_inverse"), ("recency_pyramid", "recency_uniform"),
         ("recency_inverse", "recency_uniform"), ("snap_pyramid", "recency_uniform"), ("snap_uniform", "recency_uniform")]


def load(run: str, docs: int):
    r = json.loads((REPO_ROOT / "runs" / run / f"pyramid_val_{docs}.json").read_text(encoding="utf-8"))
    return {"nbytes": np.asarray(r["bytes_per_doc"], dtype=float), "v": {k: np.asarray(d["nats"]) for k, d in r["variants"].items()}, "entries": {k: d.get("mean_entries") for k, d in r["variants"].items()}, "sched": r["schedules"]}


def table(resamples: int, seed: int) -> str:
    lines = ["Per-layer cache budgets (PyramidKV-style). Every schedule has the same mean budget across the six global layers; the pyramid gives lower layers more patches, "
             "the inverse pyramid the opposite. Mean entries is the realized mean number of cached patch entries summed over layers per query. "
             f"Validation only; paired differences use {resamples} document bootstrap resamples, 95% intervals, per seed (negative favours the first entry).", ""]
    for ds, (docs, budget, runs) in SETS.items():
        data = [load(r, docs) for r in runs]
        lines += [f"## {ds} (mean budget {budget} patches)", "", "Schedules (layer 0 to 5): " + "; ".join(f"{k}: {v}" for k, v in data[0]["sched"].items()), "",
                  "| Schedule | mean BPB | mean entries |", "|---|---|---|"]
        for k in data[0]["sched"]:
            kk = f"{k}|{budget}"
            lines.append(f"| {k} | {np.mean([bpb(d['v'][kk], d['nbytes']) for d in data]):.4f} | {np.mean([d['entries'][kk] for d in data]):.1f} |")
        lines.append(f"| full | {np.mean([bpb(d['v'][f'full|{budget}'], d['nbytes']) for d in data]):.4f} | |")
        lines += ["", "| Comparison (a - b) | " + " | ".join(f"seed {i}" for i in range(len(runs))) + " |", "|---|" + "---|" * len(runs)]
        for a, b in PAIRS:
            cells = []
            for d in data:
                idx = bootstrap_indices(len(d["nbytes"]), resamples, seed)
                m, lo, hi = paired_diff_ci(d["v"][f"{a}|{budget}"], d["v"][f"{b}|{budget}"], d["nbytes"], idx)
                cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            lines.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
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
