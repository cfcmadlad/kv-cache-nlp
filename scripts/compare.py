import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_ci, bootstrap_indices, bpb, paired_diff_ci

SETS = {"tinystories": ("val", 500, [32, 64]), "wikitext103": ("val", 60, [64, 128])}
GROUPS = {"entropy patching, seeds 0 to 2": ["blt_{}_entropy", "blt_{}_entropy_s1", "blt_{}_entropy_s2"],
          "fixed stride 4": ["blt_{}_fixed"], "fixed stride 5": ["blt_{}_fixed5"]}
PAIRS = [("entropy", "random"), ("entropy", "recency"), ("entropy", "entropy_inv")]


def load(run: str, split: str, docs: int) -> dict:
    path = REPO_ROOT / "runs" / run / f"policies_{split}_{docs}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"nbytes": np.asarray(r["bytes_per_doc"], dtype=float),
            "v": {k: np.asarray(d["nats"]) for k, d in r["variants"].items()}}


def seeds_table(resamples: int, seed: int) -> str:
    lines = ["Paired BPB difference of keeping high-entropy patches minus the comparison policy (positive means keeping high entropy is worse), "
             f"mean and 95% document bootstrap interval ({resamples} resamples), one row per trained model. Validation only.", ""]
    for ds, (split, docs, budgets) in SETS.items():
        lines += [f"## {ds}", "", "| Model | Full-cache BPB [95% CI] | Budget | recency BPB | random BPB | entropy BPB | entropy - random | entropy - recency | entropy - entropy_inv |",
                  "|---|---|---|---|---|---|---|---|---|"]
        tally = {}
        for group, runs in GROUPS.items():
            for template in runs:
                run = template.format(ds)
                r = load(run, split, docs)
                if r is None:
                    lines.append(f"| {run} | not run | | | | | | | |")
                    continue
                idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
                lo, hi = bootstrap_ci(r["v"]["full|0"], r["nbytes"], idx)
                for b in budgets:
                    cells = []
                    for a, c in PAIRS:
                        m, l, h = paired_diff_ci(r["v"][f"{a}|{b}"], r["v"][f"{c}|{b}"], r["nbytes"], idx)
                        cells.append(f"{m:+.4f} [{l:+.4f}, {h:+.4f}]")
                        tally.setdefault((group, b, c), []).append(l > 0)
                    lines.append(f"| {run} | {bpb(r['v']['full|0'], r['nbytes']):.4f} [{lo:.4f}, {hi:.4f}] | {b} | "
                                 + " | ".join(f"{bpb(r['v'][f'{m}|{b}'], r['nbytes']):.4f}" for m in ("recency", "random", "entropy"))
                                 + " | " + " | ".join(cells) + " |")
        lines += ["", "Models for which keeping high entropy is worse with an interval above zero:", "",
                  "| Group | Budget | vs random | vs recency | vs entropy_inv |", "|---|---|---|---|---|"]
        for group in GROUPS:
            for b in budgets:
                cells = [f"{sum(tally[(group, b, c)])}/{len(tally[(group, b, c)])}" if (group, b, c) in tally else "n/a" for _, c in PAIRS]
                lines.append(f"| {group} | {b} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("table", choices=["seeds"])
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    text = seeds_table(args.resamples, args.seed)
    print(text)
    if args.out:
        (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
