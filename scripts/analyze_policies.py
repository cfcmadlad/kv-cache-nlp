import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_ci, bootstrap_indices, bpb, paired_diff_ci

PAIRS = [("entropy", "random"), ("entropy", "recency"), ("entropy", "entropy_inv"), ("recency", "random")]


def analyse(result: dict, resamples: int, seed: int) -> str:
    nbytes = np.asarray(result["bytes_per_doc"], dtype=float)
    idx = bootstrap_indices(len(nbytes), resamples, seed)
    v = {k: {**d, "nats": np.asarray(d["nats"])} for k, d in result["variants"].items()}
    full = v["full|0"]
    lo, hi = bootstrap_ci(full["nats"], nbytes, idx)
    lines = [f"{result['run']} on {result['dataset']} {result['split']} ({result['docs']} docs, {int(nbytes.sum()):,} bytes), "
             f"{resamples} document bootstrap resamples, 95% intervals.", "",
             f"Full cache: BPB {bpb(full['nats'], nbytes):.4f} [{lo:.4f}, {hi:.4f}], {full['mean_cache_bytes'] / 1000:,.1f} KB mean global cache per chunk", "",
             "| Policy | Budget (patches) | Mean cache KB | BPB [95% CI] | Cost over full |", "|---|---|---|---|---|"]
    budgets = sorted({int(k.split("|")[1]) for k in v if k != "full|0"})
    modes = sorted({k.split("|")[0] for k in v if k != "full|0"})
    for b in budgets:
        for m in modes:
            d = v[f"{m}|{b}"]
            lo, hi = bootstrap_ci(d["nats"], nbytes, idx)
            lines.append(f"| {m} | {b} | {d['mean_cache_bytes'] / 1000:,.1f} | {bpb(d['nats'], nbytes):.4f} [{lo:.4f}, {hi:.4f}] | "
                         f"{bpb(d['nats'], nbytes) - bpb(full['nats'], nbytes):+.4f} |")
    lines += ["", "Paired BPB difference (first minus second, negative means the first is better), mean [95% CI]:", "",
              "| Budget | " + " | ".join(f"{a} - {b}" for a, b in PAIRS if a in modes and b in modes) + " |",
              "|---|" + "---|" * sum(1 for a, b in PAIRS if a in modes and b in modes)]
    for b in budgets:
        cells = []
        for a, c in PAIRS:
            if a in modes and c in modes:
                m, l, h = paired_diff_ci(v[f"{a}|{b}"]["nats"], v[f"{c}|{b}"]["nats"], nbytes, idx)
                cells.append(f"{m:+.4f} [{l:+.4f}, {h:+.4f}]" + (" *" if h < 0 or l > 0 else ""))
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines += ["", "* the interval excludes zero."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    result = json.loads((REPO_ROOT / args.file).read_text(encoding="utf-8"))
    text = analyse(result, args.resamples, args.seed)
    print(text)
    if args.out:
        (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
