import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_ci, bootstrap_indices, bpb, paired_diff_ci

PAIRS = [("entropy", "random"), ("entropy", "recency"), ("entropy", "entropy_inv"), ("recency", "random")]
TIERED = ("recency", "random", "entropy", "entropy_inv")


def analyse(result: dict, resamples: int, seed: int) -> str:
    nbytes = np.asarray(result["bytes_per_doc"], dtype=float)
    idx = bootstrap_indices(len(nbytes), resamples, seed)
    v = {k: {**d, "nats": np.asarray(d["nats"])} for k, d in result["variants"].items()}
    full = v["full|16"]
    s = result["settings"]
    lines = [f"{result['run']} on {result['dataset']} {result['split']} ({result['docs']} docs), global KV quantization, "
             f"high {s['hi_bits']} bits, low {s['lo_bits']} bits, {resamples} document bootstrap resamples, 95% intervals.", "",
             "| Variant | Param | Mean cache KB (% of fp16) | BPB [95% CI] | Cost over fp16 |", "|---|---|---|---|---|"]
    for key, d in v.items():
        mode, param = key.split("|")
        lo, hi = bootstrap_ci(d["nats"], nbytes, idx)
        lines.append(f"| {mode} | {param} | {d['mean_cache_bytes'] / 1000:,.1f} ({100 * d['mean_cache_bytes'] / full['mean_cache_bytes']:.0f}%) | "
                     f"{bpb(d['nats'], nbytes):.4f} [{lo:.4f}, {hi:.4f}] | {bpb(d['nats'], nbytes) - bpb(full['nats'], nbytes):+.4f} |")
    modes = {k.split("|")[0] for k in v}
    pairs = [(a, c) for a, c in PAIRS if a in modes and c in modes]
    fracs = sorted({k.split("|")[1] for k in v if k.split("|")[0] in TIERED}, key=float)
    lines += ["", "Paired BPB difference at the same high-precision fraction (negative means the first is better):", "",
              "| Fraction | " + " | ".join(f"{a} - {c}" for a, c in pairs) + " |", "|---|" + "---|" * len(pairs)]
    for f in fracs:
        cells = []
        for a, c in pairs:
            m, lo, hi = paired_diff_ci(v[f"{a}|{f}"]["nats"], v[f"{c}|{f}"]["nats"], nbytes, idx)
            cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + (" *" if hi < 0 or lo > 0 else ""))
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines += ["", "* the interval excludes zero. Cache bytes differ slightly between policies at the same fraction; see the table above."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    text = analyse(json.loads((REPO_ROOT / args.file).read_text(encoding="utf-8")), args.resamples, args.seed)
    print(text)
    if args.out:
        (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
