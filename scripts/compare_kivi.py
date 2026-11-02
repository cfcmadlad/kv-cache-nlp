import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_indices, bpb, paired_diff_ci

SETS = {"tinystories": (500, 32, ["blt_tinystories_entropy", "blt_tinystories_entropy_s1", "blt_tinystories_entropy_s2"]),
        "wikitext103": (60, 64, ["blt_wikitext103_entropy", "blt_wikitext103_entropy_s1", "blt_wikitext103_entropy_s2"])}
NAMES = ["sym", "sym_r8", "tok", "kivi", "kivi_r8"]
PAIRS = [("kivi|2|full|0", "sym|3|full|0"), ("kivi|2|full|0", "sym|2|full|0"), ("kivi_r8|2|full|0", "sym_r8|2|full|0"), ("kivi_r8|3|full|0", "sym|3|full|0"),
         ("kivi|2|full|0", "tok|2|full|0"), ("tok|3|full|0", "sym|3|full|0"), ("kivi_r8|3|snap|32", "sym|3|snap|32"), ("kivi_r8|2|snap|32", "sym|3|snap|32")]


def load(run: str, docs: int):
    path = REPO_ROOT / "runs" / run / f"kivi_val_{docs}.json"
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"nbytes": np.asarray(r["bytes_per_doc"], dtype=float), "v": {k: np.asarray(d["nats"]) for k, d in r["variants"].items()}, "cost": {k: d["mean_cache_bytes"] for k, d in r["variants"].items()}}


def key(k: str, budget: int) -> str:
    return k.replace("|32", f"|{budget}") if k.endswith("|32") else k


def table(resamples: int, seed: int) -> str:
    lines = ["KIVI-style asymmetric quantization (per-channel keys in groups of 16 patches, per-token values) against symmetric per-token quantization (sym), "
             "per-token keys and values (tok), and variants keeping the 8 most recent patches in fp16 (_r8). Cost is the mean cache size in bytes per document, "
             "including scales. Eviction rows use the SnapKV-style rule at the stated budget. "
             f"Validation only, mean over three seeds; paired differences use {resamples} document bootstrap resamples, 95% intervals, per seed (negative favours the first entry).", ""]
    for ds, (docs, budget, runs) in SETS.items():
        data = [load(r, docs) for r in runs]
        lines += [f"## {ds} (budget {budget} patches)", "", "| Variant | bits | evict | mean BPB | mean bytes | vs full BPB |", "|---|---|---|---|---|---|"]
        full = np.mean([bpb(d["v"]["full|16|none|0"], d["nbytes"]) for d in data])
        for k in data[0]["v"]:
            k2 = key(k, budget)
            if k2 not in data[0]["v"]:
                continue
            n, b, e, bud = k2.split("|")
            m = np.mean([bpb(d["v"][k2], d["nbytes"]) for d in data])
            c = np.mean([d["cost"][k2] for d in data])
            lines.append(f"| {n} | {b} | {e} | {m:.4f} | {c:,.0f} | {m - full:+.4f} |")
        lines += ["", "| Comparison (a - b) | bytes a / bytes b | " + " | ".join(f"seed {i}" for i in range(len(runs))) + " |", "|---|---|" + "---|" * len(runs)]
        for a, b in PAIRS:
            a, b = key(a, budget), key(b, budget)
            ratio = np.mean([d["cost"][a] / d["cost"][b] for d in data])
            cells = []
            for d in data:
                idx = bootstrap_indices(len(d["nbytes"]), resamples, seed)
                m, lo, hi = paired_diff_ci(d["v"][a], d["v"][b], d["nbytes"], idx)
                cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            lines.append(f"| {a} - {b} | {ratio:.3f} | " + " | ".join(cells) + " |")
        lines += ["", "Byte-matched: each variant against the piecewise-linear interpolation of the sym (or sym_r8) 2, 3 and 4-bit curve at the same mean bytes, means over seeds. Negative favours the variant.", "",
                  "| Variant | bits | evict | mean bytes | variant BPB | baseline BPB at same bytes | difference |", "|---|---|---|---|---|---|---|"]
        for ev, name, ref in (("full", "tok", "sym"), ("full", "kivi", "sym"), ("full", "kivi_r8", "sym_r8"), ("snap", "kivi_r8", "sym")):
            bud = 0 if ev == "full" else budget
            curve = sorted((np.mean([d["cost"][f"{ref}|{b}|{ev}|{bud}"] for d in data]), np.mean([bpb(d["v"][f"{ref}|{b}|{ev}|{bud}"], d["nbytes"]) for d in data])) for b in (2, 3, 4))
            for b in (2, 3, 4):
                k = f"{name}|{b}|{ev}|{bud}"
                c = np.mean([d["cost"][k] for d in data])
                m = np.mean([bpb(d["v"][k], d["nbytes"]) for d in data])
                if not curve[0][0] <= c <= curve[-1][0]:
                    lines.append(f"| {name} | {b} | {ev} | {c:,.0f} | {m:.4f} | outside {ref} range | |")
                    continue
                base = float(np.interp(c, [x for x, _ in curve], [y for _, y in curve]))
                lines.append(f"| {name} | {b} | {ev} | {c:,.0f} | {m:.4f} | {base:.4f} ({ref}) | {m - base:+.4f} |")
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
