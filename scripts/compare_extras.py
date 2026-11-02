import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_indices, bpb, paired_diff_ci

SETS = {"tinystories": (500, 32, ["blt_tinystories_entropy", "blt_tinystories_entropy_s1", "blt_tinystories_entropy_s2"]),
        "wikitext103": (60, 64, ["blt_wikitext103_entropy", "blt_wikitext103_entropy_s1", "blt_wikitext103_entropy_s2"])}
VARIANTS = ["entropy", "surprisal", "keydiff", "expected", "snap", "snapv"]


def load(run: str, docs: int):
    path = REPO_ROOT / "runs" / run / f"extras_val_{docs}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"nbytes": np.asarray(r["bytes_per_doc"], dtype=float), "v": {k: np.asarray(d["nats"]) for k, d in r["variants"].items()}}


def table(resamples: int, seed: int) -> str:
    lines = ["Further scoring rules at one budget, entropy-patched models, validation only. Surprisal ranks patches by the realised mean negative log-probability of their bytes "
             "under the entropy model instead of its predictive entropy. KeyDiff keeps keys dissimilar from an anchor (the mean key of the last 16 patches). "
             "Expected attention scores each key by the softmax of its expected logit under a Gaussian fitted to the last 16 queries. Both use pre-rotary keys and queries, a simplification. "
             "SnapKV-style with value norm (snapv) multiplies the attention score by the L1 norm of the patch's value vector. "
             f"Paired 95% document bootstrap intervals ({resamples} resamples); negative means the variant is better than the comparison.", ""]
    for ds, (docs, budget, runs) in SETS.items():
        data = [load(r, docs) for r in runs]
        lines += [f"## {ds} (budget {budget} patches)", "", "| Variant | Comparison | " + " | ".join(f"seed {i}" for i in range(len(runs))) + " | Better in |", "|---|---|" + "---|" * (len(runs) + 1)]
        for var in VARIANTS:
            for comp in ("recency", "snap"):
                if var == comp or (comp == "snap" and var in ("entropy", "surprisal")):
                    continue
                cells, wins = [], 0
                for d in data:
                    if d is None:
                        cells.append("not run")
                        continue
                    idx = bootstrap_indices(len(d["nbytes"]), resamples, seed)
                    m, lo, hi = paired_diff_ci(d["v"][f"{var}|{budget}"], d["v"][f"{comp}|{budget}"], d["nbytes"], idx)
                    cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                    wins += hi < 0
                lines.append(f"| {var} | {comp} | " + " | ".join(cells) + f" | {wins}/{len(runs)} |")
        lines += ["", "| Policy | " + " | ".join(f"BPB seed {i}" for i in range(len(runs))) + " |", "|---|" + "---|" * len(runs)]
        for name in ["full|0", f"recency|{budget}"] + [f"{v}|{budget}" for v in VARIANTS]:
            lines.append(f"| {name} | " + " | ".join(f"{bpb(d['v'][name], d['nbytes']):.4f}" if d is not None else "not run" for d in data) + " |")
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
