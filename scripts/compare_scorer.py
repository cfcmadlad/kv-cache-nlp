import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_indices, bpb, paired_diff_ci

SETS = {"tinystories": (500, 32, ["blt_tinystories_entropy", "blt_tinystories_entropy_s1", "blt_tinystories_entropy_s2"]),
        "wikitext103": (60, 64, ["blt_wikitext103_entropy", "blt_wikitext103_entropy_s1", "blt_wikitext103_entropy_s2"])}
PAIRS = [("scorer_all", "recency"), ("scorer_all", "snap"), ("scorer_all", "scorer_noent"), ("scorer_noent", "recency"), ("snap", "recency")]


def load(run: str, docs: int):
    path = REPO_ROOT / "runs" / run / f"scorer_val_{docs}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"nbytes": np.asarray(r["bytes_per_doc"], dtype=float), "v": {k: np.asarray(d["nats"]) for k, d in r["variants"].items()}}


def table(resamples: int, seed: int) -> str:
    lines = ["Learned, attention-free eviction scorer. A small network predicts, from cheap features, the attention a cached patch will receive from the next 16 patches "
             "(the quantity the SnapKV-style rule measures). Features: age, patch length, entropy statistics (dropped in scorer_noent), key and value norms. "
             "Trained per model on training documents, evaluated on validation documents. Negative differences mean the first policy is better. "
             f"Paired 95% document bootstrap intervals ({resamples} resamples), one column per seed.", ""]
    for ds, (docs, budget, runs) in SETS.items():
        data = [load(r, docs) for r in runs]
        lines += [f"## {ds} (budget {budget} patches)", "", "| Comparison | " + " | ".join(f"seed {i}" for i in range(len(runs))) + " |", "|---|" + "---|" * len(runs)]
        for a, b in PAIRS:
            cells = []
            for d in data:
                if d is None:
                    cells.append("not run")
                    continue
                idx = bootstrap_indices(len(d["nbytes"]), resamples, seed)
                m, lo, hi = paired_diff_ci(d["v"][f"{a}|{budget}"], d["v"][f"{b}|{budget}"], d["nbytes"], idx)
                cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            lines.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
        lines += ["", "| Policy | " + " | ".join(f"BPB seed {i}" for i in range(len(runs))) + " |", "|---|" + "---|" * len(runs)]
        for name in ["full|0"] + [f"{n}|{budget}" for n in ("recency", "entropy", "snap", "scorer_all", "scorer_noent")]:
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
