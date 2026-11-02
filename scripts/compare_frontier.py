import argparse
import json

import numpy as np

from data_utils import REPO_ROOT

RUNS = {"tinystories": "blt_tinystories_entropy", "wikitext103": "blt_wikitext103_entropy"}
TARGETS = [0.03, 0.05, 0.08, 0.12, 0.2, 0.3]


def load(run: str):
    r = json.loads((REPO_ROOT / "runs" / run / "frontier.json").read_text(encoding="utf-8"))
    nb = np.asarray(r["bytes_per_doc"], dtype=float)
    pts = {}
    for k, d in r["variants"].items():
        mode, budget, bits = k.split("|")
        pts[(mode, int(budget), int(bits))] = (float(np.sum(d["nats"]) / np.log(2) / nb.sum()), d["mean_cache_bytes"])
    return pts


def table() -> str:
    lines = ["Tokens against precision at a fixed number of cache bytes: eviction (recency or the SnapKV-style rule) combined with uniform K and V quantization, "
             "entry counts 16 to 256 and 16, 8, 4, 3 and 2 bits, one entropy-patched model (seed 0), validation only. Cost is bits per byte (BPB); bytes are the mean resident "
             "global cache, as a share of the full fp16 cache. For each byte share, the best configuration at or below it, and the best configuration at each width.", ""]
    for ds, run in RUNS.items():
        pts = load(run)
        full_bpb, full_bytes = pts[("full", 0, 16)]
        lines += [f"## {ds} (full fp16 cache {full_bytes / 1000:.0f} KB, {full_bpb:.4f} BPB)", "", "| Byte share | Best configuration | BPB | Excess over full | Best at 16 bit | Best at 8 bit | Best at 4 bit | Best at 3 bit | Best at 2 bit |", "|---|---|---|---|---|---|---|---|---|"]
        for t in TARGETS:
            cells, best = [], None
            for bits in (16, 8, 4, 3, 2):
                c = [(b, k) for k, (b, by) in pts.items() if k[0] != "full" and k[2] == bits and by <= t * full_bytes]
                cells.append(f"{min(c)[0]:.4f} ({min(c)[1][0]}, {min(c)[1][1]})" if c else "n/a")
                if c and (best is None or min(c)[0] < best[0]):
                    best = (min(c)[0], min(c)[1])
            lines.append(f"| {t:.0%} | {best[1][0]}, {best[1][1]} entries, {best[1][2]}-bit | {best[0]:.4f} | {best[0] - full_bpb:+.4f} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Cells give BPB with the policy and entry count in brackets.")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    args = p.parse_args()
    text = table()
    if args.out:
        (REPO_ROOT / args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
