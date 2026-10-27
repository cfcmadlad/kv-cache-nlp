import argparse
import json

import numpy as np

from compare import SETS, load
from data_utils import REPO_ROOT
from kvc.quant import entry_bytes
from kvc.stats import bootstrap_ci, bootstrap_indices, bpb, paired_diff_ci

SEEDS = ["entropy", "entropy_s1", "entropy_s2"]
STATIC = ["recency", "random", "entropy"]


def fmt(m, l, h):
    return f"{m:+.4f} [{l:+.4f}, {h:+.4f}]"


def diff(r, a, c, idx):
    return fmt(*paired_diff_ci(r["v"][a], r["v"][c], r["nbytes"], idx))


def online_table(resamples: int, seed: int) -> str:
    lines = ["Online H2O and SnapKV-style eviction against the offline approximation used in `attention_policies.md`. Online scores are accumulated during step-by-step "
             "patch decoding: the entry to drop is chosen before the new patch is attended to, from attention rows already seen (h2o: all rows; snap: the last 16). "
             "The offline scores come from a full-cache pass refined once under the resulting mask. `recency (online)` runs the same code path as the attention policies and must equal the masked recency run. "
             f"Validation only, 95% document bootstrap intervals ({resamples} resamples). Differences are the first minus the second (negative means the first is better).", ""]
    tally = {}
    for ds, (split, docs, budgets) in SETS.items():
        n = 500 if "tiny" in ds else 60
        n_online = 300 if "tiny" in ds else 60
        lines += [f"## {ds}", "", "| Model | Budget | recency | recency (online) | h2o offline | h2o online | snap offline | snap online | h2o on - recency | snap on - recency | snap on - snap off |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for suf in SEEDS:
            run = f"blt_{ds}_{suf}"
            on = load(run, split, n_online, "online.json")
            if on is None:
                continue
            main = load(run, split, docs)
            attn = load(run, split, docs, "policies_attn.json")
            keep = min(len(on["nbytes"]), len(main["nbytes"]))
            sub = lambda r: {"nbytes": r["nbytes"][:keep], "v": {k: v[:keep] for k, v in r["v"].items()}}
            on, main, attn = sub(on), sub(main), sub(attn)
            merged = {"nbytes": on["nbytes"], "v": {**{f"{k}": v for k, v in main["v"].items()}, **{k + "_off": v for k, v in attn["v"].items()},
                                                    **{k + "_on": v for k, v in on["v"].items()}}}
            idx = bootstrap_indices(keep, resamples, seed)
            for b in budgets:
                cells = [diff(merged, f"h2o|{b}_on", f"recency|{b}", idx), diff(merged, f"snap|{b}_on", f"recency|{b}", idx), diff(merged, f"snap|{b}_on", f"snap|{b}_off", idx)]
                for name, a, c in (("h2o online vs recency", f"h2o|{b}_on", f"recency|{b}"), ("snap online vs recency", f"snap|{b}_on", f"recency|{b}"),
                                   ("snap online vs snap offline", f"snap|{b}_on", f"snap|{b}_off"), ("recency online vs recency", f"recency|{b}_on", f"recency|{b}")):
                    m, lo, hi = paired_diff_ci(merged["v"][a], merged["v"][c], merged["nbytes"], idx)
                    t = tally.setdefault(name, [0, 0, 0])
                    t[0 if hi < 0 else 2 if lo > 0 else 1] += 1
                vals = [bpb(merged["v"][k], merged["nbytes"]) for k in (f"recency|{b}", f"recency|{b}_on", f"h2o|{b}_off", f"h2o|{b}_on", f"snap|{b}_off", f"snap|{b}_on")]
                lines.append(f"| {run} | {b} | " + " | ".join(f"{x:.4f}" for x in vals) + " | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Tally over all models and budgets", "", "| Comparison | first better (interval below zero) | tie | first worse |", "|---|---|---|---|"]
    lines += [f"| {k} | {v[0]} | {v[1]} | {v[2]} |" for k, v in tally.items()]
    return "\n".join(lines)


def hybrid_table(resamples: int, seed: int) -> str:
    lines = ["Recency-protected entropy eviction. The budget is split: the newest budget/2 patches are protected (a recency window), and the other half of the slots go to the patches with the highest mean entropy "
             "(`hybrid entropy`) or to random patches (`hybrid random`). Plain recency and plain high-entropy eviction (8 protected patches) are shown for reference. "
             f"Validation only, 95% document bootstrap intervals ({resamples} resamples). Differences are the first minus the second (negative means the first is better).", ""]
    for ds, (split, docs, budgets) in SETS.items():
        lines += [f"## {ds}", "", "| Model | Budget | recency | hybrid random | hybrid entropy | entropy | hybrid entropy - recency | hybrid entropy - hybrid random | hybrid random - recency |", "|---|---|---|---|---|---|---|---|---|"]
        tally = {}
        for suf in SEEDS:
            run = f"blt_{ds}_{suf}"
            main = load(run, split, docs)
            for b in budgets:
                h = load(run, split, docs, f"hybrid_{b}.json")
                if h is None or main is None:
                    continue
                r = {"nbytes": main["nbytes"], "v": {"recency": main["v"][f"recency|{b}"], "entropy": main["v"][f"entropy|{b}"], "hrandom": h["v"][f"random|{b}"], "hentropy": h["v"][f"entropy|{b}"]}}
                idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
                d = [paired_diff_ci(r["v"]["hentropy"], r["v"]["recency"], r["nbytes"], idx), paired_diff_ci(r["v"]["hentropy"], r["v"]["hrandom"], r["nbytes"], idx),
                     paired_diff_ci(r["v"]["hrandom"], r["v"]["recency"], r["nbytes"], idx)]
                for i, (m, lo, hi) in enumerate(d):
                    tally.setdefault((b, i), []).append("worse" if lo > 0 else "better" if hi < 0 else "tie")
                lines.append(f"| {run} | {b} | " + " | ".join(f"{bpb(r['v'][k], r['nbytes']):.4f}" for k in ("recency", "hrandom", "hentropy", "entropy")) + " | " + " | ".join(fmt(*x) for x in d) + " |")
        lines += ["", "Models (worse / tie / better, interval excluding zero) per comparison:", "", "| Budget | hybrid entropy vs recency | hybrid entropy vs hybrid random | hybrid random vs recency |", "|---|---|---|---|"]
        for b in budgets:
            cells = []
            for i in range(3):
                t = tally.get((b, i), [])
                cells.append(f"{t.count('worse')} / {t.count('tie')} / {t.count('better')}" if t else "n/a")
            lines.append(f"| {b} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def combo_table(resamples: int, seed: int) -> str:
    lines = ["Eviction and 4-bit quantization together (uniform 4-bit K and V, per-entry scale in fp16, all kept entries quantized). Cache bytes are computed from entry counts and bit widths. "
             "BPB is the mean over the three seeds of entropy patching; the interval is the seed-0 model's 95% document bootstrap "
             f"({resamples} resamples). Validation only. Cost is BPB minus the full fp16 cache of the same model.", ""]
    for ds, (split, docs, budgets) in SETS.items():
        rs = []
        for suf in SEEDS:
            path = REPO_ROOT / "runs" / f"blt_{ds}_{suf}" / "combo.json"
            if path.exists():
                rs.append(json.loads(path.read_text(encoding="utf-8")))
        if not rs:
            continue
        nb = np.asarray(rs[0]["bytes_per_doc"], dtype=float)
        idx = bootstrap_indices(len(nb), resamples, seed)
        keys = list(rs[0]["variants"])
        full = np.mean([bpb(np.asarray(r["variants"]["full|0|16"]["nats"]), np.asarray(r["bytes_per_doc"], dtype=float)) for r in rs])
        full_kb = np.mean([r["variants"]["full|0|16"]["mean_cache_bytes"] for r in rs]) / 1000
        lines += [f"## {ds} ({len(rs)} seeds, mean full-cache BPB {full:.4f}, mean full cache {full_kb:,.0f} KB)", "",
                  "| Setting | Mean cache (KB) | Share of full | BPB (mean of seeds) | Cost over full | Seed 0 BPB [95% CI] |", "|---|---|---|---|---|---|"]
        rows = []
        for k in keys:
            mode, b, bits = k.split("|")
            kb = np.mean([r["variants"][k]["mean_cache_bytes"] for r in rs]) / 1000
            val = np.mean([bpb(np.asarray(r["variants"][k]["nats"]), np.asarray(r["bytes_per_doc"], dtype=float)) for r in rs])
            lo, hi = bootstrap_ci(np.asarray(rs[0]["variants"][k]["nats"]), nb, idx)
            label = "full cache fp16" if mode == "full" else f"{mode} {b} patches, {'fp16' if bits == '16' else bits + '-bit'}"
            rows.append((kb, label, val, lo, hi))
        for kb, label, val, lo, hi in sorted(rows):
            front = not any(o[0] <= kb and o[2] < val for o in rows)
            lines.append(f"| {label}{' *' if front else ''} | {kb:,.0f} | {kb / full_kb:.1%} | {val:.4f} | {val - full:+.4f} | {lo:.4f} to {hi:.4f} |")
        lines += ["", "A row is marked * when no other row has equal or fewer bytes and a strictly lower BPB.", ""]
    return "\n".join(lines)


def stream_table(resamples: int, seed: int) -> str:
    lines = ["Bounded-memory streaming over long WikiText-103 documents (validation documents longer than 8,192 bytes, code-mix models trained on 4,096-byte chunks). "
             "`chunked 4k` resets the context every 4,096 bytes and keeps the full cache inside a chunk (up to about 900 patches). Streaming runs 16,384-byte chunks with a fixed patch budget, so patch positions "
             "reach about 3,600, four times beyond training; attention policies are scored online. "
             f"95% document bootstrap intervals ({resamples} resamples). Differences are the first minus the second (negative means the first is better).", ""]
    for suf in SEEDS:
        run = f"blt_wikitext103mix_{suf}"
        s4, s16 = load(run, "val", 0, "stream4k.json"), load(run, "val", 0, "stream16k.json")
        if s4 is None or s16 is None:
            continue
        idx = bootstrap_indices(len(s16["nbytes"]), resamples, seed)
        lines += [f"## {run} ({len(s16['nbytes'])} documents)", "", f"Chunked 4k, full cache: {bpb(s4['v']['full|0'], s4['nbytes']):.4f}; chunked 16k stream, full cache: {bpb(s16['v']['full|0'], s16['nbytes']):.4f}.", "",
                  "| Budget | Policy | Streaming BPB | Cost over chunked 4k full | Policy - recency |", "|---|---|---|---|---|"]
        base = s4["v"]["full|0"]
        for b in (64, 128, 256):
            for p in ("recency", "random", "entropy", "h2o", "snap"):
                if f"{p}|{b}" not in s16["v"]:
                    continue
                m, lo, hi = paired_diff_ci(s16["v"][f"{p}|{b}"], s16["v"][f"recency|{b}"], s16["nbytes"], idx)
                cost = bpb(s16["v"][f"{p}|{b}"], s16["nbytes"]) - bpb(base, s4["nbytes"])
                lines.append(f"| {b} | {p} | {bpb(s16['v'][f'{p}|{b}'], s16['nbytes']):.4f} | {cost:+.4f} | {fmt(m, lo, hi) if p != 'recency' else ''} |")
        lines.append("")
    return "\n".join(lines)


def needle_table(resamples: int, seed: int) -> str:
    lines = ["Planted rare-name recall. A random four-syllable name is introduced in one sentence near the start of a context, then repeated after the given number of filler bytes "
             "(\"... Then everybody said hello to \" followed by the name); the score is the BPB of the name after its first two letters, so a model that remembers the name can lower it. "
             "`ctrl` replaces the first mention with a different name, so it measures what the model does without the cue. `kept` is the share of the planted name's patches still in the cache when the repeat starts. "
             "Higher accuracy and lower bits per byte mean better recall. Trials are independent random names on validation text.", ""]
    for run in sorted(p.parent.name for p in (REPO_ROOT / "runs").glob("*/needle.json")):
        r = json.loads((REPO_ROOT / "runs" / run / "needle.json").read_text(encoding="utf-8"))
        lines += [f"## {run} ({r['trials']} trials per distance)", ""]
        for dist, rows in r["cells"].items():
            ctrl = np.mean(rows["ctrl|0"], 0)
            lines += [f"Distance {dist} filler bytes. Full cache {np.mean(rows['full|0'], 0)[0]:.3f} bits per byte, accuracy {np.mean(rows['full|0'], 0)[1]:.3f}; control {ctrl[0]:.3f} bits, accuracy {ctrl[1]:.3f}.", "",
                      "| Policy | Budget | Bits per byte | Accuracy | Name patches kept |", "|---|---|---|---|---|"]
            for k, v in rows.items():
                if k in ("full|0", "ctrl|0"):
                    continue
                mode, b = k.split("|")
                a = np.mean(v, 0)
                lines.append(f"| {mode} | {b} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f} |")
            lines.append("")
    return "\n".join(lines)


def bench_table(resamples: int, seed: int) -> str:
    path = REPO_ROOT / "runs" / "blt_wikitext103mix_entropy" / "bench.json"
    if not path.exists():
        return "not run"
    r = json.loads(path.read_text(encoding="utf-8"))
    lines = [f"Measured step-by-step decoding of one stream on {r['gpu']} (batch 1, fp32 weights, Python loop, one byte per step). Cache bytes are counted from the tensors held "
             "after the given number of bytes, expressed at 2 bytes per element (fp16). `Global` is the patch cache, `local` is the fixed 3-layer byte window cache. Times are cumulative wall time. "
             "Eviction here uses real removal from the cache tensors (recency, first patch and 8 newest protected).", "",
             "| Policy | Bytes decoded | Patches | Global cache (KB) | Local cache (KB) | Seconds | ms per byte since previous row |", "|---|---|---|---|---|---|---|"]
    for row in r["rows"]:
        lines.append(f"| {row['policy']} | {row['bytes']:,} | {row['patches']:,} | {row['global_bytes_fp16'] / 1000:,.0f} | {row['local_bytes_fp16'] / 1000:,.0f} | {row['seconds_total']:.1f} | {row['ms_per_byte_since_last']:.2f} |")
    return "\n".join(lines)


def scale_table(resamples: int, seed: int) -> str:
    lines = ["Ten times more training (30,000 steps, about one billion bytes) against the main models (3,000 steps), seed 0, same data, same patching and same evaluation documents. "
             f"Validation only, 95% document bootstrap intervals ({resamples} resamples). Differences are the first minus the second (negative means the first is better).", "",
             "| Model | Steps | Full BPB [95% CI] | Budget | recency | random | entropy | snap | entropy - random | entropy - recency | snap - recency |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    runs = [("tinystories", "blt_tinystories_entropy", "3,000"), ("tinystories", "blt_tinystories_entropy_x10", "30,000"),
            ("tinystories", "blt_tinystories_fixed", "3,000"), ("tinystories", "blt_tinystories_fixed_x10", "30,000")]
    for suf in ("", "_s1", "_s2"):
        runs += [("wikitext103mix", f"blt_wikitext103mix_entropy{suf}", "3,000"), ("wikitext103mix", f"blt_wikitext103mix_entropy_x10{suf}", "30,000")]
    runs += [("wikitext103mix", "blt_wikitext103mix_fixed", "3,000"), ("wikitext103mix", "blt_wikitext103mix_fixed_x10", "30,000")]
    for ds, run, steps in runs:
        split, docs, budgets = SETS[ds]
        r = load(run, split, docs)
        a = load(run, split, docs, "policies_attn.json")
        if r is None:
            continue
        if a is not None:
            r["v"].update({k: v for k, v in a["v"].items() if k != "full|0"})
        idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
        lo, hi = bootstrap_ci(r["v"]["full|0"], r["nbytes"], idx)
        for b in budgets:
            cells = [diff(r, f"entropy|{b}", f"random|{b}", idx), diff(r, f"entropy|{b}", f"recency|{b}", idx), diff(r, f"snap|{b}", f"recency|{b}", idx) if f"snap|{b}" in r["v"] else "n/a"]
            snap = f"{bpb(r['v'][f'snap|{b}'], r['nbytes']):.4f}" if f"snap|{b}" in r["v"] else "n/a"
            lines.append(f"| {run} | {steps} | {bpb(r['v']['full|0'], r['nbytes']):.4f} [{lo:.4f}, {hi:.4f}] | {b} | "
                         + " | ".join(f"{bpb(r['v'][f'{p}|{b}'], r['nbytes']):.4f}" for p in ("recency", "random", "entropy")) + f" | {snap} | " + " | ".join(cells) + " |")
    lines += ["", "## 16,384-byte models: tuning on validation", "", "| Run | Batch | Learning rate | Validation BPB (60 documents) |", "|---|---|---|---|"]
    for run in ("blt_wikitext103mix_L16k_entropy", "blt_wikitext103mix_L16k_entropy_lr3", "blt_wikitext103mix_L16k_entropy_b8", "blt_wikitext103mix_L16k_entropy_b8lr3", "blt_wikitext103mix_L16k_fixed", "blt_wikitext103mix_L16k_fixed_b8"):
        path = REPO_ROOT / "runs" / run / "eval_val_60.json"
        cfg = REPO_ROOT / "configs" / f"{run}.json"
        if path.exists():
            c = json.loads(cfg.read_text(encoding="utf-8"))
            lines.append(f"| {run} | {c['batch_size']} | {c['lr']} | {json.loads(path.read_text(encoding='utf-8'))['bpb']} |")
    return "\n".join(lines)


def ablation_seed_table(resamples: int, seed: int) -> str:
    from compare import ABLATIONS
    lines = ["Ablations repeated on seeds 1 and 2 (the main ablation table uses seed 0). Each cell is keeping high entropy minus random (positive means worse), mean and 95% document bootstrap interval "
             f"({resamples} resamples), at the larger budget of each dataset. Validation only.", "",
             "| Dataset | Setting | seed 0 | seed 1 | seed 2 | Seeds with the interval above zero |", "|---|---|---|---|---|---|"]
    for ds in ("tinystories", "wikitext103"):
        split, docs, budgets = SETS[ds]
        b = budgets[-1]
        for label, name in ABLATIONS:
            cells, worse = [], 0
            for suf in SEEDS:
                r = load(f"blt_{ds}_{suf}", split, docs, name)
                if r is None:
                    cells.append("not run")
                    continue
                idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
                m, lo, hi = paired_diff_ci(r["v"][f"entropy|{b}"], r["v"][f"random|{b}"], r["nbytes"], idx)
                cells.append(fmt(m, lo, hi))
                worse += lo > 0
            lines.append(f"| {ds} (budget {b}) | {label} | " + " | ".join(cells) + f" | {worse}/3 |")
    return "\n".join(lines)


def pg19_table(resamples: int, seed: int) -> str:
    lines = ["Long books (PG-19 test split, `emozilla/pg19` mirror; the first 100,000 bytes of each of the first 20 books) decoded one byte at a time with a real fixed-size cache; patch positions grow to about 22,000, "
             "two orders of magnitude beyond the 4,096-byte training chunks. `reset4096` resets the context every 4,096 bytes with a full cache inside each chunk; `full` keeps every patch for the whole book. "
             f"Models are WikiText code-mix models, so PG-19 (pre-1919 English) is out of domain. 95% book bootstrap intervals ({resamples} resamples). Differences are the first minus the second (negative means the first is better).", ""]
    for run in ("blt_wikitext103mix_entropy_x10", "blt_wikitext103mix_entropy"):
        parts = sorted((REPO_ROOT / "runs" / run).glob("stream_pg19_*.json"), key=lambda p: int(p.stem.split("_")[2]))
        if not parts:
            continue
        rs = [json.loads(p.read_text(encoding="utf-8")) for p in parts]
        nbytes = np.concatenate([np.asarray(r["bytes_per_doc"], dtype=float) for r in rs])
        v = {k: np.concatenate([np.asarray(r["variants"][k]["nats"]) for r in rs]) for k in rs[0]["variants"]}
        idx = bootstrap_indices(len(nbytes), resamples, seed)
        base = next(k for k in v if k.startswith("reset"))
        lo, hi = bootstrap_ci(v["full|0"], nbytes, idx)
        lines += [f"## {run} ({len(nbytes)} books, {int(nbytes.sum()):,} bytes)", "", f"{base[:-2]} full cache within chunks: {bpb(v[base], nbytes):.4f}; whole-book full cache: {bpb(v['full|0'], nbytes):.4f} [{lo:.4f}, {hi:.4f}].", "",
                  "| Budget | Policy | BPB | Policy - reset | Policy - recency |", "|---|---|---|---|---|"]
        for k in v:
            if k in ("full|0", base):
                continue
            mode, b = k.split("|")
            d = fmt(*paired_diff_ci(v[k], v[base], nbytes, idx))
            r = "" if mode == "recency" else fmt(*paired_diff_ci(v[k], v[f"recency|{b}"], nbytes, idx))
            lines.append(f"| {b} | {mode} | {bpb(v[k], nbytes):.4f} | {d} | {r} |")
        lines.append("")
    return "\n".join(lines)


def memory_table(resamples: int, seed: int) -> str:
    per = lambda bits, layers=6, dim=384, heads=6: entry_bytes(layers, dim, heads, bits)
    bytes_per_patch, bytes_per_token = 4.5, 4.0
    local = 3 * 256 * 2 * 256 * 2
    lines = ["Cache memory per sequence at long contexts, computed from entry counts (6 layers, width 384, 6 heads in every model, so only the number of positions differs). "
             f"The BPE Transformer is the Part 2 model at {bytes_per_token:.1f} bytes per token (measured on validation text: 4.1 for TinyStories, 3.9 for WikiText-103); a byte-level Transformer "
             f"keeps one entry per byte; the byte-level BLT here makes one global entry per {bytes_per_patch} bytes. Fixed costs not shown: the byte window caches of the BLT "
             f"({local / 1000:,.0f} KB at 256 bytes) and the entropy model's own window cache (about 2,100 KB), plus weights (13.3M and 3.3M parameters). Sizes in MB.", "",
             "| Context (bytes) | Byte-level Transformer, full | BPE Transformer, full | BLT patches, full fp16 | BLT, 256 patches fp16 | BLT, 256 patches 4-bit | BLT, 64 patches 4-bit |", "|---|---|---|---|---|---|---|"]
    mb = lambda v: f"{v / 1e6:,.1f}"
    for L in (4096, 16384, 65536, 262144, 1048576):
        p = L / bytes_per_patch
        lines.append(f"| {L:,} | {mb(L * per(16))} | {mb(L / bytes_per_token * per(16))} | {mb(p * per(16))} | {mb(min(p, 256) * per(16))} | {mb(min(p, 256) * per(4))} | {mb(min(p, 64) * per(4))} |")
    return "\n".join(lines)


TABLES = {"online": online_table, "hybrid": hybrid_table, "combo": combo_table, "stream": stream_table, "needle": needle_table, "bench": bench_table,
          "scale": scale_table, "pg19": pg19_table, "ablseeds": ablation_seed_table, "memory": memory_table}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("table", choices=list(TABLES))
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    text = TABLES[args.table](args.resamples, args.seed)
    print(text)
    if args.out:
        (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
