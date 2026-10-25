import argparse
import json

import numpy as np

from data_utils import REPO_ROOT
from kvc.stats import bootstrap_ci, bootstrap_indices, bpb, paired_diff_ci

SETS = {"tinystories": ("val", 500, [32, 64]), "wikitext103": ("val", 60, [64, 128]), "tinystoriesmix": ("val", 500, [32, 64]), "wikitext103mix": ("val", 60, [64, 128])}
GROUPS = {"entropy patching, seeds 0 to 2": ["blt_{}_entropy", "blt_{}_entropy_s1", "blt_{}_entropy_s2"],
          "fixed stride 4, seeds 0 to 2": ["blt_{}_fixed", "blt_{}_fixed_s1", "blt_{}_fixed_s2"],
          "fixed stride 5, seeds 0 to 2": ["blt_{}_fixed5", "blt_{}_fixed5_s1", "blt_{}_fixed5_s2"]}
PAIRS = [("entropy", "random"), ("entropy", "recency"), ("entropy", "entropy_inv")]


def ran(ds: str) -> bool:
    return (REPO_ROOT / "runs" / f"blt_{ds}_entropy" / "quant_hi4_lo2.json").exists()


def load(run: str, split: str, docs: int, name: str = None) -> dict:
    path = REPO_ROOT / "runs" / run / (name or f"policies_{split}_{docs}.json")
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"nbytes": np.asarray(r["bytes_per_doc"], dtype=float),
            "v": {k: np.asarray(d["nats"]) for k, d in r["variants"].items()}}


def seeds_table(resamples: int, seed: int) -> str:
    lines = ["Paired BPB difference of keeping high-entropy patches minus the comparison policy (positive means keeping high entropy is worse), "
             f"mean and 95% document bootstrap interval ({resamples} resamples), one row per trained model. Validation only.", ""]
    for ds, (split, docs, budgets) in SETS.items():
        if not ran(ds):
            continue
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


ABLATIONS = [("entropy model, mean (main setting)", None), ("model's own entropy, mean", "ablate_own.json"), ("entropy model, max", "ablate_agg_max.json"),
             ("entropy model, first byte", "ablate_agg_first.json"), ("entropy model, sum", "ablate_agg_sum.json"),
             ("sink 0", "ablate_sink0.json"), ("sink 4", "ablate_sink4.json")]
ALL = ["recency", "random", "entropy", "entropy_inv"]


def ablation_table(resamples: int, seed: int) -> str:
    lines = [f"Seed-0 entropy-patched model, validation only, 95% document bootstrap intervals ({resamples} resamples). Differences are keeping high entropy minus the other policy (positive means worse).", ""]
    for ds, (split, docs, budgets) in SETS.items():
        if not ran(ds):
            continue
        lines += [f"## {ds}: scoring rule, signal and sink size", "", "| Setting | Budget | recency | random | entropy | entropy_inv | entropy - random | entropy - recency |", "|---|---|---|---|---|---|---|---|"]
        for label, name in ABLATIONS:
            r = load(f"blt_{ds}_entropy", split, docs, name)
            if r is None:
                lines.append(f"| {label} | not run | | | | | | |")
                continue
            idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
            for b in budgets:
                cells = []
                for c in ("random", "recency"):
                    m, l, h = paired_diff_ci(r["v"][f"entropy|{b}"], r["v"][f"{c}|{b}"], r["nbytes"], idx)
                    cells.append(f"{m:+.4f} [{l:+.4f}, {h:+.4f}]")
                lines.append(f"| {label} | {b} | " + " | ".join(f"{bpb(r['v'][f'{m}|{b}'], r['nbytes']):.4f}" for m in ALL) + " | " + " | ".join(cells) + " |")
        lines.append("")
    r = load("blt_tinystories_fixed1", "val", 500)
    patch = load("blt_tinystories_fixed", "val", 500)
    lines += ["## tinystories: byte-level global cache (stride 1) against patch cache (stride 4), same bytes per entry", ""]
    if r is None:
        return "\n".join(lines + ["not run"])
    idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
    full = bpb(r["v"]["full|0"], r["nbytes"])
    lines += [f"Byte-level model full-cache BPB {full:.4f}; patch model (stride 4) {bpb(patch['v']['full|0'], patch['nbytes']):.4f}. Budget is the number of global cache entries; every entry costs the same bytes, so equal budgets are equal bytes.", "",
              "| Model | Budget (entries) | Bytes of text spanned | recency | random | entropy | entropy_inv | cost of recency over full | entropy - random |", "|---|---|---|---|---|---|---|---|---|"]
    for name, res, span in (("byte-level (stride 1)", r, 1), ("patch (stride 4)", patch, 4)):
        ridx = bootstrap_indices(len(res["nbytes"]), resamples, seed)
        for b in (32, 64, 144, 288):
            if f"entropy|{b}" not in res["v"]:
                continue
            m, l, h = paired_diff_ci(res["v"][f"entropy|{b}"], res["v"][f"random|{b}"], res["nbytes"], ridx)
            full_b = bpb(res["v"]["full|0"], res["nbytes"])
            lines.append(f"| {name} | {b} | {b * span} | " + " | ".join(f"{bpb(res['v'][f'{p}|{b}'], res['nbytes']):.4f}" for p in ALL)
                         + f" | {bpb(res['v'][f'recency|{b}'], res['nbytes']) - full_b:+.4f} | {m:+.4f} [{l:+.4f}, {h:+.4f}] |")
    return "\n".join(lines)


QUANT_LABEL = {"uniform": "uniform {} bits", "recency": "tiers 4/2 bit, recent {}", "random": "tiers 4/2 bit, random {}",
               "entropy": "tiers 4/2 bit, keep high entropy {}", "entropy_inv": "tiers 4/2 bit, keep low entropy {}"}


def load_quant(run: str, name: str, docs: int) -> dict:
    path = REPO_ROOT / "runs" / run / name
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["docs"] == docs
    return {k: (np.asarray(d["nats"]), d["mean_cache_bytes"]) for k, d in r["variants"].items()}, np.asarray(r["bytes_per_doc"], dtype=float)


def comparison_table(resamples: int, seed: int) -> str:
    lines = [f"Seed-0 entropy-patched BLT, validation only, 95% document bootstrap intervals ({resamples} resamples). Cache is the mean resident global KV size over positions of a 4,096-byte chunk, "
             "computed from entry counts and bit widths, not measured. Eviction rows keep the first patch and the 8 newest. A row is marked * when no other row has equal or fewer bytes and a strictly lower BPB.", ""]
    for ds, (split, docs, budgets) in SETS.items():
        if not ran(ds):
            continue
        r = load(f"blt_{ds}_entropy", split, docs)
        q, qbytes = load_quant(f"blt_{ds}_entropy", "quant_hi4_lo2.json", docs)
        idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
        rows = []
        for k, v in r["v"].items():
            mode, b = k.split("|")
            if k != "full|0" and mode not in ALL:
                continue
            kb = json.loads((REPO_ROOT / "runs" / f"blt_{ds}_entropy" / f"policies_{split}_{docs}.json").read_text(encoding="utf-8"))["variants"][k]["mean_cache_bytes"] / 1000
            rows.append(("full cache fp16" if k == "full|0" else f"evict to {b} patches, {mode}", kb, v, r["nbytes"], idx))
        qidx = bootstrap_indices(len(qbytes), resamples, seed)
        for k, (nats, kbytes) in q.items():
            mode, param = k.split("|")
            if mode == "full" or (mode == "uniform" and param == "16"):
                continue
            rows.append((QUANT_LABEL[mode].format(param if mode == "uniform" else f"{float(param):g} high"), kbytes / 1000, nats, qbytes, qidx))
        rows.sort(key=lambda x: x[1])
        full_nats = r["v"]["full|0"]
        full_kb = [x for x in rows if x[0] == "full cache fp16"][0][1]
        scored = []
        for name, kb, nats, nbytes, ix in rows:
            lo, hi = bootstrap_ci(nats, nbytes, ix)
            scored.append((name, kb, bpb(nats, nbytes), lo, hi))
        lines += [f"## {ds} (full-cache BPB {bpb(full_nats, r['nbytes']):.4f})", "", "| Setting | Mean KV cache (KB) | Share of full | BPB [95% CI] | Cost over full |", "|---|---|---|---|---|"]
        full_bpb = bpb(full_nats, r["nbytes"])
        for name, kb, b_, lo, hi in scored:
            front = not any(o[1] <= kb and o[2] < b_ for o in scored)
            lines.append(f"| {name}{' *' if front else ''} | {kb:,.0f} | {kb / full_kb:.0%} | {b_:.4f} [{lo:.4f}, {hi:.4f}] | {b_ - full_bpb:+.4f} |")
        lines.append("")
    return "\n".join(lines)


def code_table(resamples: int, seed: int) -> str:
    lines = ["Models evaluated on the first 200 validation files of the code set (Python, Java, JavaScript; bigcode/the-stack-smol). Rows without mix were trained on prose only (out of domain, including their entropy model); wikitext103mix was trained with 15% code. "
             f"Seed-0 models, 95% document bootstrap intervals ({resamples} resamples).", "",
             "| Trained on | Full-cache BPB [95% CI] | Mean patch length | Budget | recency | random | entropy | entropy_inv | entropy - random | entropy - recency |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in SETS:
        run = f"blt_{ds}_entropy"
        if not (REPO_ROOT / "runs" / run / "policies_code_200.json").exists():
            continue
        r = load(run, "val", 200, "policies_code_200.json")
        info = json.loads((REPO_ROOT / "runs" / run / "eval_code_200.json").read_text(encoding="utf-8"))
        idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
        lo, hi = bootstrap_ci(r["v"]["full|0"], r["nbytes"], idx)
        for b in (64, 128):
            cells = []
            for c in ("random", "recency"):
                m, l, h = paired_diff_ci(r["v"][f"entropy|{b}"], r["v"][f"{c}|{b}"], r["nbytes"], idx)
                cells.append(f"{m:+.4f} [{l:+.4f}, {h:+.4f}]")
            lines.append(f"| {ds} | {bpb(r['v']['full|0'], r['nbytes']):.4f} [{lo:.4f}, {hi:.4f}] | {info['mean_patch_len']} | {b} | "
                         + " | ".join(f"{bpb(r['v'][f'{m}|{b}'], r['nbytes']):.4f}" for m in ALL) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


ATTN_RUNS = ["entropy", "entropy_s1", "entropy_s2", "fixed", "fixed5"]


def merged(run: str, split: str, docs: int):
    r = load(run, split, docs)
    a = load(run, split, docs, "policies_attn.json")
    if r is None:
        return None
    if a is None:
        return r
    r["v"].update({k: v for k, v in a["v"].items() if k != "full|0"})
    return r


def attention_table(resamples: int, seed: int) -> str:
    lines = ["Attention-score baselines against recency, random and keeping high entropy. h2o keeps the patches with the largest accumulated attention received so far "
             "(averaged over heads and layers); snap uses attention from the last 16 patches only. Scores come from a full-cache pass refined once under the resulting mask, "
             f"so they approximate online scoring. Validation only, 95% document bootstrap intervals ({resamples} resamples). Differences are the attention policy minus the comparison (negative means the attention policy is better).", ""]
    for ds, (split, docs, budgets) in SETS.items():
        lines += [f"## {ds}", "", "| Model | Budget | recency | random | entropy | h2o | snap | h2o - recency | h2o - random | snap - recency | snap - random |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for suf in ATTN_RUNS:
            run = f"blt_{ds}_{suf}"
            r = merged(run, split, docs)
            if r is None:
                continue
            idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
            for b in budgets:
                cells = []
                for a, c in (("h2o", "recency"), ("h2o", "random"), ("snap", "recency"), ("snap", "random")):
                    m, l, h = paired_diff_ci(r["v"][f"{a}|{b}"], r["v"][f"{c}|{b}"], r["nbytes"], idx)
                    cells.append(f"{m:+.4f} [{l:+.4f}, {h:+.4f}]")
                lines.append(f"| {run} | {b} | " + " | ".join(f"{bpb(r['v'][f'{p}|{b}'], r['nbytes']):.4f}" for p in ("recency", "random", "entropy", "h2o", "snap")) + " | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


LONG = [("blt_wikitext103mix_entropy", "4,096 bytes", [64, 128]), ("blt_wikitext103mix_L16k_entropy", "16,384 bytes, batch 2", [64, 128, 256, 512]),
        ("blt_wikitext103mix_L16k_entropy_b8", "16,384 bytes, batch 8", [64, 128, 256, 512]),
        ("blt_wikitext103mix_fixed", "4,096 bytes", [64, 128]), ("blt_wikitext103mix_L16k_fixed", "16,384 bytes, batch 2", [64, 128, 256, 512]),
        ("blt_wikitext103mix_L16k_fixed_b8", "16,384 bytes, batch 8", [64, 128, 256, 512])]


def long_table(resamples: int, seed: int) -> str:
    lines = ["WikiText-103 validation (60 documents) with the code-mix models trained and evaluated with 4,096-byte chunks (the main models) or 16,384-byte chunks (L16k, same bytes seen in training). "
             f"Documents have a mean length of about 19,000 bytes, so 16,384-byte chunks cover most of a document. 95% document bootstrap intervals ({resamples} resamples). Differences are the first policy minus the second.", "",
             "| Model | Chunk | Full BPB [95% CI] | Budget | recency | random | entropy | h2o | snap | entropy - random | h2o - recency | snap - recency |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, chunk, budgets in LONG:
        r = merged(run, "val", 60)
        if r is None:
            continue
        idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
        lo, hi = bootstrap_ci(r["v"]["full|0"], r["nbytes"], idx)
        for b in budgets:
            if f"entropy|{b}" not in r["v"]:
                continue
            cells = []
            for a, c in (("entropy", "random"), ("h2o", "recency"), ("snap", "recency")):
                m, l, h = paired_diff_ci(r["v"][f"{a}|{b}"], r["v"][f"{c}|{b}"], r["nbytes"], idx)
                cells.append(f"{m:+.4f} [{l:+.4f}, {h:+.4f}]")
            lines.append(f"| {run} | {chunk} | {bpb(r['v']['full|0'], r['nbytes']):.4f} [{lo:.4f}, {hi:.4f}] | {b} | "
                         + " | ".join(f"{bpb(r['v'][f'{p}|{b}'], r['nbytes']):.4f}" for p in ("recency", "random", "entropy", "h2o", "snap")) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def test_table(resamples: int, seed: int) -> str:
    lines = ["WikiText-103 test split (62 documents), evaluated once with the policies and budgets fixed on validation. Models are the three seeds of entropy patching and of fixed strides 4 and 5, "
             f"without and with the 15% code mix. 95% document bootstrap intervals ({resamples} resamples). Differences are the first policy minus the second (negative means the first is better).", "",
             "| Model | Full BPB [95% CI] | Budget | recency | random | entropy | h2o | snap | entropy - random | entropy - recency | snap - recency |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in ("wikitext103", "wikitext103mix"):
        for suf in ["entropy", "entropy_s1", "entropy_s2", "fixed", "fixed_s1", "fixed_s2", "fixed5", "fixed5_s1", "fixed5_s2"]:
            run = f"blt_{ds}_{suf}"
            r = load(run, "test", 62)
            if r is None:
                continue
            idx = bootstrap_indices(len(r["nbytes"]), resamples, seed)
            lo, hi = bootstrap_ci(r["v"]["full|0"], r["nbytes"], idx)
            for b in (64, 128):
                cells = []
                for a, c in (("entropy", "random"), ("entropy", "recency"), ("snap", "recency")):
                    m, l, h = paired_diff_ci(r["v"][f"{a}|{b}"], r["v"][f"{c}|{b}"], r["nbytes"], idx)
                    cells.append(f"{m:+.4f} [{l:+.4f}, {h:+.4f}]")
                lines.append(f"| {run} | {bpb(r['v']['full|0'], r['nbytes']):.4f} [{lo:.4f}, {hi:.4f}] | {b} | "
                             + " | ".join(f"{bpb(r['v'][f'{p}|{b}'], r['nbytes']):.4f}" for p in ("recency", "random", "entropy", "h2o", "snap")) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("table", choices=["seeds", "ablations", "table", "code", "attention", "long", "test"])
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    text = {"seeds": seeds_table, "ablations": ablation_table, "table": comparison_table, "code": code_table, "attention": attention_table, "long": long_table, "test": test_table}[args.table](args.resamples, args.seed)
    print(text)
    if args.out:
        (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
