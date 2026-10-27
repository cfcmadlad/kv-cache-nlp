import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_demo import MAX_BYTES, load_pair
from data_utils import REPO_ROOT, read_jsonl_texts
from kvc.cache import Policy, patch_entropy_scores, patch_scores, visible_mask
from kvc.patching import doc_entropy_and_nats, patch_flags
from models.entropy import BOS
from train_blt import patch_settings

POLICIES = ("recency", "random", "entropy", "entropy_inv")
CLASSES = ("space or newline", "digit", "uppercase letter", "word-initial lowercase", "word-internal lowercase", "punctuation or other")
TERCILES = ("lowest third", "middle third", "highest third")


def byte_class(doc: np.ndarray) -> np.ndarray:
    prev = np.concatenate([[32], doc[:-1]])
    prev_word = ((prev >= 48) & (prev <= 57)) | ((prev >= 65) & (prev <= 90)) | ((prev >= 97) & (prev <= 122)) | (prev >= 128)
    lower = (doc >= 97) & (doc <= 122)
    out = np.full(len(doc), 5)
    out[lower & prev_word] = 4
    out[lower & ~prev_word] = 3
    out[(doc >= 65) & (doc <= 90)] = 2
    out[(doc >= 48) & (doc <= 57)] = 1
    out[(doc == 32) | (doc == 10)] = 0
    return out


def spans(doc: np.ndarray, flags: np.ndarray) -> list:
    return [bytes(p).decode("utf-8", errors="replace").replace("\n", " ") for p in np.split(doc.astype(np.uint8), np.flatnonzero(flags)[1:])]


@torch.no_grad()
def run_doc(text: str, model, entropy, cfg: dict, budget: int, device: str, sink: int, recent: int, draws: int) -> dict:
    raw = text.encode("utf-8")[:MAX_BYTES]
    ps = patch_settings(cfg)
    h = doc_entropy_and_nats(entropy, raw, device)[0]
    n = len(raw)
    doc = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    flags = patch_flags(h, np.arange(n), np.zeros(n, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
    cs = np.cumsum(flags)
    pid = cs - flags
    P = int(cs[-1]) + 1
    score = patch_entropy_scores(pid, h.astype(np.float64), P, "mean")
    inputs = (torch.from_numpy(np.array([BOS, *doc[:-1]]))[None].to(device), torch.from_numpy(flags)[None].to(device))
    target = torch.from_numpy(doc).to(device)

    def nats(mask=None):
        extra = () if mask is None else (torch.from_numpy(mask)[None].to(device),)
        return F.cross_entropy(model((*inputs, *extra))[0][0].float(), target, reduction="none").cpu().numpy()

    out = {"doc": doc, "flags": flags, "pid": pid, "score": score, "P": P, "full": nats(), "nats": {}, "final": {}}
    for mode in POLICIES:
        pol = Policy(mode, budget, sink, recent)
        reps = draws if mode == "random" else 1
        total = np.zeros(n)
        for r in range(reps):
            mask = visible_mask(P, patch_scores(pol, P, score, np.random.default_rng([0, r])), pol)
            total += nats(mask) / reps
            if r == 0:
                out["final"][mode] = mask[-1].copy()
        out["nats"][mode] = total
    return out


def bpb(nats: np.ndarray, count: int) -> float:
    return float(np.sum(nats) / math.log(2) / max(count, 1))


def show(spans_: list, ids: list, kept: np.ndarray, limit: int) -> str:
    parts, used = [], 0
    for k, s in zip(ids, spans_):
        if kept[k]:
            parts.append(s)
            used += len(s)
        if used >= limit:
            break
    return "|".join(parts)


def context(doc: np.ndarray, i: int, width: int = 48) -> str:
    left = bytes(doc[max(0, i - width):i].astype(np.uint8)).decode("utf-8", errors="replace").replace("\n", " ")
    target = bytes([int(doc[i])]).decode("utf-8", errors="replace").replace("\n", "\\n")
    return f"{left}[{target}]"


def dataset_report(name: str, docs: list, budget: int, top: int, example_chars: int) -> list:
    all_scores = np.concatenate([d["score"][1:] for d in docs])
    cuts = np.quantile(all_scores, [1 / 3, 2 / 3])
    modes = ("full", *POLICIES)
    nats_of = lambda d, m: d["full"] if m == "full" else d["nats"][m]
    tercile = [np.digitize(d["score"][d["pid"]], cuts) for d in docs]
    total_bytes = sum(len(d["doc"]) for d in docs)
    lines = [f"## {name}", "", f"{len(docs)} validation documents, {total_bytes:,} bytes (each cut to 4,096 bytes), budget {budget} patches, protected set the first patch and the 8 newest. Random is the mean of 3 draws.", ""]
    lines += ["| Policy | BPB | Cost over full cache |", "|---|---|---|"]
    full = bpb(sum(d["full"].sum() for d in docs), total_bytes)
    for m in modes:
        v = bpb(sum(nats_of(d, m).sum() for d in docs), total_bytes)
        lines.append(f"| {m} | {v:.4f} | {v - full:+.4f} |")
    lines += ["", f"### BPB by entropy tercile of the patch holding the byte (cut points {cuts[0]:.3f} and {cuts[1]:.3f} nats)", "",
              "Cost over the full cache, in BPB, within each tercile (the full-cache BPB of the tercile is in the first column).", "",
              "| Tercile | Share of bytes | Full-cache BPB | recency | random | entropy | entropy_inv |", "|---|---|---|---|---|---|---|"]
    for t, label in enumerate(TERCILES):
        cnt = sum(int((tr == t).sum()) for tr in tercile)
        base = bpb(sum(d["full"][tr == t].sum() for d, tr in zip(docs, tercile)), cnt)
        cells = [f"{bpb(sum(d['nats'][m][tr == t].sum() for d, tr in zip(docs, tercile)), cnt) - base:+.4f}" for m in POLICIES]
        lines.append(f"| {label} | {cnt / total_bytes:.1%} | {base:.4f} | " + " | ".join(cells) + " |")
    lines += ["", "### Which patches the final cache keeps", "", "Share of evictable patches (all but the first and the 8 newest) that are still in the cache at the last patch of the document, by entropy tercile of the patch.", "",
              "| Policy | lowest third | middle third | highest third |", "|---|---|---|---|"]
    for m in POLICIES:
        cells = []
        for t in range(3):
            kept = seen = 0
            for d in docs:
                ids = np.arange(1, d["P"] - 8)
                ids = ids[np.digitize(d["score"][ids], cuts) == t]
                kept += int(d["final"][m][ids].sum())
                seen += len(ids)
            cells.append(f"{kept / max(seen, 1):.1%}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lengths = {m: [] for m in POLICIES}
    pool = []
    for d in docs:
        ids = np.arange(1, d["P"] - 8)
        size = np.bincount(d["pid"], minlength=d["P"])
        pool.append(size[ids])
        for m in POLICIES:
            lengths[m].append(size[ids[d["final"][m][ids]]])
    lines += ["", "Mean length in bytes of the evictable patches that are kept, against all evictable patches (" + f"{np.concatenate(pool).mean():.2f}" + "):", "",
              "| Policy | Mean kept patch length (bytes) | Share of kept patches that are 1 or 2 bytes long |", "|---|---|---|"]
    for m in POLICIES:
        kept_len = np.concatenate(lengths[m])
        lines.append(f"| {m} | {kept_len.mean():.2f} | {(kept_len <= 2).mean():.1%} |")
    lines += ["", f"Share of all evictable patches that are 1 or 2 bytes long: {(np.concatenate(pool) <= 2).mean():.1%}."]
    lines += ["", "### Which kinds of bytes lose most under entropy eviction", "", "Mean extra loss in bits per byte, by class of the target byte, over the full-cache loss. Positive is worse.", "",
              "| Byte class | Count | Full-cache loss (bits) | recency | entropy | entropy minus recency |", "|---|---|---|---|---|---|"]
    classes = [byte_class(d["doc"]) for d in docs]
    for c, label in enumerate(CLASSES):
        sel = [cl == c for cl in classes]
        cnt = sum(int(s.sum()) for s in sel)
        if cnt == 0:
            continue
        agg = lambda m: sum(nats_of(d, m)[s].sum() for d, s in zip(docs, sel)) / math.log(2) / cnt
        lines.append(f"| {label} | {cnt:,} | {agg('full'):.3f} | {agg('recency') - agg('full'):+.4f} | {agg('entropy') - agg('full'):+.4f} | {agg('entropy') - agg('recency'):+.4f} |")
    delta = [(d["nats"]["entropy"] - d["nats"]["recency"]) / math.log(2) for d in docs]
    flat = np.concatenate(delta)
    lines += ["", f"Entropy loses to recency on {(flat > 0).mean():.1%} of bytes and wins on {(flat < 0).mean():.1%}; the largest 1 percent of bytes by loss increase carry {flat[flat >= np.quantile(flat, 0.99)].sum() / flat[flat > 0].sum():.1%} of the total increase.", ""]
    ranked = sorted(((v, di, i) for di, dl in enumerate(delta) for i, v in enumerate(dl)), key=lambda t: t[0])
    for title, picks in (("largest loss increase under entropy against recency", ranked[::-1][:top]), ("largest loss decrease under entropy against recency", ranked[:top])):
        lines += [f"### Bytes with the {title}", "", "| Document | Position | Context and target byte (in brackets) | Full cache (bits) | recency (bits) | entropy (bits) | Patch holding the byte, entropy score (nats) |", "|---|---|---|---|---|---|---|"]
        for _, di, i in picks:
            d = docs[di]
            lines.append(f"| {di} | {i} | `{context(d['doc'], i)}` | {d['full'][i] / math.log(2):.2f} | {d['nats']['recency'][i] / math.log(2):.2f} | {d['nats']['entropy'][i] / math.log(2):.2f} | {d['score'][d['pid'][i]]:.2f} |")
        lines.append("")
    return lines


def example(name: str, d: dict, di: int, budget: int, limit: int) -> list:
    sp = spans(d["doc"], d["flags"])
    ids = list(range(1, len(sp) + 1))
    n = len(d["doc"])
    lines = [f"### {name}: document {di}", "", f"{n} bytes, {d['P'] - 1} patches plus the start patch, {n / (d['P'] - 1):.1f} bytes per patch. Full-cache BPB {bpb(d['full'], n):.3f}.", "",
             f"Text with patch boundaries (`|`), first {limit} bytes:", "", "```", "|".join(sp)[:limit], "```", "", f"Patches in the cache at the last patch of the document, budget {budget} (first {limit // 2} bytes of the kept text, patches in order, `|` between them):", ""]
    for m in POLICIES:
        kept = d["final"][m]
        lines += [f"- {m}, BPB {bpb(d['nats'][m], n):.3f} ({bpb(d['nats'][m], n) - bpb(d['full'], n):+.3f} over full): `{show(sp, ids, kept, limit // 2)}`"]
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/part3/error_analysis.md")
    ap.add_argument("--tinystories-docs", type=int, default=100)
    ap.add_argument("--wikitext-docs", type=int, default=30)
    ap.add_argument("--min-bytes", type=int, default=600)
    ap.add_argument("--budgets", type=int, nargs=2, default=[32, 64])
    ap.add_argument("--sink", type=int, default=1)
    ap.add_argument("--recent", type=int, default=8)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--example-chars", type=int, default=420)
    ap.add_argument("--example-docs", type=int, nargs=2, default=[0, 0])
    args = ap.parse_args()
    device = "cpu"
    lines = ["Qualitative and quantitative error analysis of cache eviction, seed-0 entropy-patched BLT models from the `v0.3-part3` release, validation documents, CPU. Regenerate with `python scripts/error_analysis_part3.py`.", ""]
    for (name, count), budget, ex in zip((("tinystories", args.tinystories_docs), ("wikitext103", args.wikitext_docs)), args.budgets, args.example_docs):
        model, entropy, cfg = load_pair(name, device)
        texts = [t for t in read_jsonl_texts(REPO_ROOT / "data" / "processed" / name / "val.jsonl", 500) if len(t.encode("utf-8")) >= args.min_bytes][:count]
        docs = [run_doc(t, model, entropy, cfg, budget, device, args.sink, args.recent, args.draws) for t in texts]
        print(f"{name}: {len(docs)} documents", flush=True)
        lines += dataset_report(name, docs, budget, args.top, args.example_chars)
        lines += example(name, docs[ex], ex, budget, args.example_chars)
    out = REPO_ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
