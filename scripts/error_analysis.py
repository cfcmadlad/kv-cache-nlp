"""Quantitative + qualitative error analysis comparing both baselines on the same val docs.

Scores every val document with both models individually (not just the aggregate BPB the
training scripts report), then reports:
  - quantitative: per-model BPB distribution stats, and how the two models' per-doc
    difficulty correlates (do they struggle on the same documents, or different ones?)
  - qualitative: the actual worst/best-scoring documents per model, and the documents
    where the two models disagree most -- concrete examples of what each model gets wrong

Usage:
    python scripts/error_analysis.py --dataset tinystories
    python scripts/error_analysis.py --dataset wikitext103 --eval-max-docs 300
"""

import argparse
import statistics
from pathlib import Path

from data_utils import load_ngram, load_transformer, read_jsonl_texts, safe_print
from train_bpe_transformer import per_document_bpb

PREVIEW_CHARS = 200


def ngram_per_doc_bpb(model, texts: list) -> list:
    scores = []
    for text in texts:
        doc = text.encode("utf-8")
        scores.append(-model.log2_prob_doc(doc) / len(doc) if doc else float("nan"))
    return scores


def distribution_stats(scores: list) -> dict:
    clean = [s for s in scores if s == s]  # drop NaN
    if not clean:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "stdev": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": len(clean),
        "mean": round(statistics.mean(clean), 4),
        "median": round(statistics.median(clean), 4),
        "stdev": round(statistics.stdev(clean), 4) if len(clean) > 1 else 0.0,
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
    }


def correlation(a: list, b: list) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x == x and y == y]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    try:
        return round(statistics.correlation(xs, ys), 4)
    except statistics.StatisticsError:
        return float("nan")  # one of the two series is constant (zero variance)


def length_bucket_stats(texts: list, scores: list, num_buckets: int = 3) -> list:
    """Average BPB grouped into terciles by document byte length -- do longer documents
    get scored differently than short ones? Always produces at most num_buckets groups
    (fewer if there aren't enough scored docs to fill them)."""
    rows = [(len(t.encode("utf-8")), s) for t, s in zip(texts, scores) if s == s]
    rows.sort(key=lambda r: r[0])
    boundaries = [round(i * len(rows) / num_buckets) for i in range(num_buckets + 1)]
    buckets = []
    for start, end in zip(boundaries, boundaries[1:]):
        chunk = rows[start:end]
        if not chunk:
            continue
        lengths = [c[0] for c in chunk]
        bpbs = [c[1] for c in chunk]
        buckets.append({
            "byte_range": f"{min(lengths)}-{max(lengths)}",
            "docs": len(chunk),
            "avg_bpb": round(statistics.mean(bpbs), 4),
        })
    return buckets


def preview(text: str) -> str:
    text = text.replace("\n", " ").strip()
    return text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")


def worst_best_examples(texts: list, scores: list, n: int) -> dict:
    """Worst n and best n scoring docs. Capped so the two lists can never overlap --
    with too few valid docs to fill both without overlap, both are shrunk accordingly."""
    indexed = [(s, i) for i, s in enumerate(scores) if s == s]
    indexed.sort()
    n = min(n, len(indexed) // 2)
    best = indexed[:n]
    worst = indexed[-n:][::-1] if n else []
    return {
        "worst": [{"bpb": round(s, 4), "text": preview(texts[i])} for s, i in worst],
        "best": [{"bpb": round(s, 4), "text": preview(texts[i])} for s, i in best],
    }


def format_report(dataset: str, ngram_scores: list, transformer_scores: list,
                   texts: list, num_examples: int) -> str:
    lines = [f"# Error analysis: {dataset}\n"]

    lines.append("## Quantitative\n")
    lines.append(f"Docs scored: {len(texts)}\n")
    lines.append("| | n-gram | transformer |")
    lines.append("|---|---|---|")
    ngram_stats = distribution_stats(ngram_scores)
    tf_stats = distribution_stats(transformer_scores)
    for key in ["mean", "median", "stdev", "min", "max"]:
        lines.append(f"| {key} | {ngram_stats[key]} | {tf_stats[key]} |")
    lines.append(f"\nPer-doc BPB correlation between models: **{correlation(ngram_scores, transformer_scores)}**")
    lines.append("(close to 1.0 = models struggle on the same docs; close to 0 = different failure modes)\n")

    lines.append("### BPB by document length (n-gram)")
    for b in length_bucket_stats(texts, ngram_scores):
        lines.append(f"- {b['byte_range']} bytes ({b['docs']} docs): avg BPB {b['avg_bpb']}")
    lines.append("\n### BPB by document length (transformer)")
    for b in length_bucket_stats(texts, transformer_scores):
        lines.append(f"- {b['byte_range']} bytes ({b['docs']} docs): avg BPB {b['avg_bpb']}")

    lines.append("\n## Qualitative\n")
    for name, scores in [("n-gram", ngram_scores), ("transformer", transformer_scores)]:
        examples = worst_best_examples(texts, scores, num_examples)
        lines.append(f"### {name} -- worst-scoring docs (highest BPB, model most \"surprised\")")
        for ex in examples["worst"]:
            lines.append(f"- **{ex['bpb']} bpb**: {ex['text']}")
        lines.append(f"\n### {name} -- best-scoring docs (lowest BPB, most predictable)")
        for ex in examples["best"]:
            lines.append(f"- **{ex['bpb']} bpb**: {ex['text']}")
        lines.append("")

    # improvement > 0 means the transformer scored lower (better) BPB than n-gram on that doc.
    improvements = [(n - t, i) for i, (n, t) in enumerate(zip(ngram_scores, transformer_scores))
                     if n == n and t == t]
    improvements.sort(reverse=True)
    lines.append("### Where the transformer wins most over n-gram (largest BPB improvement)")
    for imp, i in improvements[:num_examples]:
        lines.append(f"- improvement {imp:.4f} bpb: {preview(texts[i])}")
    lines.append("\n### Where n-gram is relatively most competitive (smallest/negative improvement)")
    for imp, i in improvements[-num_examples:]:
        lines.append(f"- improvement {imp:.4f} bpb: {preview(texts[i])}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "wikitext103"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("eval"))
    parser.add_argument("--eval-max-docs", type=int, default=300)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    val_path = args.data_dir / args.dataset / "val.jsonl"
    print(f"Loading val docs from {val_path} (max {args.eval_max_docs})...")
    texts = read_jsonl_texts(val_path, args.eval_max_docs)

    print("Loading n-gram checkpoint...")
    ngram = load_ngram(args.checkpoints_dir, args.dataset)
    print("Loading transformer checkpoint...")
    model, tokenizer, config = load_transformer(args.checkpoints_dir, args.dataset, args.device)

    print(f"Scoring {len(texts)} docs with both models...")
    ngram_scores = ngram_per_doc_bpb(ngram, texts)
    transformer_scores = per_document_bpb(model, texts, tokenizer, config["block_size"], args.device)

    report = format_report(args.dataset, ngram_scores, transformer_scores, texts, args.num_examples)

    # Save before printing: real corpora (WikiText-103 especially) contain characters
    # outside the Windows console's default codepage, which crashes print() -- the saved
    # file must not be lost just because the console can't display every character.
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"error_analysis_{args.dataset}.md"
    out_path.write_text(report, encoding="utf-8")

    safe_print(report)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
