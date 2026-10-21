import argparse
import json
import math

import torch

from data_utils import REPO_ROOT, load_ngram, load_transformer, read_jsonl_texts
from train_bpe_transformer import NATS_TO_BITS, doc_chunks, score_chunks

CHECKPOINTS = REPO_ROOT / "checkpoints"
SETS = {"tinystories": ("val", 500), "wikitext103": ("val", 60)}
FIELDS = ["bpb", "mean_patch_len", "cache_bytes_per_1000"]


def load_docs(dataset: str, split: str, n: int) -> list:
    return read_jsonl_texts(REPO_ROOT / "data" / "processed" / dataset / f"{split}.jsonl", n)


def ngram_row(dataset: str, texts: list) -> dict:
    model = load_ngram(CHECKPOINTS, dataset)
    bits = sum(-model.log2_prob_doc(t.encode("utf-8")) for t in texts if t)
    return {"bpb": bits / sum(len(t.encode("utf-8")) for t in texts)}


def first_token_bytes(texts: list, tokenizer, block: int) -> int:
    total = 0
    for text, enc in zip(texts, tokenizer.encode_batch(texts)):
        if len(enc.ids) < 2:
            continue
        char_pos = 0
        for start in range(0, len(enc.ids), block):
            if len(enc.ids[start:start + block]) < 2:
                break
            total += len(text[char_pos:enc.offsets[start][1]].encode("utf-8"))
            char_pos = enc.offsets[start + len(enc.ids[start:start + block]) - 1][1]
    return total


def bpe_rows(dataset: str, texts: list, device: str) -> tuple:
    model, tokenizer, cfg = load_transformer(CHECKPOINTS, dataset, device)
    chunks, _ = doc_chunks(texts, tokenizer, cfg["block_size"])
    results = score_chunks(model, chunks, device)
    bits, nbytes = sum(b for b, _, _ in results), sum(n for _, n, _ in results)
    tokens = sum(len(i) for i, *_ in chunks)
    cache = 2 * cfg["n_layer"] * cfg["n_embd"] * 2
    extra = {"cache_bytes_per_1000": round(1000 * tokens / nbytes * cache), "mean_patch_len": round(nbytes / tokens, 3)}
    corrected = bits / (nbytes - first_token_bytes(texts, tokenizer, cfg["block_size"]))
    return {"bpb": bits / nbytes, **extra}, {"bpb": corrected, **extra}


def run_json(run: str, name: str) -> dict:
    path = REPO_ROOT / "runs" / run / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(row, key) -> str:
    if row is None or key not in row:
        return "n/a"
    v = row[key]
    return f"{v:,}" if isinstance(v, int) else f"{v:.4f}" if key == "bpb" else f"{v:g}"


def phase2(device: str) -> str:
    lines = ["Matched-document BPB (lower is better). Test splits are not used.", ""]
    for dataset, (split, n) in SETS.items():
        texts = load_docs(dataset, split, n)
        reported, corrected = bpe_rows(dataset, texts, device)
        entropy = run_json(f"entropy_{dataset}", "patching.json")
        rows = [("Kneser-Ney byte 4-gram", ngram_row(dataset, texts)),
                ("BPE transformer, reported", reported), ("BPE transformer, corrected", corrected),
                ("Entropy model (3.3M)", entropy and {"bpb": entropy["bpb"], "mean_patch_len": entropy["mean_patch_len"]})]
        for mode in ("fixed", "entropy"):
            r = run_json(f"blt_{dataset}_{mode}", f"eval_{split}_{n}.json")
            rows.append((f"BLT {mode} patching (13.3M)", r and {"bpb": r["bpb"], "mean_patch_len": r["mean_patch_len"],
                                                               "cache_bytes_per_1000": r["global_cache_bytes_per_1000"]}))
        lines += [f"### {dataset} {split}, first {n} docs", "", "| Model | BPB | Bytes per patch or token | Global KV bytes per 1000 input bytes |", "|---|---|---|---|"]
        lines += [f"| {name} | {fmt(row, 'bpb')} | {fmt(row, 'mean_patch_len')} | {fmt(row, 'cache_bytes_per_1000')} |" for name, row in rows]
        lines.append("")
    lines.append("Fixed overhead not in the table: BLT local encoder and decoder windows 0.79 MB, entropy model window 2.10 MB (fp16).")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = phase2(device)
    print(text)
    if args.out:
        (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
