"""Train + evaluate the BPE Transformer baseline (10-30M params).

Usage:
    python scripts/train_bpe_transformer.py --dataset tinystories
    python scripts/train_bpe_transformer.py --dataset wikitext103 --max-steps 3000 --device cuda
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tqdm import tqdm

from data_utils import read_jsonl_texts  # also puts repo root on sys.path, see data_utils.py
from models.transformer import GPT

EOD = "<|endoftext|>"  # document separator, inserted between packed docs
NATS_TO_BITS = 1.0 / math.log(2)
TRAIN_SAMPLE_SIZE = 200  # docs used for the reported train-set BPB (train set itself can be huge)


def get_or_train_tokenizer(train_texts: list, vocab_size: int, path: Path) -> Tokenizer:
    if path.exists():
        return Tokenizer.from_file(str(path))
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(train_texts, vocab_size=vocab_size, special_tokens=[EOD])
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return Tokenizer.from_file(str(path))  # reload so we always return the plain Tokenizer type


def pack_tokens(texts: list, tokenizer: Tokenizer, eod_id: int) -> torch.Tensor:
    """Concatenate every doc's token ids into one long stream, separated by EOD."""
    ids = []
    for enc in tokenizer.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eod_id)
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    assert len(data) > block_size, (
        f"dataset has only {len(data)} tokens, needs more than block_size={block_size}"
    )
    starts = torch.randint(len(data) - block_size, (batch_size,))
    offsets = torch.arange(block_size)
    x = data[starts.unsqueeze(1) + offsets]
    y = data[starts.unsqueeze(1) + offsets + 1]
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, block_size: int, batch_size: int,
                   device: str, num_batches: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _doc_chunks(texts: list, tokenizer: Tokenizer, block_size: int):
    """Split every doc's tokens into non-overlapping <=block_size chunks, each paired with
    the exact byte length of the text span it covers and the source doc's index (so scores
    can be re-aggregated per document, e.g. for error analysis). One doc can yield multiple
    chunks; skipped (too-short) docs are counted and reported by the caller."""
    chunks = []  # (input_ids, target_ids, num_bytes, doc_index)
    skipped = 0
    for doc_index, (text, enc) in enumerate(zip(texts, tokenizer.encode_batch(texts))):
        if len(enc.ids) < 2:
            skipped += 1
            continue
        char_pos = 0
        for start in range(0, len(enc.ids), block_size):
            ids = enc.ids[start:start + block_size]
            if len(ids) < 2:
                break  # trailing sliver too short to form an (input, target) pair
            # Byte-level BPE merges don't always align with UTF-8 character boundaries, so
            # decoding truncated token ids can land mid-character and corrupt the byte count.
            # Slicing the ORIGINAL text at this chunk's character offsets avoids that.
            end_char = enc.offsets[start + len(ids) - 1][1]
            num_bytes = len(text[char_pos:end_char].encode("utf-8"))
            chunks.append((ids[:-1], ids[1:], num_bytes, doc_index))
            char_pos = end_char
    return chunks, skipped


@torch.no_grad()
def _score_chunks(model: GPT, chunks: list, device: str, batch_size: int = 32) -> list:
    """Batched, masked forward passes over chunks of possibly-different lengths -- scoring
    them one at a time is orders of magnitude slower once long documents mean many chunks
    each. Returns (bits, num_bytes, doc_index) per chunk."""
    model.eval()
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        max_len = max(len(inp) for inp, _, _, _ in batch)
        idx = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        targets = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool, device=device)
        for j, (inp, tgt, _, _) in enumerate(batch):
            n = len(inp)
            idx[j, :n] = torch.tensor(inp, device=device)
            targets[j, :n] = torch.tensor(tgt, device=device)
            mask[j, :n] = True

        logits, _ = model(idx)  # no targets -> raw logits, we do the (masked) loss ourselves
        token_losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                        reduction="none").view(len(batch), max_len)
        per_chunk_bits = (token_losses * mask).sum(dim=1) * NATS_TO_BITS
        for (_, _, num_bytes, doc_index), bits in zip(batch, per_chunk_bits.tolist()):
            results.append((bits, num_bytes, doc_index))
    model.train()
    return results


def bits_per_byte(model: GPT, texts: list, tokenizer: Tokenizer,
                   block_size: int, device: str, batch_size: int = 32) -> float:
    """Bits-per-byte over whole documents -- comparable to the n-gram baseline's metric
    (which also scores full documents), regardless of BPE vocab size. See _doc_chunks/
    _score_chunks for how fixed-context scoring over long documents works.
    """
    chunks, skipped = _doc_chunks(texts, tokenizer, block_size)
    chunk_results = _score_chunks(model, chunks, device, batch_size)
    total_bits = sum(bits for bits, _, _ in chunk_results)
    total_bytes = sum(num_bytes for _, num_bytes, _ in chunk_results)
    if skipped:
        print(f"  (skipped {skipped}/{len(texts)} docs with <2 tokens)")
    return total_bits / total_bytes if total_bytes else float("nan")


def per_document_bpb(model: GPT, texts: list, tokenizer: Tokenizer,
                      block_size: int, device: str, batch_size: int = 32) -> list:
    """Bits-per-byte for each document individually (nan for a doc too short to score),
    aligned index-for-index with `texts` -- used for error analysis, not training."""
    chunks, _ = _doc_chunks(texts, tokenizer, block_size)
    chunk_results = _score_chunks(model, chunks, device, batch_size)

    bits_per_doc = [0.0] * len(texts)
    bytes_per_doc = [0] * len(texts)
    for bits, num_bytes, doc_index in chunk_results:
        bits_per_doc[doc_index] += bits
        bytes_per_doc[doc_index] += num_bytes

    return [bits_per_doc[i] / bytes_per_doc[i] if bytes_per_doc[i] else float("nan")
            for i in range(len(texts))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "wikitext103"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))

    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--max-train-docs", type=int, default=None,
                         help="Cap training docs. None = full corpus (fine on GPU; unlike the "
                              "n-gram baseline, this script isn't pure-Python-bottlenecked).")
    parser.add_argument("--eval-max-docs", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_path = args.data_dir / args.dataset / "train.jsonl"
    val_path = args.data_dir / args.dataset / "val.jsonl"
    print(f"Loading train docs from {train_path} (max {args.max_train_docs})...")
    train_texts = read_jsonl_texts(train_path, args.max_train_docs)
    print(f"Loading val docs from {val_path} (max {args.eval_max_docs})...")
    val_texts = read_jsonl_texts(val_path, args.eval_max_docs)

    # vocab_size and the training doc count are both part of the tokenizer's identity --
    # a cache trained with different values would otherwise be silently reused (e.g. a
    # quick smoke test's small-sample tokenizer getting reused for the real full-corpus run).
    docs_tag = args.max_train_docs if args.max_train_docs is not None else "all"
    tokenizer_path = args.out / f"bpe_{args.dataset}_v{args.vocab_size}_d{docs_tag}_tokenizer.json"
    tokenizer = get_or_train_tokenizer(train_texts, args.vocab_size, tokenizer_path)
    eod_id = tokenizer.token_to_id(EOD)
    vocab_size = tokenizer.get_vocab_size()

    if vocab_size != args.vocab_size and tokenizer_path.exists():
        # BPE on a small corpus can converge to fewer merges than requested, so the actual
        # vocab (== the model's embedding size, what load_transformer() will search for
        # later) can differ from the requested size baked into the filename above -- rename
        # so a future load can find this file by its real vocab size.
        actual_path = args.out / f"bpe_{args.dataset}_v{vocab_size}_d{docs_tag}_tokenizer.json"
        tokenizer_path.rename(actual_path)

    print("Packing tokens...")
    train_data = pack_tokens(train_texts, tokenizer, eod_id)
    val_data = pack_tokens(val_texts, tokenizer, eod_id)
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,} | Vocab: {vocab_size}")

    model = GPT(vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout)
    model.to(args.device)
    print(f"Model params: {model.num_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        return args.lr

    print(f"Training for {args.max_steps} steps on {args.device}...")
    t0 = time.time()
    for step in tqdm(range(args.max_steps), desc="training", unit="step"):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)

        x, y = get_batch(train_data, args.block_size, args.batch_size, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # standard divergence guard
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.device)
            tqdm.write(f"step {step:5d} | train_loss {loss.item():.3f} | val_loss {val_loss:.3f}")
    train_time = time.time() - t0

    final_val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.device, num_batches=50)
    sample_size = min(TRAIN_SAMPLE_SIZE, len(train_texts))
    train_bpb = bits_per_byte(model, train_texts[:sample_size], tokenizer, args.block_size, args.device)
    val_bpb = bits_per_byte(model, val_texts, tokenizer, args.block_size, args.device)
    print(f"Train time: {train_time:.1f}s")
    print(f"Final val loss: {final_val_loss:.3f}")
    print(f"Train BPB ({sample_size}-doc sample): {train_bpb:.3f}")
    print(f"Val BPB: {val_bpb:.3f}")

    # Hyperparameters go in the filename (like the n-gram baseline's order{K}) so a sweep
    # over architectures doesn't silently overwrite a previous run's checkpoint/results.
    tag = f"{args.dataset}_L{args.n_layer}H{args.n_head}D{args.n_embd}V{vocab_size}"
    config = {
        "vocab_size": vocab_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / f"transformer_{tag}.pt"
    torch.save({"config": config, "state_dict": model.state_dict()}, model_path)
    print(f"Saved model to {model_path}")

    # Flat schema (no nested sub-dict) so a harness reading this alongside the n-gram
    # baseline's results.json can look up any field the same way in both files.
    results = {
        "model": "bpe_transformer",
        "dataset": args.dataset,
        **config,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "num_params": model.num_params(),
        "train_docs": len(train_texts),
        "val_docs": len(val_texts),
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "train_time_sec": round(train_time, 1),
        "final_val_loss": round(final_val_loss, 4),
        "train_bpb_sample": round(train_bpb, 4),
        "val_bpb": round(val_bpb, 4),
    }
    results_path = args.out / f"transformer_{tag}_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
