"""Step 2b: train and evaluate the BPE Transformer baseline (13.8M parameters by default).

Trains (or reuses) a byte-level BPE tokenizer on the training documents, packs all
documents into one token stream separated by <|endoftext|>, trains models/transformer.py
on random 256-token windows, then scores whole validation documents in bits per byte.
Writes the tokenizer, checkpoints/transformer_<tag>.pt and a _results.json that
bpb_harness.py reads. bits_per_byte and per_document_bpb are also imported by
error_analysis.py, demo.py and check_setup.py.

Usage (the reported runs; both used --grad-clip 0, see the README):
    python scripts/train_bpe_transformer.py --dataset tinystories --max-steps 3000 --grad-clip 0
    python scripts/train_bpe_transformer.py --dataset wikitext103 --max-steps 3000 --grad-clip 0
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tqdm import tqdm

from data_utils import check_not_truncated, read_jsonl_texts  # also puts repo root on sys.path
from models.transformer import GPT

EOD = "<|endoftext|>"  # document separator, inserted between packed docs
NATS_TO_BITS = 1.0 / math.log(2)
TRAIN_SAMPLE_SIZE = 200  # train-set BPB is reported on this many docs only, to keep it fast


def get_or_train_tokenizer(train_texts: list, vocab_size: int, path: Path) -> Tokenizer:
    """Load the tokenizer at path if it exists, else train a byte-level BPE one and save it."""
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
    """batch_size random windows from the packed stream; y is x shifted one token left."""
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
    """Mean cross-entropy (nats per token) over num_batches random windows. Only used
    for the progress log; the reported metric is bits_per_byte below."""
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _doc_chunks(texts: list, tokenizer: Tokenizer, block_size: int):
    """Split each document's tokens into non-overlapping chunks of at most block_size tokens.

    Documents are longer than the 256-token context, and truncating them would score less
    text than the n-gram model does, so every byte of every document is covered by some
    chunk. Each chunk is (input_ids, target_ids, num_bytes, doc_index): num_bytes is the
    UTF-8 length of the text the chunk covers (the BPB denominator) and doc_index lets
    per_document_bpb regroup chunks by document. The first token of each chunk is only
    context, never a target, so its bits are not counted but its bytes are (0.37 to 0.55
    percent of bytes on our evaluation sets; this makes the reported BPB slightly optimistic,
    see the README section on evaluation)."""
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
    """Total bits for each chunk, as (bits, num_bytes, doc_index).

    Chunks are padded to the longest in the batch and a mask drops the padded positions
    from the loss, so chunks of different lengths can share one forward pass (one chunk
    at a time was far too slow on WikiText). Loss is in nats; NATS_TO_BITS converts."""
    was_training = model.training
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
    model.train(was_training)
    return results


def bits_per_byte(model: GPT, texts: list, tokenizer: Tokenizer,
                   block_size: int, device: str, batch_size: int = 32) -> float:
    """Bits per byte over whole documents: total bits / total UTF-8 bytes.

    Dividing by bytes rather than tokens makes the number independent of the tokenizer,
    so it is directly comparable with the byte-level n-gram model."""
    chunks, skipped = _doc_chunks(texts, tokenizer, block_size)
    chunk_results = _score_chunks(model, chunks, device, batch_size)
    total_bits = sum(bits for bits, _, _ in chunk_results)
    total_bytes = sum(num_bytes for _, num_bytes, _ in chunk_results)
    if skipped:
        print(f"  (skipped {skipped}/{len(texts)} docs with <2 tokens)")
    return total_bits / total_bytes if total_bytes else float("nan")


def per_document_bpb(model: GPT, texts: list, tokenizer: Tokenizer,
                      block_size: int, device: str, batch_size: int = 32) -> list:
    """Bits per byte of each document separately (nan if too short to score), in the same
    order as texts. Used by error_analysis.py and the demo, not by training."""
    chunks, _ = _doc_chunks(texts, tokenizer, block_size)
    chunk_results = _score_chunks(model, chunks, device, batch_size)

    bits_per_doc = [0.0] * len(texts)
    bytes_per_doc = [0] * len(texts)
    for bits, num_bytes, doc_index in chunk_results:
        bits_per_doc[doc_index] += bits
        bytes_per_doc[doc_index] += num_bytes

    return [bits_per_doc[i] / bytes_per_doc[i] if bytes_per_doc[i] else float("nan")
            for i in range(len(texts))]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tinystories", "wikitext103"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="processed data folder")
    parser.add_argument("--out", type=Path, default=Path("checkpoints"),
                         help="where to write the tokenizer, model and results")

    parser.add_argument("--vocab-size", type=int, default=8000, help="BPE vocabulary size")
    parser.add_argument("--block-size", type=int, default=256, help="context length in tokens")
    parser.add_argument("--n-layer", type=int, default=6, help="transformer blocks")
    parser.add_argument("--n-head", type=int, default=6, help="attention heads per block")
    parser.add_argument("--n-embd", type=int, default=384, help="model width")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max-steps", type=int, default=1000, help="optimizer steps (the reported runs used 3000)")
    parser.add_argument("--batch-size", type=int, default=32, help="windows per step")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate after warm-up")
    parser.add_argument("--warmup-steps", type=int, default=100, help="linear learning-rate warm-up steps")
    parser.add_argument("--eval-interval", type=int, default=200, help="log validation loss every N steps")
    parser.add_argument("--max-train-docs", type=int, default=None,
                         help="training documents to use (default: all of train.jsonl)")
    parser.add_argument("--eval-max-docs", type=int, default=500, help="validation documents to score")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                         help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                         help="max gradient norm, 0 disables clipping. The reported checkpoints were "
                              "trained before this flag existed, which is equivalent to 0.")
    return parser.parse_args()


def load_texts(args) -> tuple:
    """The training and validation documents, as lists of strings."""
    train_path = args.data_dir / args.dataset / "train.jsonl"
    val_path = args.data_dir / args.dataset / "val.jsonl"
    print(f"Loading train docs from {train_path} (max {args.max_train_docs})...")
    train_texts = read_jsonl_texts(train_path, args.max_train_docs)
    check_not_truncated(train_path, len(train_texts), args.max_train_docs)
    print(f"Loading val docs from {val_path} (max {args.eval_max_docs})...")
    val_texts = read_jsonl_texts(val_path, args.eval_max_docs)
    return train_texts, val_texts


def load_tokenizer(train_texts: list, args) -> Tokenizer:
    """Reuse or train the BPE tokenizer for this dataset, vocab size and training-doc count."""
    # The vocab size and training-doc count are in the tokenizer filename so a tokenizer
    # trained on a small sample is never silently reused for a full-corpus run.
    docs_tag = args.max_train_docs if args.max_train_docs is not None else "all"
    tokenizer_path = args.out / f"bpe_{args.dataset}_v{args.vocab_size}_d{docs_tag}_tokenizer.json"
    tokenizer = get_or_train_tokenizer(train_texts, args.vocab_size, tokenizer_path)

    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != args.vocab_size:
        # BPE on a small corpus can stop with fewer merges than requested. The model's
        # embedding size is the actual vocab, and load_transformer() looks the tokenizer up
        # by that number, so also save it under the actual size. The requested-size file
        # stays in place so the next run with the same arguments reuses it.
        actual_path = args.out / f"bpe_{args.dataset}_v{vocab_size}_d{docs_tag}_tokenizer.json"
        shutil.copyfile(tokenizer_path, actual_path)
    return tokenizer


def train(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor, args) -> float:
    """Run args.max_steps AdamW steps with linear warm-up; return the training time in seconds."""
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
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.device)
            tqdm.write(f"step {step:5d} | train_loss {loss.item():.3f} | val_loss {val_loss:.3f}")
    return time.time() - t0


def save_outputs(model: GPT, args, metrics: dict) -> None:
    """Write checkpoints/transformer_<tag>.pt and the matching _results.json."""
    # Hyperparameters go in the filename so a run with a different architecture cannot
    # overwrite an earlier checkpoint.
    vocab_size = model.tok_emb.num_embeddings
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

    # Flat keys, shared with the n-gram results file, so bpb_harness.py reads both the same way.
    results = {
        "model": "bpe_transformer",
        "dataset": args.dataset,
        **config,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "num_params": model.num_params(),
        **metrics,
    }
    results_path = args.out / f"transformer_{tag}_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {results_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_texts, val_texts = load_texts(args)
    tokenizer = load_tokenizer(train_texts, args)
    vocab_size = tokenizer.get_vocab_size()

    print("Packing tokens...")
    eod_id = tokenizer.token_to_id(EOD)
    train_data = pack_tokens(train_texts, tokenizer, eod_id)
    val_data = pack_tokens(val_texts, tokenizer, eod_id)
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,} | Vocab: {vocab_size}")

    model = GPT(vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout)
    model.to(args.device)
    print(f"Model params: {model.num_params():,}")

    train_time = train(model, train_data, val_data, args)

    final_val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.device, num_batches=50)
    sample_size = min(TRAIN_SAMPLE_SIZE, len(train_texts))
    train_bpb = bits_per_byte(model, train_texts[:sample_size], tokenizer, args.block_size, args.device)
    val_bpb = bits_per_byte(model, val_texts, tokenizer, args.block_size, args.device)
    print(f"Train time: {train_time:.1f}s")
    print(f"Final val loss: {final_val_loss:.3f}")
    print(f"Train BPB ({sample_size}-doc sample): {train_bpb:.3f}")
    print(f"Val BPB: {val_bpb:.3f}")

    save_outputs(model, args, {
        "train_docs": len(train_texts),
        "val_docs": len(val_texts),
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "train_time_sec": round(train_time, 1),
        "final_val_loss": round(final_val_loss, 4),
        "train_bpb_sample": round(train_bpb, 4),
        "val_bpb": round(val_bpb, 4),
    })


if __name__ == "__main__":
    main()
