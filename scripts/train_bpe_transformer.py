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

from data_utils import check_not_truncated, read_jsonl_texts
from models.transformer import GPT

EOD = "<|endoftext|>"
NATS_TO_BITS = 1.0 / math.log(2)
TRAIN_SAMPLE_SIZE = 200


def get_or_train_tokenizer(train_texts: list, vocab_size: int, path: Path) -> Tokenizer:
    if not path.exists():
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(train_texts, vocab_size=vocab_size, special_tokens=[EOD])
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(path))
    return Tokenizer.from_file(str(path))


def pack_tokens(texts: list, tokenizer: Tokenizer, eod_id: int) -> torch.Tensor:
    ids = []
    for enc in tokenizer.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eod_id)
    return torch.tensor(ids, dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    assert len(data) > block_size, f"dataset has {len(data)} tokens, needs more than {block_size}"
    idx = torch.randint(len(data) - block_size, (batch_size, 1)) + torch.arange(block_size)
    return data[idx].to(device), data[idx + 1].to(device)


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, block_size: int, batch_size: int,
                  device: str, num_batches: int = 20) -> float:
    model.eval()
    losses = [model(*get_batch(data, block_size, batch_size, device))[1].item() for _ in range(num_batches)]
    model.train()
    return sum(losses) / len(losses)


def doc_chunks(texts: list, tokenizer: Tokenizer, block_size: int):
    chunks, skipped = [], 0
    for doc_index, (text, enc) in enumerate(zip(texts, tokenizer.encode_batch(texts))):
        if len(enc.ids) < 2:
            skipped += 1
            continue
        char_pos = 0
        for start in range(0, len(enc.ids), block_size):
            ids = enc.ids[start:start + block_size]
            if len(ids) < 2:
                break
            end_char = enc.offsets[start + len(ids) - 1][1]
            chunks.append((ids[:-1], ids[1:], len(text[char_pos:end_char].encode("utf-8")), doc_index))
            char_pos = end_char
    return chunks, skipped


@torch.no_grad()
def score_chunks(model: GPT, chunks: list, device: str, batch_size: int = 32) -> list:
    was_training = model.training
    model.eval()
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        max_len = max(len(inp) for inp, *_ in batch)
        idx = torch.zeros(len(batch), max_len, dtype=torch.long)
        targets = torch.zeros(len(batch), max_len, dtype=torch.long)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for j, (inp, tgt, _, _) in enumerate(batch):
            idx[j, :len(inp)] = torch.tensor(inp)
            targets[j, :len(tgt)] = torch.tensor(tgt)
            mask[j, :len(inp)] = True
        idx, targets, mask = idx.to(device), targets.to(device), mask.to(device)
        logits, _ = model(idx)
        nats = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view(len(batch), max_len)
        bits = ((nats * mask).sum(dim=1) * NATS_TO_BITS).tolist()
        results.extend((b, num_bytes, doc_index) for b, (_, _, num_bytes, doc_index) in zip(bits, batch))
    model.train(was_training)
    return results


def bits_per_byte(model: GPT, texts: list, tokenizer: Tokenizer, block_size: int,
                  device: str, batch_size: int = 32) -> float:
    chunks, skipped = doc_chunks(texts, tokenizer, block_size)
    results = score_chunks(model, chunks, device, batch_size)
    if skipped:
        print(f"  (skipped {skipped}/{len(texts)} docs with <2 tokens)")
    total_bytes = sum(n for _, n, _ in results)
    return sum(b for b, _, _ in results) / total_bytes if total_bytes else float("nan")


def per_document_bpb(model: GPT, texts: list, tokenizer: Tokenizer, block_size: int,
                     device: str, batch_size: int = 32) -> list:
    chunks, _ = doc_chunks(texts, tokenizer, block_size)
    bits, nbytes = [0.0] * len(texts), [0] * len(texts)
    for b, n, i in score_chunks(model, chunks, device, batch_size):
        bits[i] += b
        nbytes[i] += n
    return [b / n if n else float("nan") for b, n in zip(bits, nbytes)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tinystories", "wikitext103"], required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--max-train-docs", type=int, default=None)
    p.add_argument("--eval-max-docs", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables; the reported checkpoints used 0")
    return p.parse_args()


def load_tokenizer(train_texts: list, args) -> Tokenizer:
    docs_tag = args.max_train_docs if args.max_train_docs is not None else "all"
    path = args.out / f"bpe_{args.dataset}_v{args.vocab_size}_d{docs_tag}_tokenizer.json"
    tokenizer = get_or_train_tokenizer(train_texts, args.vocab_size, path)
    actual = tokenizer.get_vocab_size()
    if actual != args.vocab_size:
        shutil.copyfile(path, args.out / f"bpe_{args.dataset}_v{actual}_d{docs_tag}_tokenizer.json")
    return tokenizer


def train(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor, args) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Training for {args.max_steps} steps on {args.device}...")
    t0 = time.time()
    for step in tqdm(range(args.max_steps), desc="training", unit="step"):
        for g in optimizer.param_groups:
            g["lr"] = args.lr * (step + 1) / args.warmup_steps if step < args.warmup_steps else args.lr
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
    vocab_size = model.tok_emb.num_embeddings
    tag = f"{args.dataset}_L{args.n_layer}H{args.n_head}D{args.n_embd}V{vocab_size}"
    config = {"vocab_size": vocab_size, "block_size": args.block_size, "n_layer": args.n_layer,
              "n_head": args.n_head, "n_embd": args.n_embd, "dropout": args.dropout}
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / f"transformer_{tag}.pt"
    torch.save({"config": config, "state_dict": model.state_dict()}, model_path)
    results = {"model": "bpe_transformer", "dataset": args.dataset, **config,
               "batch_size": args.batch_size, "lr": args.lr, "max_steps": args.max_steps,
               "grad_clip": args.grad_clip, "seed": args.seed, "num_params": model.num_params(), **metrics}
    results_path = args.out / f"transformer_{tag}_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {model_path} and {results_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    train_path = args.data_dir / args.dataset / "train.jsonl"
    train_texts = read_jsonl_texts(train_path, args.max_train_docs)
    check_not_truncated(train_path, len(train_texts), args.max_train_docs)
    val_texts = read_jsonl_texts(args.data_dir / args.dataset / "val.jsonl", args.eval_max_docs)
    print(f"Loaded {len(train_texts):,} train and {len(val_texts):,} val docs")

    tokenizer = load_tokenizer(train_texts, args)
    eod_id = tokenizer.token_to_id(EOD)
    train_data = pack_tokens(train_texts, tokenizer, eod_id)
    val_data = pack_tokens(val_texts, tokenizer, eod_id)
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,} | Vocab: {tokenizer.get_vocab_size()}")

    model = GPT(tokenizer.get_vocab_size(), args.block_size, args.n_layer, args.n_head,
                args.n_embd, args.dropout).to(args.device)
    print(f"Model params: {model.num_params():,}")
    train_time = train(model, train_data, val_data, args)

    final_val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, args.device, num_batches=50)
    sample = train_texts[:TRAIN_SAMPLE_SIZE]
    train_bpb = bits_per_byte(model, sample, tokenizer, args.block_size, args.device)
    val_bpb = bits_per_byte(model, val_texts, tokenizer, args.block_size, args.device)
    print(f"Train time {train_time:.1f}s | final val loss {final_val_loss:.3f} | "
          f"train BPB ({len(sample)} docs) {train_bpb:.3f} | val BPB {val_bpb:.3f}")
    save_outputs(model, args, {
        "train_docs": len(train_texts), "val_docs": len(val_texts),
        "train_tokens": len(train_data), "val_tokens": len(val_data),
        "train_time_sec": round(train_time, 1), "final_val_loss": round(final_val_loss, 4),
        "train_bpb_sample": round(train_bpb, 4), "val_bpb": round(val_bpb, 4),
    })


if __name__ == "__main__":
    main()
