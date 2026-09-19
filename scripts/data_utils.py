"""Shared helpers: reading data/processed/<dataset>/<split>.jsonl, and loading trained
checkpoints (used by error_analysis.py and demo.py -- both need "the trained model +
its matching tokenizer for this dataset" and shouldn't each reimplement that lookup).

Also puts the repo root on sys.path (needed for `from models... import ...`) as a side
effect of importing this module -- every script here imports it, so this is the one place
that needs to do it rather than repeating the same sys.path hack in every script.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def safe_print(text: str) -> None:
    """print() that can't crash on real corpus text -- Windows consoles default to a
    codepage (e.g. cp1252) that can't display arbitrary Unicode (WikiText-103 alone has
    Greek letters, em-dashes, etc.), and an unhandled UnicodeEncodeError would otherwise
    kill the script after the interesting work is already done."""
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def read_jsonl_texts(path: Path, max_docs: int = None) -> list:
    """Read the "text" field of each line, capped at max_docs docs (None = all).
    Blank lines are skipped and don't count against max_docs."""
    docs = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            if max_docs is not None and len(docs) >= max_docs:
                break
            try:
                docs.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"{path}:{line_num}: {e}") from e
    return docs


def _find_latest(checkpoints_dir: Path, pattern: str) -> Path:
    matches = sorted(checkpoints_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {checkpoints_dir}")
    return matches[-1]


def load_ngram(checkpoints_dir: Path, dataset: str):
    """Load the most recently trained n-gram checkpoint for `dataset`."""
    from models.ngram import KneserNeyByteLM
    path = _find_latest(checkpoints_dir, f"ngram_{dataset}_order*.pkl")
    return KneserNeyByteLM.load(path)


def load_transformer(checkpoints_dir: Path, dataset: str, device: str = "cpu"):
    """Load the most recently trained transformer checkpoint for `dataset`, plus the
    tokenizer it was trained with. Returns (model, tokenizer, config)."""
    import torch
    from tokenizers import Tokenizer
    from models.transformer import GPT

    ckpt_path = _find_latest(checkpoints_dir, f"transformer_{dataset}_*.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    vocab_size = ckpt["config"]["vocab_size"]
    tok_path = _find_latest(checkpoints_dir, f"bpe_{dataset}_v{vocab_size}_*_tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tok_path))
    return model, tokenizer, ckpt["config"]
