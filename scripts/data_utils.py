"""Shared helpers used by every other script: read data/processed/<dataset>/<split>.jsonl,
load a trained checkpoint (with its matching tokenizer), and print corpus text safely.

Importing this module also puts the repo root on sys.path, so `from models... import ...`
works however a script is launched. Doing it here means no other script repeats it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def safe_print(text: str) -> None:
    """print() that cannot crash on corpus text. Windows consoles default to a codepage
    (e.g. cp1252) that cannot show every Unicode character in WikiText-103, and the
    resulting UnicodeEncodeError would otherwise kill the script mid-demo."""
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def read_jsonl_texts(path: Path, max_docs: int = None) -> list:
    """Read the "text" field of each line, stopping after max_docs documents (None = all).
    Blank lines are skipped and do not count. Because reading stops early, a file holding
    only the first N documents gives the same result as the full file for max_docs <= N."""
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


def check_not_truncated(path: Path, docs_read: int, max_docs: int = None) -> None:
    """Exit if path holds fewer documents than were asked for and than its stats.json records.

    The release ships data/processed/wikitext103/train.jsonl cut to its first 3,000
    documents. Training on it with a larger --max-train-docs (or none) would otherwise run
    on less data than requested without any warning."""
    stats_path = path.parent / "stats.json"
    if not stats_path.exists():
        return
    expected = json.loads(stats_path.read_text(encoding="utf-8")).get(path.stem, {}).get("num_docs")
    if expected is None:
        return
    wanted = expected if max_docs is None else min(max_docs, expected)
    if docs_read < wanted:
        sys.exit(f"{path} has only {docs_read:,} documents but {wanted:,} were requested "
                 f"({stats_path.name} records {expected:,}). This is probably the truncated copy "
                 "from the release; rebuild the full file with scripts/prepare_data.py.")


def _find_latest(checkpoints_dir: Path, pattern: str) -> Path:
    """Newest file (by modification time) in checkpoints_dir matching pattern."""
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
