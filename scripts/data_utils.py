import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def read_jsonl_texts(path: Path, max_docs: int = None) -> list:
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
    stats_path = path.parent / "stats.json"
    if not stats_path.exists():
        return
    expected = json.loads(stats_path.read_text(encoding="utf-8")).get(path.stem, {}).get("num_docs")
    if expected is None:
        return
    wanted = expected if max_docs is None else min(max_docs, expected)
    if docs_read < wanted:
        sys.exit(f"{path} has {docs_read:,} documents but {wanted:,} were requested. This is the "
                 "truncated release copy; rebuild it with scripts/prepare_data.py.")


def _find_latest(checkpoints_dir: Path, pattern: str) -> Path:
    matches = sorted(checkpoints_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {checkpoints_dir}")
    return matches[-1]


def load_ngram(checkpoints_dir: Path, dataset: str):
    from models.ngram import KneserNeyByteLM
    return KneserNeyByteLM.load(_find_latest(checkpoints_dir, f"ngram_{dataset}_order*.pkl"))


def load_transformer(checkpoints_dir: Path, dataset: str, device: str = "cpu"):
    import torch
    from tokenizers import Tokenizer
    from models.transformer import GPT

    ckpt = torch.load(_find_latest(checkpoints_dir, f"transformer_{dataset}_*.pt"),
                      map_location=device, weights_only=True)
    model = GPT(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    vocab_size = ckpt["config"]["vocab_size"]
    tokenizer = Tokenizer.from_file(str(_find_latest(checkpoints_dir, f"bpe_{dataset}_v{vocab_size}_*_tokenizer.json")))
    return model, tokenizer, ckpt["config"]
