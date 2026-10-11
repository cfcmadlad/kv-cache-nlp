"""Smoke test: confirm this machine can run the demo with the downloaded models.

Checks that PyTorch and the other required packages import (and whether CUDA is
available), that every file the demo needs is present, loads all four trained models, scores one fixed sentence with each,
checks the transformer parameter count and the BPB values stored in the results JSONs,
and prints the comparison table that demo option 2 shows. No training, under a minute.

--full additionally re-scores the reported evaluation sets from scratch and checks the
six headline numbers (about 30 seconds on a GPU, about 2 minutes on a CPU).

Usage:
    python scripts/check_setup.py
    python scripts/check_setup.py --full
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DATA_DIR = REPO_ROOT / "data" / "processed"
DATASETS = ["tinystories", "wikitext103"]
SENTENCE = "Once upon a time, there was a little girl who loved to read books."
TRANSFORMER_PARAMS = 13_817_856

# val_bpb as written by the training scripts, per (model, dataset).
RESULTS_JSON_BPB = {
    ("kneser_ney_ngram", "tinystories"): 1.8477,
    ("kneser_ney_ngram", "wikitext103"): 2.4103,
    ("bpe_transformer", "tinystories"): 0.6939,
    ("bpe_transformer", "wikitext103"): 1.5200,
}

# Reported matched-document BPB: (dataset, split, docs) -> (n-gram, transformer).
REPORTED_BPB = {
    ("tinystories", "val", 500): (1.8085, 0.6939),
    ("wikitext103", "val", 60): (2.4103, 1.5200),
    ("wikitext103", "test", 62): (2.4381, 1.5309),
}

REQUIRED_FILES = [
    "checkpoints/ngram_tinystories_order4.pkl",
    "checkpoints/ngram_wikitext103_order4.pkl",
    "checkpoints/transformer_tinystories_L6H6D384V8000.pt",
    "checkpoints/transformer_wikitext103_L6H6D384V8000.pt",
    "checkpoints/bpe_tinystories_v8000_dall_tokenizer.json",
    "checkpoints/bpe_wikitext103_v8000_dall_tokenizer.json",
    "checkpoints/ngram_tinystories_order4_results.json",
    "checkpoints/ngram_wikitext103_order4_results.json",
    "checkpoints/transformer_tinystories_L6H6D384V8000_results.json",
    "checkpoints/transformer_wikitext103_L6H6D384V8000_results.json",
    "data/processed/tinystories/train.jsonl",
    "data/processed/tinystories/val.jsonl",
    "data/processed/wikitext103/train.jsonl",
    "data/processed/wikitext103/val.jsonl",
    "data/processed/wikitext103/test.jsonl",
]


class SetupError(Exception):
    """A failed check, with a message that says how to fix it."""


def check_torch(requested: str):
    """Report the PyTorch version and GPU, and return the device to use."""
    try:
        import torch
    except ImportError:
        raise SetupError("PyTorch is not installed. Install it first (see the README Quickstart).")
    has_cuda = torch.cuda.is_available()
    gpu = f" ({torch.cuda.get_device_name(0)})" if has_cuda else " (CPU only, the demo still works)"
    print(f"torch {torch.__version__}, CUDA available: {has_cuda}{gpu}")
    if requested == "cuda" and not has_cuda:
        raise SetupError("--device cuda was given but CUDA is not available. Use --device cpu, "
                         "or install a CUDA build of PyTorch (see the README Quickstart).")
    device = requested if requested != "auto" else ("cuda" if has_cuda else "cpu")
    print(f"Using device: {device}")
    return device


def check_packages():
    """Fail with an install hint if a package from requirements.txt is missing."""
    for package in ["tokenizers", "tqdm"]:
        try:
            __import__(package)
        except ImportError:
            raise SetupError(f"The {package} package is not installed. Run: pip install -r requirements.txt")


def check_files():
    """Fail if any file the demo needs is missing."""
    missing = [f for f in REQUIRED_FILES if not (REPO_ROOT / f).exists()]
    if missing:
        raise SetupError("Missing files:\n  " + "\n  ".join(missing)
                         + "\nRun: python scripts/fetch_assets.py")
    print(f"All {len(REQUIRED_FILES)} required files present.")


def load_models(device):
    """Load both models for both datasets and check the Transformer parameter count."""
    from data_utils import load_ngram, load_transformer

    models = {}
    for dataset in DATASETS:
        try:
            models[("ngram", dataset)] = load_ngram(CHECKPOINTS_DIR, dataset)
            models[("transformer", dataset)] = load_transformer(CHECKPOINTS_DIR, dataset, device)
        except Exception as e:
            raise SetupError(f"Could not load the {dataset} models ({e}). "
                             "Re-download them with: python scripts/fetch_assets.py --force")
        num_params = models[("transformer", dataset)][0].num_params()
        if num_params != TRANSFORMER_PARAMS:
            raise SetupError(f"{dataset} transformer has {num_params:,} parameters, "
                             f"expected {TRANSFORMER_PARAMS:,}. Wrong checkpoint file?")
    print(f"Loaded 2 n-gram models and 2 transformers ({TRANSFORMER_PARAMS:,} parameters each).")
    return models


def score_sentence(models, device):
    """Print the BPB of SENTENCE under all four models."""
    from train_bpe_transformer import bits_per_byte

    print(f'\nBPB of "{SENTENCE}"')
    doc = SENTENCE.encode("utf-8")
    for dataset in DATASETS:
        ngram_bpb = models[("ngram", dataset)].bits_per_byte([doc])
        model, tokenizer, config = models[("transformer", dataset)]
        tf_bpb = bits_per_byte(model, [SENTENCE], tokenizer, config["block_size"], device)
        print(f"  {dataset:12s} n-gram {ngram_bpb:.3f}   transformer {tf_bpb:.3f}")


def check_results_table():
    """Check the stored val_bpb values and print the demo option 2 table."""
    from bpb_harness import format_table, load_results

    results = load_results(CHECKPOINTS_DIR)
    found = {(r["model"], r["dataset"]): r["val_bpb"] for r in results}
    for key, expected in RESULTS_JSON_BPB.items():
        if found.get(key) != expected:
            raise SetupError(f"results JSON for {key} has val_bpb {found.get(key)}, expected {expected}. "
                             "Re-download with: python scripts/fetch_assets.py --force")
    print("\nComparison table (demo option 2):\n")
    print(format_table(results))


def check_reported_numbers(models, device):
    """Re-score the reported evaluation sets and compare with REPORTED_BPB to four decimals."""
    from data_utils import read_jsonl_texts
    from train_bpe_transformer import bits_per_byte

    print("Re-scoring the reported evaluation sets:")
    for (dataset, split, num_docs), (ngram_expected, tf_expected) in REPORTED_BPB.items():
        start = time.time()
        texts = read_jsonl_texts(DATA_DIR / dataset / f"{split}.jsonl", num_docs)
        if len(texts) != num_docs:
            raise SetupError(f"{dataset}/{split} has {len(texts)} docs, expected {num_docs}.")
        ngram_bpb = models[("ngram", dataset)].bits_per_byte([t.encode("utf-8") for t in texts])
        model, tokenizer, config = models[("transformer", dataset)]
        tf_bpb = bits_per_byte(model, texts, tokenizer, config["block_size"], device)
        got = (round(ngram_bpb, 4), round(tf_bpb, 4))
        ok = got == (ngram_expected, tf_expected)
        print(f"  {dataset}/{split} ({num_docs} docs): n-gram {got[0]:.4f} (reported {ngram_expected:.4f}), "
              f"transformer {got[1]:.4f} (reported {tf_expected:.4f})  "
              f"{'match' if ok else 'MISMATCH'}  [{time.time() - start:.0f}s]")
        if not ok:
            raise SetupError(f"{dataset}/{split} does not reproduce the reported numbers.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="also re-score the reported evaluation sets (about 30 s on GPU, 2 min on CPU)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="where to run the transformers (default: cuda if available)")
    args = parser.parse_args()

    start = time.time()
    try:
        device = check_torch(args.device)
        check_packages()
        check_files()
        models = load_models(device)
        score_sentence(models, device)
        check_results_table()
        if args.full:
            check_reported_numbers(models, device)
    except SetupError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
    print(f"\nPASS ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
