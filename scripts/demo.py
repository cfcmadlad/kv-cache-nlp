"""Interactive CLI demo for the Part 2 review.

Option 1 trains a baseline live on a 3,000-document subset by running the real training
script with small settings, so the examiner sees genuine progress output.
Option 2 prints the comparison table of the full-scale trained models (bpb_harness.py).
Option 3 loads a full-scale model, scores a sentence you type and continues it.

Usage:
    python scripts/demo.py
"""

import subprocess
import sys
from pathlib import Path

import torch

from data_utils import load_ngram, load_transformer, safe_print
from train_bpe_transformer import per_document_bpb

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
# Live demo runs write here, never to checkpoints/ itself: an early demo run once
# overwrote a real checkpoint that had the same filename.
DEMO_CHECKPOINTS_DIR = "checkpoints/demo"


def ask(prompt: str, choices: list) -> str:
    """Prompt until the user types one of choices."""
    choice_str = "/".join(choices)
    while True:
        answer = input(f"{prompt} ({choice_str}): ").strip().lower()
        if answer in choices:
            return answer
        print(f"  Please enter one of: {choice_str}")


def train_live():
    """Option 1: run the real training script on a 3,000-document subset."""
    dataset = ask("Dataset", ["tinystories", "wikitext103"])
    model = ask("Model", ["ngram", "transformer"])

    print(f"\nTraining {model} on {dataset} live (small, fast settings for the demo).")
    print(f"Output goes to {DEMO_CHECKPOINTS_DIR}/, the real checkpoints are not touched.\n")
    if model == "ngram":
        cmd = [sys.executable, "scripts/train_ngram.py", "--dataset", dataset,
               "--order", "4", "--max-train-docs", "3000", "--eval-max-docs", "300",
               "--out", DEMO_CHECKPOINTS_DIR]
    else:
        cmd = [sys.executable, "scripts/train_bpe_transformer.py", "--dataset", dataset,
               "--max-train-docs", "3000", "--eval-max-docs", "300",
               "--vocab-size", "2000", "--max-steps", "300", "--eval-interval", "50",
               "--out", DEMO_CHECKPOINTS_DIR]

    # A subprocess sharing this terminal shows exactly the progress output of a real run.
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"\n  Training failed (exit code {result.returncode}), see the error above. "
              "If data/processed/ is missing, run python scripts/fetch_assets.py")


def show_comparison():
    """Option 2: print the comparison table of the full-scale results."""
    print()
    result = subprocess.run([sys.executable, "scripts/bpb_harness.py"], cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"\n  Could not build the comparison (exit code {result.returncode}). "
              "Run python scripts/fetch_assets.py to get the trained models.")


def ngram_score_and_continue(model, text: str) -> tuple:
    """BPB of text under the n-gram model, and 120 sampled bytes that follow it."""
    doc = text.encode("utf-8")
    bpb = -model.log2_prob_doc(doc) / len(doc)
    full = model.generate(doc, max_new_bytes=120)
    return bpb, full[len(doc):].decode("utf-8", errors="replace")


def transformer_score_and_continue(model, tokenizer, config: dict, text: str, device: str) -> tuple:
    """BPB of text under the Transformer, and 60 tokens sampled after it (temperature 0.8)."""
    bpb = per_document_bpb(model, [text], tokenizer, config["block_size"], device)[0]
    ids = tokenizer.encode(text).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out_ids = model.generate(idx, max_new_tokens=60)[0].tolist()
    return bpb, tokenizer.decode(out_ids[len(ids):])


def explore_model():
    """Option 3: load a full-scale model, then score and continue sentences the user types."""
    dataset = ask("Dataset", ["tinystories", "wikitext103"])
    model_kind = ask("Model", ["ngram", "transformer"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if model_kind == "ngram":
            model = load_ngram(CHECKPOINTS_DIR, dataset)
        else:
            model, tokenizer, config = load_transformer(CHECKPOINTS_DIR, dataset, device)
    except FileNotFoundError as e:
        print(f"  {e}. Run python scripts/fetch_assets.py to get the trained models.")
        return
    except Exception as e:
        # A half-written checkpoint must not crash a live demo.
        print(f"  Could not load that checkpoint ({e}). "
              "Re-download it with python scripts/fetch_assets.py --force")
        return

    print("\nType a sentence to score it and see the model continue it. Empty line to go back.\n")
    while True:
        text = input("> ").strip()
        if not text:
            return
        try:
            if model_kind == "ngram":
                bpb, continuation = ngram_score_and_continue(model, text)
            else:
                bpb, continuation = transformer_score_and_continue(model, tokenizer, config, text, device)
            safe_print(f"  BPB for your text: {bpb:.3f}")
            safe_print(f"  Model continues with: {continuation}")
        except Exception as e:
            print(f"  Could not score that: {e}")


def main():
    while True:
        print("\n=== kv-cache-nlp baseline demo ===")
        print("1) Train a model live (quick demo run)")
        print("2) Show real baseline comparison (BPB harness)")
        print("3) Explore a trained model (score your text / see it generate)")
        print("4) Quit")
        choice = input("> ").strip()

        if choice == "1":
            train_live()
        elif choice == "2":
            show_comparison()
        elif choice == "3":
            explore_model()
        elif choice == "4":
            return
        else:
            print("  Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
