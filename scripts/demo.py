"""Interactive CLI demo -- train a baseline live, show real results, explore a model.

For the Part 2 demo/viva: option 1 runs actual training (small/fast settings) with the
same live progress output as a real run, satisfying "training shown during the demo".
Options 2-3 show the real, full-scale results already produced by prepare_data.py /
train_ngram.py / train_bpe_transformer.py / bpb_harness.py.

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


def ask(prompt: str, choices: list) -> str:
    choice_str = "/".join(choices)
    while True:
        answer = input(f"{prompt} ({choice_str}): ").strip().lower()
        if answer in choices:
            return answer
        print(f"  Please enter one of: {choice_str}")


DEMO_CHECKPOINTS_DIR = "checkpoints/demo"  # separate from checkpoints/ -- a demo run must
                                            # never overwrite the real results other scripts read


def train_live():
    dataset = ask("Dataset", ["tinystories", "wikitext103"])
    model = ask("Model", ["ngram", "transformer"])

    print(f"\nTraining {model} on {dataset} live (small/fast settings for the demo).")
    print(f"Saved separately under {DEMO_CHECKPOINTS_DIR}/ -- this won't touch the real results.\n")
    if model == "ngram":
        cmd = [sys.executable, "scripts/train_ngram.py", "--dataset", dataset,
               "--order", "4", "--max-train-docs", "3000", "--eval-max-docs", "300",
               "--out", DEMO_CHECKPOINTS_DIR]
    else:
        cmd = [sys.executable, "scripts/train_bpe_transformer.py", "--dataset", dataset,
               "--max-train-docs", "3000", "--eval-max-docs", "300",
               "--vocab-size", "2000", "--max-steps", "300", "--eval-interval", "50",
               "--out", DEMO_CHECKPOINTS_DIR]

    # Runs as a real subprocess (inherits this terminal) so the same tqdm progress bars
    # and print statements a real training run produces are what the demo shows live.
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"\n  Training failed (exit code {result.returncode}) -- see the traceback above."
              f" Has `prepare_data.py --dataset {dataset}` been run yet?")


def show_comparison():
    print()
    result = subprocess.run([sys.executable, "scripts/bpb_harness.py"], cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"\n  Couldn't build the comparison (exit code {result.returncode}) -- "
              "train both baselines first (the real scripts, not the quick demo mode).")


def explore_model():
    dataset = ask("Dataset", ["tinystories", "wikitext103"])
    model_kind = ask("Model", ["ngram", "transformer"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if model_kind == "ngram":
            model = load_ngram(CHECKPOINTS_DIR, dataset)
        else:
            model, tokenizer, config = load_transformer(CHECKPOINTS_DIR, dataset, device)
    except FileNotFoundError as e:
        print(f"  {e} -- train this model first (option 1, or the real training scripts).")
        return
    except Exception as e:
        # A checkpoint can exist but be corrupt/half-written (e.g. an earlier run was
        # interrupted) -- this must degrade gracefully too, not crash a live demo.
        print(f"  Couldn't load that checkpoint ({e}) -- try retraining it.")
        return

    print("\nType a sentence to score it and see the model continue it. Empty line to go back.\n")
    while True:
        text = input("> ").strip()
        if not text:
            return
        try:
            if model_kind == "ngram":
                doc = text.encode("utf-8")
                bpb = -model.log2_prob_doc(doc) / len(doc)
                full = model.generate(doc, max_new_bytes=120)
                continuation = full[len(doc):].decode("utf-8", errors="replace")
            else:
                bpb = per_document_bpb(model, [text], tokenizer, config["block_size"], device)[0]
                ids = tokenizer.encode(text).ids
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                out_ids = model.generate(idx, max_new_tokens=60)[0].tolist()
                continuation = tokenizer.decode(out_ids[len(ids):])
            safe_print(f"  BPB for your text: {bpb:.3f}")
            safe_print(f"  Model continues with: {continuation}")
        except Exception as e:
            print(f"  Couldn't score that: {e}")


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
