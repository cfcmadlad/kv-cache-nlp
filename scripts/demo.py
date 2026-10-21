import subprocess
import sys
from pathlib import Path

import torch

from data_utils import REPO_ROOT, load_ngram, load_transformer, read_jsonl_texts, safe_print
from train_bpe_transformer import per_document_bpb

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DEMO_CHECKPOINTS_DIR = "checkpoints/demo"
ASSET_ZIP_NAME = "kv-cache-nlp-part2-assets.zip"
REQUIRED_FILES = [
    "checkpoints/ngram_tinystories_order4.pkl",
    "checkpoints/ngram_wikitext103_order4.pkl",
    "checkpoints/transformer_tinystories_L6H6D384V8000.pt",
    "checkpoints/transformer_wikitext103_L6H6D384V8000.pt",
    "checkpoints/bpe_tinystories_v8000_dall_tokenizer.json",
    "checkpoints/bpe_wikitext103_v8000_dall_tokenizer.json",
    "data/processed/tinystories/train.jsonl",
    "data/processed/tinystories/val.jsonl",
    "data/processed/wikitext103/train.jsonl",
    "data/processed/wikitext103/val.jsonl",
]


PART3_ZIP_NAME = "kv-cache-nlp-part3-assets.zip"
PART3_FILES = [f"runs/{r}/model.pt" for r in ("entropy_tinystories", "blt_tinystories_entropy", "entropy_wikitext103", "blt_wikitext103_entropy")]


def run(args: list) -> int:
    sys.stdout.flush()
    return subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=False).returncode


def get_assets(required: list, zip_name: str, fetch_cmd: list, size: str) -> bool:
    if all((REPO_ROOT / f).exists() for f in required):
        return True
    print("First run: fetching the trained models and data.")
    for folder in (REPO_ROOT, REPO_ROOT.parent, REPO_ROOT.parent.parent, Path.home() / "Downloads"):
        local_zip = folder / zip_name
        if local_zip.is_file():
            print(f"Using the local copy {local_zip}")
            if run([*fetch_cmd, "--zip", str(local_zip)]) == 0:
                return True
            break
    print(f"Downloading from GitHub ({size}, one time only).")
    return run(fetch_cmd) == 0


def ensure_assets() -> bool:
    return get_assets(REQUIRED_FILES, ASSET_ZIP_NAME, ["scripts/fetch_assets.py"], "138 MB")


def ensure_part3_assets() -> bool:
    return get_assets(PART3_FILES, PART3_ZIP_NAME, ["scripts/part3_assets.py", "fetch"], "124 MB")


def read_line(prompt: str) -> str:
    return input(prompt).replace("﻿", "").strip()


def ask(prompt: str, choices: list) -> str:
    while (answer := read_line(f"{prompt} ({'/'.join(choices)}): ").lower()) not in choices:
        print(f"  Please enter one of: {'/'.join(choices)}")
    return answer


def train_live():
    dataset = ask("Dataset", ["tinystories", "wikitext103"])
    model = ask("Model", ["ngram", "transformer"])
    print(f"\nTraining {model} on {dataset} (demo settings, output in {DEMO_CHECKPOINTS_DIR}/).\n")
    common = ["--dataset", dataset, "--max-train-docs", "3000", "--eval-max-docs", "300", "--out", DEMO_CHECKPOINTS_DIR]
    if model == "ngram":
        cmd = ["scripts/train_ngram.py", "--order", "4", *common]
    else:
        cmd = ["scripts/train_bpe_transformer.py", "--vocab-size", "2000", "--max-steps", "300",
               "--eval-interval", "50", *common]
    if (code := run(cmd)) != 0:
        print(f"\n  Training failed (exit code {code}).")


def explore_model():
    dataset = ask("Dataset", ["tinystories", "wikitext103"])
    kind = ask("Model", ["ngram", "transformer"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if kind == "ngram":
            model = load_ngram(CHECKPOINTS_DIR, dataset)
        else:
            model, tokenizer, config = load_transformer(CHECKPOINTS_DIR, dataset, device)
    except Exception as e:
        print(f"  Could not load the checkpoint ({e}). Run: python scripts/fetch_assets.py --force")
        return
    print("\nType a sentence to score and continue it. Empty line to go back.\n")
    while text := read_line("> "):
        try:
            if kind == "ngram":
                doc = text.encode("utf-8")
                bpb = -model.log2_prob_doc(doc) / len(doc)
                continuation = model.generate(doc, 120)[len(doc):].decode("utf-8", errors="replace")
            else:
                bpb = per_document_bpb(model, [text], tokenizer, config["block_size"], device)[0]
                ids = tokenizer.encode(text).ids
                out = model.generate(torch.tensor([ids]), max_new_tokens=60)[0].tolist()
                continuation = tokenizer.decode(out[len(ids):])
            safe_print(f"  BPB: {bpb:.3f}\n  Continuation: {continuation}")
        except Exception as e:
            print(f"  Could not score that: {e}")


def compress_cache():
    if not ensure_part3_assets():
        return
    from cache_demo import MAX_BYTES, analyse, load_pair
    dataset = ask("Dataset", ["tinystories", "wikitext103"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, entropy, cfg = load_pair(dataset, device)
    samples = read_jsonl_texts(REPO_ROOT / "data" / "processed" / dataset / "val.jsonl", 20)
    print(f"\nType or paste one line of text (up to {MAX_BYTES} bytes), or press Enter to use a validation document.\n")
    text = read_line("> ") or max(samples[:5], key=len)
    budget = int(read_line("Global cache budget in patches (default 32): ") or 32)
    result = analyse(text, model, entropy, cfg, budget, device)
    safe_print(f"\n{result['bytes']} bytes in {result['patches']} patches ({result['mean_patch_len']:.1f} bytes per patch). "
               f"Patch boundaries:\n{result['marked'][:600]}\n")
    print(f"{'Cache setting':<36}{'BPB':>8}{'Mean KV cache':>16}")
    for name, bpb, size in result["rows"]:
        print(f"{name:<36}{bpb:>8.3f}{size / 1000:>13.1f} KB")


def show_part3_results():
    path = REPO_ROOT / "docs" / "results_summary.md"
    safe_print(path.read_text(encoding="utf-8") if path.exists() else "docs/results_summary.md is missing.")


def main():
    if not ensure_assets():
        sys.exit("\nCould not get the trained models and data; see the message above.")
    actions = {"1": train_live, "2": lambda: run(["scripts/bpb_harness.py"]), "3": explore_model, "4": compress_cache, "5": show_part3_results}
    while True:
        print("\n=== kv-cache-nlp demo ===\n1) Train a model live\n2) Baseline comparison table\n"
              "3) Score and continue your text\n4) Compress the KV cache of a text (Part 3)\n5) Part 3 results\n6) Quit")
        choice = read_line("> ")
        if choice == "6":
            return
        actions.get(choice, lambda: print("  Please enter a number from 1 to 6."))()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
