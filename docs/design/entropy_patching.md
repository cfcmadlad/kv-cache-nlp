# Entropy model and patching

## What exists

| Path | Purpose |
|---|---|
| `models/entropy.py` | 3.29M parameter byte model: 4 layers, width 256, 4 heads, RoPE, causal sliding window of 512 bytes, BOS as the 257th input id |
| `models/rope.py` | Rotary embeddings shared with the BLT model |
| `kvc/patching.py` | Entropy from logits, global-threshold patch starts, maximum patch length, threshold tuning, word-start statistics |
| `scripts/train_entropy.py`, `configs/entropy/entropy_*.json` | Training, resumable, windows of 1024 bytes sampled from the byte stream with document masks |
| `scripts/patch_entropy.py` | Validation BPB, threshold tuned for mean patch length 4.5, word-start statistics |
| `scripts/cache_entropy.py` | Per-byte entropies for train and validation, so that BLT training and evaluation share identical boundaries |
| `tests/test_entropy.py` | Causality, window reach, documents isolated, boundaries ignore the future |

## Patching rule

The rule is that of the BLT, a global threshold: a patch opens at byte t when the entropy of the model's prediction for byte t (computed from the bytes before t alone) exceeds the threshold; the first byte always opens a patch, and a patch longer than the maximum (16) is split. The threshold is found by bisection on validation text, so that the mean patch length comes to 4.5.

BPB counts every byte of a document. The first byte is predicted from BOS, so that no byte serves only as context, which avoids the optimism of the Part 2 figure.
