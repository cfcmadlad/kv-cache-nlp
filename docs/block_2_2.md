# Block 2.2: entropy model and patching

Rubric: Novelty (3), Results and error analysis (2), Demo and viva (20).

## What exists

| Path | Purpose |
|---|---|
| `models/entropy.py` | 3.29M parameter byte model: 4 layers, width 256, 4 heads, RoPE, causal sliding window of 512 bytes, BOS as the 257th input id |
| `models/rope.py` | Rotary embeddings shared with the BLT model |
| `kvc/patching.py` | Entropy from logits, global-threshold patch starts, maximum patch length, threshold tuning, word-start statistics |
| `scripts/train_entropy.py`, `configs/entropy_*.json` | Training, resumable, windows of 1024 bytes sampled from the byte stream with document masks |
| `scripts/patch_entropy.py` | Validation BPB, threshold tuned for mean patch length 4.5, word-start statistics |
| `scripts/cache_entropy.py` | Per-byte entropies for train and validation, so BLT training and evaluation use identical boundaries |
| `tests/test_entropy.py` | Causality, window reach, documents isolated, boundaries ignore the future |

## Patching rule

Follows BLT's global threshold: a patch starts at byte t when the entropy of the model's prediction for byte t (computed from bytes before t only) is above a threshold, the first byte always starts a patch, and patches longer than a maximum (16) are split. The threshold is found by bisection on validation text so the mean patch length is 4.5.

BPB counts every byte of a document: the first byte is predicted from BOS, so nothing is context only (this avoids the Part 2 optimism).

## Viva notes

Why a sliding window? The entropy model is a cache cost too; a 512-byte window bounds it.
Why must boundaries ignore the future? At generation time the next byte does not exist yet, and the patch decision has to be made before it is read. The causality test changes later bytes and checks that earlier boundaries stay the same.
Why tune the threshold on validation? Mean patch length is a property of the text distribution, not of the loss; only a single scalar is fitted.
