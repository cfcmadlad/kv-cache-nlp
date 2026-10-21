# Results summary (validation only, 21 Oct 2026)

All numbers are bits per byte (BPB) on matched validation documents: TinyStories first 500 documents (405,231 bytes) and WikiText-103 all 60 documents (1,129,194 bytes). The test splits have not been touched. One training seed per model. Intervals are 95 percent bootstrap intervals over documents. Regenerate any table with the command in its file header; the raw tables are in `results/part3/` and `results/phase2.md`.

## Gate table (`python scripts/results.py --phase 2`)

| Model | TinyStories | WikiText | Bytes per patch or token | Global KV bytes per 1000 input bytes |
|---|---|---|---|---|
| Kneser-Ney byte 4-gram | 1.8085 | 2.4103 | n/a | n/a |
| BPE transformer 13.8M, reported | 0.6939 | 1.5200 | 4.1 / 4.0 | 2.22 MB / 2.31 MB |
| BPE transformer, corrected | 0.6978 | 1.5257 | same | same |
| Entropy model 3.3M | 0.8438 | 1.6983 | 4.5 | n/a |
| BLT 13.3M, fixed stride 4 | 0.7726 | 1.7600 | 4.0 | 2.31 MB |
| BLT 13.3M, entropy patching | 0.7980 | 1.6801 | 4.5 | 2.05 MB |

Part 2 rows reproduce exactly. The BLT-style models do not beat the BPE transformer; this is a smaller, shorter-trained model (3,000 steps, 100M bytes seen) without the n-gram hashing or cross-attention of the original BLT.

Entropy patching versus fixed stride: worse on TinyStories (0.7980 against 0.7726, 3.3 percent) and better on WikiText (1.6801 against 1.7600, 4.5 percent), with 11 percent fewer global cache entries per byte in both. The 26 Oct fallback rule (more than 10 percent worse at equal steps) is not triggered, so entropy patching stays the main model. The fixed stride model has shorter patches (4.0 against 4.5), so the comparison is not at equal patch length.

## Entropy model checks

Patch starts at word starts: 67.8 percent on TinyStories (chance 19.9 percent) and 66.7 percent on WikiText (chance 16.3 percent). Patch starts that land on a space byte: 2.8 and 1.3 percent. Mean patch length 4.5 by construction (threshold tuned on validation: 1.09 nats TinyStories, 2.17 nats WikiText).

## Eviction at a fixed budget (global cache patches kept)

BPB, entropy-patched BLT. Full cache: TinyStories 0.7980, WikiText 1.6801. Protected for every policy: the first patch and the 8 newest.

| Dataset | Budget (patches, share of full cache bytes) | recency | random | keep low entropy | keep high entropy |
|---|---|---|---|---|---|
| TinyStories | 32 (16%) | 0.8078 | 0.8115 | 0.8089 | 0.8231 |
| TinyStories | 64 (28%) | 0.8016 | 0.8033 | 0.8026 | 0.8109 |
| WikiText | 64 (7%) | 1.6836 | 1.6860 | 1.6901 | 1.6922 |
| WikiText | 128 (14%) | 1.6814 | 1.6833 | 1.6864 | 1.6885 |

Keeping high-entropy patches (the proposal) is worse than random and than recency at every budget on both datasets; the paired differences are small (at most 0.015 BPB) and every interval against random and recency excludes zero. Against random and recency the same holds for the fixed-stride models; with maximum instead of mean aggregation entropy moves close to random but is still not better (TinyStories budget 64: 0.8039 against 0.8033 random and 0.8016 recency). Files: `results/part3/policies_*`.

## Precision tiers (global K and V, high 4 bits and low 2 bits, half of the patches at the high width, about 20 percent of fp16 bytes)

| Dataset | recency | random | keep low entropy | keep high entropy | uniform 3-bit | uniform 4-bit (27%) |
|---|---|---|---|---|---|---|
| TinyStories (fp16 0.7980) | 0.8006 | 0.8336 | 0.8258 | 0.8398 | 0.8028 | 0.7990 |
| WikiText (fp16 1.6801) | 1.6815 | 1.6937 | 1.6902 | 1.6972 | 1.6820 | 1.6805 |

Uniform 4-bit is within 0.001 BPB of fp16 at 27 percent of the bytes. Entropy-guided tiers lose to random and recency again.

## Why entropy does not work as a keep signal here

Attention received by patches at least 8 patches old, by entropy third (`results/part3/attention_*.md`). On TinyStories the rank correlation between patch entropy and attention is negative in all six global layers (-0.04 to -0.38), and layers 1 and 2 give the highest-entropy third about half the attention of the lowest third. WikiText shows the same sign in five of six layers (-0.06 to -0.26) and +0.02 in layer 1. The model reads predictable patches more than surprising ones. Rare, high-entropy bytes are mostly resolved locally by the 256-byte local windows and are not what later bytes look back for.

## Caveats

- One seed per model; the planned three seeds for the entropy-patched BLT are not run.
- WikiText models were trained without the planned 15 percent code mix (the code dataset is gated).
- Evaluation chunks are 4,096 bytes (about 900 patches), so budgets and gains are specific to that context.
- Eviction and quantization are simulated by masks and rounding, checked equal to step-by-step decoding in tests; cache bytes are computed, not measured.
- The Block 2.0 cluster path (signal, requeue, cu128 venv) is untested.
