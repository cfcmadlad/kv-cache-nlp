# Part 3 report text: entropy as a keep signal for byte-level KV caches

Draft written 21 Oct 2026 from `docs/results_summary.md`. Every number below is in that file or in `results/part3/`. Validation data only; the test splits are untouched.

## 1. Problem and idea

Byte-level language models read raw UTF-8, so a sentence spans about four times as many positions as in BPE tokens and the KV cache grows with it. The Byte Latent Transformer groups bytes into patches using the next-byte entropy of a small model, so that entropy is already computed. We asked whether the same entropy can decide which cache entries to evict or to store at lower precision, compared with recency (a StreamingLLM-style window with a sink) and random choice at the same number of cache bytes. The patch is the natural unit: one global cache entry per patch.

Related work (`NOVELTY.md`) uses entropy to set cache budgets or as one term of a mixed score, on subword models; we found no test of entropy-ranked retention on byte-level models. The idea is plausible and cheap, which is why a negative result is informative.

## 2. System

A 13.3M-parameter BLT-style model: a local encoder over bytes (256-byte windows), 6 global transformer layers over patches, and a local decoder. Patches come from a separate 3.3M-parameter byte model (entropy threshold tuned for 4.5 bytes per patch) or from a fixed stride. Incremental decoding equals the full forward pass to 1e-5, and tests check that every eviction and quantization policy, simulated with masks in one pass, equals step-by-step decoding with entries actually removed.

Cache policies share one budget interface and a protected set (the first patch and the 8 newest): recency, random (three draws), keep high entropy (the proposal), keep low entropy (the opposite control). Precision tiers store K and V at 4 or 2 bits per patch by the same rules, against uniform 2, 3, 4 and 8 bits. Cache size is computed from entry counts and bit widths.

## 3. Setup

Data: TinyStories (first 500 validation stories) and WikiText-103 (60 validation articles). Models are small and short-trained (3,000 steps, 100M bytes). Context is 4,096 bytes. BPB is scored on identical documents for every model. Differences are paired over documents with 95 percent bootstrap intervals (2,000 resamples). Three seeds for the entropy-patched models (prose-only and code-mix WikiText), one seed for the fixed-stride controls, the byte-level model and all ablations.

## 4. Results

**Base models.** The BLT-style models reach 0.80 (TinyStories) and 1.68 (WikiText) BPB with entropy patching, against 0.69 and 1.52 for the Part 2 BPE transformer. They do not beat it; they are smaller and without the n-gram hashing and cross-attention of the original BLT. Entropy patching places 67 percent of patch starts at word starts (chance 16 to 20 percent). Seed spread in BPB is 0.057 on TinyStories and 0.028 on WikiText, so entropy patching beats fixed stride on WikiText (by 0.065 to 0.081) but not on TinyStories.

**Main result: the proposal is negative.** Keeping high-entropy patches is worse than random and than recency in all three seeds, at every budget, on both datasets, and for fixed-stride models. The paired difference against random is +0.005 to +0.025 BPB; every interval excludes zero. Keeping low-entropy patches is about as good as recency on TinyStories (within 0.002 BPB at 32 and 64 patches) and between random and keep-high-entropy on WikiText. Effects are small: the worst policy at 32 patches (32 percent of the full cache bytes) costs 0.025 BPB over a full cache on TinyStories.

**Precision.** Uniform 4-bit K and V costs 0.0010 BPB on TinyStories and 0.0004 on WikiText at 27 percent of the fp16 bytes. Entropy-guided tiers lose to random and to recency tiers. No setting that keeps high-entropy patches is on the bytes-against-BPB frontier (`results/part3/comparison.md`); recency eviction at small budgets and uniform 3-bit to 8-bit are.

**Code mix.** Prose-only models cannot read code (7 to 8 BPB), so the WikiText entropy model and BLT were retrained with 15 percent code (three seeds, fixed-stride controls). The BLT loses about 0.05 BPB on WikiText (1.7482 mean against 1.6958); seed 2 is again the worst. Keeping high-entropy patches is still worse than random and recency in 3 of 3 seeds and on code files, with effects of +0.002 to +0.004 BPB, and keeping low-entropy patches is no longer better than keeping high-entropy ones. Recency remains the best policy.

**Ablations.** The model's own entropy gives the same result as the separate entropy model. No scoring rule (mean, maximum, first byte, sum) beats random; the sum rule ties random on TinyStories. The sink size does not matter at this context. A byte-level cache (one entry per byte) loses much more than a patch cache at equal entry counts: at 32 entries recency costs +0.229 BPB against +0.013 with patches. Patches win per byte of cache; bytes win only with about 4.5 times more cache.

## 5. Error analysis

Why is entropy a poor keep signal? Attention received by old patches correlates negatively with patch entropy in all six global layers on TinyStories (-0.04 to -0.38) and five of six on WikiText; the highest-entropy third receives about half the attention of the lowest third in two layers. The model reads predictable patches more than surprising ones, and surprising bytes are largely handled by the local windows. This is one seed and a correlation; no intervention tests it. The code-mix models complicate the story: the attention correlation with entropy is small and positive in five of six layers (-0.34 in layer 1), so the mechanism above is not stable even though the outcome is. Other limitations are listed in section 7.

## 6. Challenges

See `CHALLENGES.md`. The ones that shaped the result: a cache accounting error that halved the byte shares of full caches (found by putting eviction and quantization rows in one table, fixed, all tables regenerated, BPB unchanged), seed variance as large as some headline gaps (the earlier TinyStories patching comparison was withdrawn), an uninformative code evaluation, and a cluster that needed a new environment before any job ran.

## 7. Limitations and what would settle them

- Small models, 4,096-byte contexts and next-byte prediction on short documents. Rare surprising spans may matter for retrieval or copying; untested.
- Simulated caches; sizes are computed.
- Seed 0 trained on a different GPU from seeds 1 and 2; evaluation of the same checkpoint gave identical BPB on both GPUs.
- Ablations, quantization and attention analysis use one seed.
- Attention-score baselines (H2O, SnapKV) are not implemented, so we cannot say whether any signal beats recency; only that entropy does not.
- TinyStories was not retrained with code. A longer context and the test split are the next steps.
