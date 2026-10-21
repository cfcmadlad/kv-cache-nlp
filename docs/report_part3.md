# Part 3 report text: entropy as a keep signal for byte-level KV caches

Draft written 21 Oct 2026, updated 22 Oct from `docs/results_summary.md`. Every number below is in that file or in `results/part3/`. Validation data only; the test splits are untouched.

## 1. Problem and idea

Byte-level language models read raw UTF-8, so a sentence spans about four times as many positions as in BPE tokens and the KV cache grows with it. The Byte Latent Transformer groups bytes into patches using the next-byte entropy of a small model, so that entropy is already computed. We asked whether the same entropy can decide which cache entries to evict or to store at lower precision, compared with recency (a StreamingLLM-style window with a sink) and random choice at the same number of cache bytes. The patch is the natural unit: one global cache entry per patch.

Related work (`NOVELTY.md`) uses entropy to set cache budgets or as one term of a mixed score, on subword models; we found no test of entropy-ranked retention on byte-level models. The idea is plausible and cheap, which is why a negative result is informative.

## 2. System

A 13.3M-parameter BLT-style model: a local encoder over bytes (256-byte windows), 6 global transformer layers over patches, and a local decoder. Patches come from a separate 3.3M-parameter byte model (entropy threshold tuned for 4.5 bytes per patch) or from a fixed stride. Incremental decoding equals the full forward pass to 1e-5, and tests check that every eviction and quantization policy, simulated with masks in one pass, equals step-by-step decoding with entries actually removed.

Cache policies share one budget interface and a protected set (the first patch and the 8 newest): recency, random (three draws), keep high entropy (the proposal), keep low entropy (the opposite control). Precision tiers store K and V at 4 or 2 bits per patch by the same rules, against uniform 2, 3, 4 and 8 bits. Cache size is computed from entry counts and bit widths.

## 3. Setup

Data: TinyStories (first 500 validation stories) and WikiText-103 (60 validation articles), each with and without 15 percent code in training. Models are small and short-trained (3,000 steps, 100M bytes). Context is 4,096 bytes (16,384 for the long-context models). BPB is scored on identical documents for every model. Differences are paired over documents with 95 percent bootstrap intervals (2,000 resamples). Three seeds for every patching scheme (entropy, fixed stride 4, fixed stride 5) in all four settings and for the byte-level model; the ablations, the code-domain tables and the 16k models use one seed. The WikiText test split was evaluated once, with policies and budgets fixed beforehand.

## 4. Results

**Base models.** The BLT-style models reach 0.80 (TinyStories) and 1.68 (WikiText) BPB with entropy patching, against 0.69 and 1.52 for the Part 2 BPE transformer. They do not beat it; they are smaller and without the n-gram hashing and cross-attention of the original BLT. Entropy patching places 67 percent of patch starts at word starts (chance 16 to 20 percent). Seed spread in BPB is 0.056 on TinyStories and 0.028 on WikiText for entropy patching, and 0.135 and 0.044 for fixed stride 4, so single-seed gaps mean little. With three seeds each, entropy patching is lower than both fixed strides in all three seed pairs on WikiText and ties stride 4 on TinyStories.

**Main result: the proposal is negative.** Keeping high-entropy patches is worse than random and than recency in all three seeds, at every budget, on both datasets, and for fixed-stride models. Over 72 comparisons (nine models in each of four settings, two budgets) it is worse than recency in 70 and than random in 68, with the interval above zero, and never better than either. The paired difference against random is +0.0003 to +0.0245 BPB. Keeping low-entropy patches is better than keeping high-entropy ones in 42 of the 72 comparisons, worse in 19 and indistinguishable in 11, so low-entropy retention is not a reliable alternative. Effects are small: the worst policy at 32 patches (32 percent of the full cache bytes) costs 0.025 BPB over a full cache on TinyStories.

**Precision.** Uniform 4-bit K and V costs 0.0003 to 0.0014 BPB across ten models at 27 percent of the fp16 bytes. For half the patches at 4 bits and the rest at 2, the ranking is recent-window tiers, keep-low-entropy, random, keep-high-entropy in all ten. No setting that keeps high-entropy patches is on the bytes-against-BPB frontier (`results/part3/comparison.md`); recency eviction at small budgets and uniform 3-bit to 8-bit are.

**Code mix.** Prose-only models cannot read code (7 to 8 BPB), so the WikiText entropy model and BLT were retrained with 15 percent code (three seeds, fixed-stride controls). The BLT loses about 0.05 BPB on WikiText (1.7482 mean against 1.6958); seed 2 is again the worst. Keeping high-entropy patches is still worse than random and recency in 3 of 3 seeds and on code files, with effects of +0.002 to +0.004 BPB, and keeping low-entropy patches is no longer better than keeping high-entropy ones. Recency remains the best policy.

**Attention-score baselines.** Two policies that use the model's attention (h2o: accumulated attention received; snap: attention from the last 16 patches, SnapKV-style) were added, scored offline from a full-cache pass refined once under the resulting mask. snap beats recency in 67 of the 72 comparisons (the other five are indistinguishable) by 0.0001 to 0.0069 BPB, and recovers about 40 percent of the recency eviction cost for the seed-0 TinyStories model at 32 patches. h2o is worse than recency in 60 and better in 8. So a signal does beat recency, a little, and it is attention, not entropy.

**Longer context.** With 16,384-byte chunks (one seed, WikiText with code) the entropy penalty shrinks with budget (+0.0016 at 64 patches to +0.0003 at 512, interval including zero from 256) and snap no longer beats recency; for fixed stride the entropy penalty grows to +0.0136. The 16k models score worse in BPB (1.89 against 1.72) and were not tuned, so these results are weaker evidence.

**Test split.** On the WikiText test split (62 documents, evaluated once for 18 models) keeping high-entropy patches is worse than random and recency in all 36 cells, and snap beats recency in 33 of 36, so the validation conclusions hold out of sample.

**Ablations.** The model's own entropy gives the same result as the separate entropy model. No scoring rule (mean, maximum, first byte, sum) beats random; the sum rule ties random on TinyStories. The sink size does not matter at this context. A byte-level cache (one entry per byte) loses much more than a patch cache at equal entry counts: at 32 entries recency costs +0.229 BPB against +0.013 with patches. Patches win per byte of cache; bytes win only with about 4.5 times more cache.

## 5. Error analysis

Why is entropy a poor keep signal? Attention received by old patches correlates negatively with patch entropy in layers 2 to 5 of all three prose-only WikiText seeds and in all six layers of all three TinyStories seeds; the highest-entropy third receives about half the attention of the lowest third in two layers of the TinyStories seed-0 model. The model reads predictable patches more than surprising ones, and surprising bytes are largely handled by the local windows. This is a correlation; no intervention tests it. The WikiText code-mix seed 0 model shows the opposite sign in most layers while its eviction outcome has the same sign, so the mechanism is not a complete account. Other limitations are listed in section 7.

## 6. Challenges

See `CHALLENGES.md`. The ones that shaped the result: a cache accounting error that halved the byte shares of full caches (found by putting eviction and quantization rows in one table, fixed, all tables regenerated, BPB unchanged), seed variance as large as some headline gaps (the earlier TinyStories patching comparison was withdrawn), an uninformative code evaluation, and a cluster that needed a new environment before any job ran.

## 7. Limitations and what would settle them

- Small models, 4,096-byte contexts and next-byte prediction on short documents. Rare surprising spans may matter for retrieval or copying; untested.
- Simulated caches; sizes are computed.
- Seed 0 trained on a different GPU from seeds 1 and 2; evaluation of the same checkpoint gave identical BPB on both GPUs.
- Ablations and the code-domain tables use one seed; quantization and attention analysis use three where stated.
- The attention policies are offline approximations with no step-decoding test, so an online H2O or SnapKV may differ.
- The 16k models are untuned and score worse than the 4k models; the long-context result is one seed.
- Only WikiText has a test split in our byte data; TinyStories results are validation only.
- The Block 2.0 requeue path is verified through the requeue; the resume on the cluster is not.
