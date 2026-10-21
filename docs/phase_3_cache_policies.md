# Phase 3: patch-level cache policies

Rubric: Novelty (3), Improvement (3), Results and error analysis (2), Demo and viva (20).

## What is compressed

The global model's keys and values, one entry per patch per layer. Local encoder, local decoder and entropy-model caches are fixed windows and counted as overhead. One global cache entry costs 2 (K and V) x 6 layers x 384 x 2 bytes (fp16) = 9,216 bytes.

## Policies (`kvc/cache.py`)

All policies share the same protected set: the first `sink` patches (default 1, the BOS patch) and the `recent` newest patches (default 8). When the cache holds more than `budget` patches after an insertion, one unprotected patch is evicted. Policies differ only in the score used to pick the victim (the lowest score goes):

| Policy | Score | Role |
|---|---|---|
| recency | patch index | StreamingLLM-style sliding window with a sink |
| random | uniform random, seeded, 3 seeds averaged | control |
| entropy | mean (or max) next-byte entropy of the bytes in the patch | the proposal: keep high-entropy patches |
| entropy_inv | negative of the above | opposite control: keep low-entropy patches |

Entropy comes from the separate 3.3M entropy model (already needed for patching), so the signal costs nothing at decode time.

## How evaluation works

Eviction is simulated with a visibility mask over patches in one parallel forward pass: when patch i is inserted, the mask row i marks the entries still in the cache, and evicted entries never return. `tests/test_cache.py` checks that this equals real step-by-step decoding with entries deleted from the cache, to 1e-5, for every policy and both entropy aggregations. `scripts/eval_policies.py` writes per-document negative log-likelihoods for every policy and budget so bootstrap confidence intervals (`kvc/stats.py`, resampling documents, paired for differences) can be computed without rerunning.

Evaluation chunks are 4096 bytes (the training context), about 900 patches, so budgets from 16 to 512 patches cover 2 percent to 57 percent of a full chunk.

## Results

Keeping high-entropy patches is worse than random and than recency at every budget on both datasets; recency is the best. Details and the attention analysis that explains it: `docs/results_summary.md`. Budgets below 10 patches are impossible with a sink of 1 and a protected recent window of 8.

## What is not covered

Attention-score methods (H2O, SnapKV) are not implemented. They need per-layer attention scores and would be a separate baseline. The x-axis of the result curves is mean cache entries times 9,216 bytes.

## Viva notes

Why a protected recent window for all policies? Without it, random and entropy would sometimes drop the byte context needed right now, and the comparison would test the window rather than the signal.
Why include entropy_inv? If keeping low-entropy patches did equally well, the entropy signal would not be doing real work.
Why mask instead of real eviction for evaluation? Same numbers, tested equal, and the whole chunk runs in one pass instead of thousands of single-byte steps.
