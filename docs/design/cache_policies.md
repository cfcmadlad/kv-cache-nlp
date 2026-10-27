# Patch-level cache policies

## What is compressed

The keys and values of the global model, one entry per patch per layer. The caches of the local encoder, the local decoder and the entropy model are fixed windows and are reckoned as overhead. One global entry costs 2 (K and V) x 6 layers x 384 x 2 bytes (fp16) = 9,216 bytes.

## Policies (`kvc/cache.py`)

All policies guard the same protected set: the first `sink` patches (by default 1, the BOS patch) and the `recent` newest patches (by default 8). When, after an insertion, the cache holds more than `budget` patches, one unprotected patch is put to death. The policies differ solely in the score that picks the victim (the lowest score goes):

| Policy | Score | Role |
|---|---|---|
| recency | patch index | StreamingLLM-style sliding window with a sink |
| random | uniform random, seeded, 3 seeds averaged | control |
| entropy | mean (or max) next-byte entropy of the bytes in the patch | the proposal: keep high-entropy patches |
| entropy_inv | negative of the above | opposite control: keep low-entropy patches |
| h2o, snap | attention received so far; attention received from the last 16 patches | attention-guided rules (H2O and SnapKV-style), scored offline and online |

The entropy comes from the separate 3.3M entropy model that patching already requires, so that the signal costs nothing at decode time.

## How evaluation works

Eviction is simulated by a visibility mask over patches in one parallel forward pass: when patch i is inserted, row i of the mask marks the entries still in the cache, and an evicted entry never returns. `tests/test_cache.py` checks, to 1e-5 and for every policy and both entropy aggregations, that this equals real step-by-step decoding in which entries are removed from the cache. `scripts/eval_policies.py` writes the per-document negative log-likelihood for every policy and budget, so that bootstrap confidence intervals (`kvc/stats.py`, resampling documents, paired for differences) may be computed without rerunning anything.

Evaluation chunks are 4096 bytes (the training context), about 900 patches, so that budgets of 16 to 512 patches span 2 to 57 percent of a full chunk.

## Results

To keep the high-entropy patches is worse than random and worse than recency at every budget on both datasets; recency is the best of the four. The details, and the attention analysis that explains the verdict, are in `docs/results_summary.md`. Budgets below 10 patches are impossible, a sink of 1 and a protected window of 8 already filling them.

## What the offline tables do not settle

The attention scores of H2O and SnapKV in `scripts/eval_policies.py` are offline approximations. The online versions, which choose the victim before the new patch is attended to, live in `BLT.forward_online` and are reported in `results/part3/online_policies.md`. The x-axis of the result curves is the mean number of cache entries times 9,216 bytes.
