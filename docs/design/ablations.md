# Ablations, seeds and controls

## What was run

All of it was done on the cluster, one job at a time, by `scripts/campaign/run_seeds.sh`, `scripts/campaign/run_ablations.sh`, `scripts/campaign/run_policies.sh` and `scripts/campaign/run_code_eval.sh`; tables from `scripts/compare.py` (`seeds`, `ablations`, `table`, `code`).

| Question | Run | Table |
|---|---|---|
| Does the sign hold across training seeds? | entropy-patched models, seeds 0 to 2, both datasets | `results/part3/seeds.md` |
| Is it specific to entropy patching? | fixed-stride 4 and 5 models with the same policies | `results/part3/seeds.md` |
| Does the signal source matter? | separate entropy model against the model's own output entropy | `results/part3/ablations.md` |
| Does the scoring rule matter? | mean, maximum, first byte, sum over the patch | `results/part3/ablations.md` |
| Does the sink matter? | sink 0, 1, 4 | `results/part3/ablations.md` |
| Patches or bytes? | a stride-1 model (one global entry per byte) with the same policies at the same entry counts | `results/part3/ablations.md` |
| Does an attention-based signal beat recency? | h2o and snap (SnapKV-style) policies on every model, offline-scored | `results/part3/attention_policies.md` |
| Does it hold with a longer context? | code-mix WikiText models trained and evaluated on 16,384-byte chunks, budgets 64 to 512 | `results/part3/long_context.md` |
| Does it hold on the test split? | 18 WikiText models, once, policies and budgets fixed on validation | `results/part3/test_split.md` |
| Does it transfer to code? | WikiText and TinyStories models retrained with 15 percent code (entropy model, 3 seeds, fixed strides 4 and 5 with 3 seeds), full set of ablations and tables repeated; prose-only models on code as a negative control | `results/part3/code_domain.md`, the `wikitext103mix` sections of the other tables |

Policies and budgets are the same in every run, and so is the protected set (the first patch and the 8 newest) for every policy. Random uses three draws, averaged. `--signal own` takes the entropy of the model's own next-byte distribution from a full-cache pass; `--agg first|sum` are fresh scoring rules, tested against step-by-step decoding in `tests/test_cache.py`.

## What the results say

1. To keep the high-entropy patches is worse than random and worse than recency in every seed, at every budget and on both datasets, and the fixed-stride models bear the same sign.
2. The model's own entropy returns the same verdict as the separate entropy model.
3. No scoring rule beats random. The sum rule, which favours long patches, draws level with random on TinyStories.
4. The size of the sink has no measurable effect at 4,096 bytes of context.
5. Patch-level caches lose far less than byte-level caches at the same number of entries; and entropy is worse than random at byte level also.
6. snap, a SnapKV-style attention rule, beats recency by 0.0001 to 0.0069 BPB in 67 of 72 comparisons (never worse); h2o is worse than recency in 60. The scores are offline approximations. With chunks of 16,384 bytes the entropy penalty dwindles as the budget grows, and snap no longer outdoes recency. On the WikiText test split high entropy is worse than random and recency in all 36 cells.
7. With 15 percent of code in training the sign is unchanged on WikiText and on code files, though the effect shrinks to about a third; keeping low-entropy patches ceases to beat keeping high-entropy ones, and the correlation of attention with entropy remains negative in the middle and upper layers of most models (the first code-mix seed was an outlier). Prose-only models score 7 to 8 BPB on code and therefore testify nothing about it.

## Limits

Seed 0 of the prose-only models was trained on a laptop GPU and the others on the cluster; the code-mix models were all trained on the cluster. Seed 2 is the worst or second-worst seed for most schemes. Every patching scheme and the byte-level model have three seeds; the ablations and the long-context models have one. The own-entropy variant is computed offline from a full-cache pass. A mean patch length of 1.0 in the byte-level model means that nothing is pooled locally, so that it is a different architecture in effect, and not merely a different cache.
