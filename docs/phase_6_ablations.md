# Phase 6: ablations, seeds and controls

Rubric: Results and error analysis (2), Novelty (3), Demo and viva (20).

## What was run

All on the cluster, one job at a time, from `scripts/run_seeds.sh`, `scripts/run_ablations.sh`, `scripts/run_policies.sh` and `scripts/run_code_eval.sh`; tables from `scripts/compare.py` (`seeds`, `ablations`, `table`, `code`).

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

Policies and budgets are identical across runs; the protected set (first patch and 8 newest) is the same for every policy. Random uses three draws averaged. `--signal own` takes the entropy of the model's own next-byte distribution from a full-cache pass; `--agg first|sum` are new scoring rules, tested against step-by-step decoding in `tests/test_cache.py`.

## What the results say

1. Keeping high-entropy patches is worse than random and than recency in every seed, every budget and both datasets, and also for fixed-stride models.
2. The own-entropy signal gives the same answer as the separate entropy model.
3. No scoring rule beats random. The sum rule, which prefers long patches, ties random on TinyStories.
4. The sink size has no measurable effect at 4,096 bytes of context.
5. Patch-level caches lose far less than byte-level caches at the same entry count. Entropy is worse than random at byte level too.
6. snap, a SnapKV-style attention rule, beats recency by 0.0001 to 0.0069 BPB in 67 of 72 comparisons (never worse); h2o is worse than recency in 60. The scores are offline approximations. With 16,384-byte chunks the entropy penalty shrinks with budget and snap no longer beats recency. On the WikiText test split high entropy is worse than random and recency in all 36 cells.
7. With 15 percent code in training the sign is unchanged on WikiText and code files but about a third of the size; keeping low-entropy patches stops beating keeping high-entropy ones, and the attention correlation with entropy stays negative in the middle and upper layers of most models (the first code-mix seed was an outlier). Prose-only models score 7 to 8 BPB on code, so they say nothing about code.

## Limits

Seed 0 of the prose-only models was trained on a laptop GPU and the others on the cluster; the code-mix models were all trained on the cluster. Seed 2 is the worst or second-worst seed for most schemes. Every patching scheme and the byte-level model have three seeds; ablations and the long-context models use one. The own-entropy variant is computed offline from a full-cache pass. The byte-level model's mean patch length of 1.0 means no local pooling, so it is a different architecture in effect, not only a different cache.

## Viva notes

Why test fixed-stride models with the same policies? If the entropy signal only failed because entropy patching makes patches uneven, fixed strides would show a positive result; they show the same sign.
Why compare the model's own entropy with the entropy model? The proposal could fail because a separate 3.3M model is a poor proxy for what the big model finds surprising. It makes no difference.
Why does a sum rule tie random? Unknown. A sum favours long patches, so it mixes a length preference with the entropy preference; I have not separated the two.
Why is the sink irrelevant here? Probably because the local layers see a 256-byte window of raw bytes and the documents are short; I did not test longer contexts, where a sink may matter.
What might change the conclusion? Untested here: a task where rare surprising spans are queried later, such as retrieval or long-range copying, rather than next-byte prediction on short documents.
Why score the attention policies offline? The tested simulation applies one mask per chunk, so attention scores cannot come from the evicted cache; I refine them once under the resulting mask. An online version would need score updates in the step API, which I have not built, so the numbers may differ.
Why does snap help and h2o not? Unknown. A guess: snap scores recent attention, which stays close to recency while correcting it, whereas accumulated attention favours old patches that were popular early. I have not tested that.
Why did the 16k models score worse than the 4k ones? Unknown. They saw four times fewer sequences per step and were not tuned; I did not test either.
