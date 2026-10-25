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
Why was online scoring built later, and does it change the result? The first tables scored attention offline (one pass, refined once under the mask) because the mask simulation cannot see the evicted cache. `BLT.forward_online` then ran the global layers patch by patch with a real cache and accumulated attention as it came; snap online is never worse than offline and beats recency in 24 of 24 comparisons, h2o online is worse than recency in 22 of 24, so the conclusions held. What remains untested is a step-decoding test for the attention policies themselves; the online loop is checked equal to step decoding for recency and random only.
Why does snap help and h2o not? Unknown. A guess: snap scores recent attention, which stays close to recency while correcting it, whereas accumulated attention favours old patches that were popular early. I have not tested that.
Why did the first 16k models score worse than the 4k ones, and what does the tuned result say? They were under-trained: batch 8 with learning rate 0.001 gave 1.5843 BPB against 1.8920 at batch 2, and a lower rate was worse at both sizes. With the tuned models the entropy penalty does not shrink (+0.016 to +0.021 against random at 64 to 512 patches). The batch-8 runs also see four times more bytes than the 4k models, so I cannot separate context length from training amount.
Does a better-trained model change the entropy result? It makes it larger: WikiText code-mix models trained ten times longer score 1.29 to 1.31 BPB and keeping high entropy costs +0.024 to +0.029 BPB against random at 64 patches in all three seeds (+0.002 at 3,000 steps). The TinyStories version overfit (37 epochs of a 26.7 MB file) and is not used.
Why can a fixed-size cache beat the full cache? The baseline resets the context every 4,096 bytes, so it forgets everything at each boundary; a recency window of 64 patches carries on across the boundary and scores 0.0008 to 0.0032 BPB better. Against a full cache over the whole 16,384-byte chunk (patch positions beyond training, 1.7201, 1.7178 and 1.8149 BPB for the three seeds) the 64-patch recency window scores 1.7180, 1.7109 and 1.8154, so it is better in two seeds and 0.0005 worse in the third; all of this is one dataset and 46 documents.
Do the models retrieve anything from the cache? No. In the planted-name test the score with the cue equals the score without it, even for the 30,000-step models, so nothing about retrieval can be concluded and the entropy rule's higher chance of keeping the name does not help.
Is the memory saving real? Measured on the GPU with real eviction: the global cache stays at 0.59 to 2.36 MB for 64 to 256 patches while a full cache reaches 137 MB at 65,536 bytes, and per-byte decoding time stays at 0.9 ms against 2.0 ms. The fixed byte-window caches (786 KB) and the entropy model (about 2.1 MB of cache) are not reduced, so at short contexts they dominate.
