# Novelty and related work

## Research question

In a byte-level model, the entropy of the next byte is already available, because it determines where each patch ends, and a patch is a natural unit for a cache entry. We examined whether the same entropy can also determine which entries a bounded cache should retain, or at what precision they should be stored. It was compared with random, recency, attention-based and quantization alternatives at equal cache size in bytes, using bits per byte (BPB) against bytes, paired bootstrap intervals, three seeds per setting and a test split that was used once.

## Contributions

1. **A controlled evaluation of entropy-ranked retention on a byte-level patch model.** We found no earlier evaluation of this combination (a byte-level model, a patch-granular cache, entropy as the sole retention signal, and random and recency controls at equal bytes). The result is negative: keeping high-entropy patches is worse than random in 68 and worse than recency in 70 of 72 comparisons, and the penalty increases with training (`docs/results_summary.md`, `results/part3/seeds.md`).
2. **An analysis of what the entropy rule retains.** The mean-entropy rule retains almost exclusively patches of one or two bytes (98.0 and 99.7 percent of the retained patches), which helps on a small number of surprising bytes and hurts on the many easy ones (`docs/error_analysis.md`).
3. **An online, tested implementation of H2O and SnapKV-style policies over patch caches** (`BLT.forward_online`, `tests/test_blt.py`). The SnapKV-style rule outperforms recency in 24 of 24 online comparisons (`results/part3/online_policies.md`).
4. **Measured bounded-memory streaming for a byte-level model**, with entries actually evicted. Memory and time per byte remain constant, and BPB is lower than with a context reset on long WikiText documents and on 20 PG-19 books (`results/part3/stream_16k.md`, `pg19_stream.md`, `deploy_bench.md`).
5. **An analytic account of compute** that shows where the BLT is economical (cache memory, and FLOPs relative to a byte-level Transformer) and where it is not (FLOPs relative to a BPE Transformer) (`results/part3/flops.md`).

## What is not new

The BLT, StreamingLLM, H2O, SnapKV, KV-cache quantization and the use of uncertainty in cache management for subword models are published work, and we do not claim them. We implement simplified versions for comparison: mean pooling in place of cross-attention, no n-gram hash embeddings, and a symmetric per-vector absmax quantizer.

## Closest published work

Each entry was checked on arXiv (title, authors, date) on 27 October 2026. The last column states how much of each paper was read.

| Work | Contribution | Difference from this work | Read |
|---|---|---|---|
| BLT, Pagnoni et al. (arXiv 2412.09871) | Entropy-based dynamic patches, a latent Transformer over patches, FLOP-controlled scaling to 8B parameters | Ours is a small, simplified BLT; we study compression of its KV cache, which the abstract does not address | Abstract |
| StreamingLLM, Xiao et al. (2309.17453) | Sink plus recent window for streaming | Our recency policy, applied to the patch caches of a byte-level model | Abstract |
| H2O, Zhang et al. (2306.14048) | Retains positions with the largest accumulated attention | Implemented online over patches; worse than recency in 22 of 24 comparisons here | Abstract |
| SnapKV, Li et al. (2404.14469) | Selects positions attended to by the last queries | Our SnapKV-style rule uses attention from the last 16 patches; better than recency in 24 of 24 comparisons | Abstract |
| KIVI (2402.02750), KVQuant (2401.18079) | Low-bit KV quantization | We use a simpler quantizer and test whether entropy can allocate bits (it does not beat recency tiers) | Abstract |
| CONF-KV, Li and Miao (2605.24786) | Next-token entropy sets a per-step cache budget; tokens ranked by attention and recency; random eviction as a control; subword models | There entropy determines how much to keep; here it ranks which patches to keep. Their random control is the closest precedent for ours | Full text |
| Counter-causal surprise, Gould and van den Hengel (2607.27600) | Scores cached tokens by how well later tokens predict them | A backward signal that needs an extra pass; ours is forward entropy | Full text |
| InfoKV, Kai et al. (2606.26875) | Combines predictive entropy and layer-wise change with attention for long reasoning | Entropy is one component there; we test it alone | Abstract |
| Don't Waste Bits, Haeri Boroujeni et al. (2604.04722) | A learned controller assigns each token 2 to 16 bits using features that include entropy | We test entropy alone as the bit-width rule | Abstract |
| Scratchpad Patching, Zheng et al. (2605.09630) | Uses next-byte uncertainty to add temporary slots in byte patches; reports a smaller KV cache | The closest byte-level work we found; it modifies the model, whereas we keep the model fixed and choose what the cache holds | Abstract |

For the three works read only in abstract (InfoKV, Don't Waste Bits, Scratchpad Patching), the statement of difference is provisional.

## Claims

1. On a byte-level model, entropy-ranked patch eviction is not better than random or recency at equal bytes, across three seeds, four settings, tenfold longer training, longer contexts and a test split.
2. Attention-guided eviction, 4-bit quantization and bounded recency caches are effective on patch caches, with the figures listed in `README.md`.
3. These conclusions apply to 13.3M-parameter models that do not retrieve named facts.
