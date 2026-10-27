# Novelty and related work

## What we studied

In byte-level models, next-byte entropy is already computed to place patch boundaries, and a patch is a natural cache unit. We tested whether that entropy can also decide which cache entries to keep or to store at low precision, against random, recency, attention-based and quantization baselines at equal cache bytes, judged on BPB against cache bytes with paired bootstrap intervals, three seeds per setting and a test split.

## What is new

1. **A controlled test of entropy-ranked retention on a byte-level patch model.** We found no earlier test of this combination (byte-level model, patch-granular cache, entropy as the keep signal, random and recency controls at equal bytes). The result is negative: keeping high-entropy patches is worse than random in 68 and than recency in 70 of 72 comparisons, and the penalty grows with training (`docs/results_summary.md`, `results/part3/seeds.md`).
2. **An explanation of what the entropy rule keeps.** The mean-entropy rule keeps almost only one- and two-byte patches (98.0 and 99.7 percent), and gains on a few surprising bytes but loses on the many easy ones (`docs/error_analysis_part3.md`).
3. **An online, tested implementation of H2O and SnapKV-style policies over patch caches** (`BLT.forward_online`, `tests/test_blt.py`). The SnapKV-style policy beats recency in 24 of 24 online comparisons (`results/part3/online_policies.md`).
4. **Measured bounded-memory streaming on a byte-level model**, with real eviction: constant memory and time per byte, and better BPB than resetting the context on long WikiText documents and 20 PG-19 books (`results/part3/stream_16k.md`, `pg19_stream.md`, `deploy_bench.md`).
5. **An analytic compute comparison** that locates where the BLT saves (cache memory, and FLOPs against a byte-level Transformer) and where it does not (FLOPs against BPE) (`results/part3/flops.md`).

## What is not new

The BLT, StreamingLLM, H2O, SnapKV, KV quantization, and the use of uncertainty for cache management on subword models are published. We implement simplified versions for comparison (mean pooling instead of cross-attention in the BLT, no n-gram hash embeddings, a symmetric per-vector absmax quantizer) and do not claim them.

## Closest published work

All entries below were checked on arXiv (title, authors, date) on 27 Oct 2026. "Read" says how much of the paper we read.

| Work | What it does | How this work differs | Read |
|---|---|---|---|
| BLT, Pagnoni et al. (arXiv 2412.09871) | Entropy-based dynamic patches, latent Transformer over patches, FLOP-controlled scaling to 8B | Our model is a small simplified BLT; we study compression of its KV cache, which the abstract does not address | Abstract |
| StreamingLLM, Xiao et al. (2309.17453) | Sink plus recent window for streaming | Our recency policy; we apply it to patch caches of a byte model | Abstract |
| H2O, Zhang et al. (2306.14048) | Keeps positions with the largest accumulated attention | Implemented online over patches; worse than recency in 22 of 24 here | Abstract |
| SnapKV, Li et al. (2404.14469) | Selects positions attended to by the last queries | Our SnapKV-style rule uses attention from the last 16 patches; better than recency in 24 of 24 | Abstract |
| KIVI (2402.02750), KVQuant (2401.18079) | Low-bit KV quantization | We use a simpler quantizer and test whether entropy can allocate bits (it cannot beat recency tiers) | Abstract |
| CONF-KV, Li and Miao (2605.24786) | Next-token entropy sets a per-step cache budget; tokens ranked by attention and recency; random eviction as a control; subword models | Entropy sets how much to keep there; here it ranks which patches to keep. Their random control is the closest precedent for ours | Full text |
| Counter-causal surprise, Gould and van den Hengel (2607.27600) | Scores cached tokens by how well later tokens predict them | Backward signal needing an extra pass; ours is forward entropy | Full text |
| InfoKV, Kai et al. (2606.26875) | Mixes predictive entropy and layer-wise change with attention for long reasoning | Entropy is one ingredient there; we test it alone | Abstract |
| Don't Waste Bits, Haeri Boroujeni et al. (2604.04722) | Learned controller gives each token 2 to 16 bits using features including entropy | We test entropy alone as the bit-width rule | Abstract |
| Scratchpad Patching, Zheng et al. (2605.09630) | Uses next-byte uncertainty to add temporary slots in byte patches; reports a smaller KV cache | Closest byte-level work we found; it changes the model, we keep the model and select what the cache holds. Worth reading in full before submission | Abstract |

For the three works read only at abstract level (InfoKV, Don't Waste Bits, Scratchpad Patching) the statement of difference is provisional.

## Claims we defend

1. Entropy-ranked patch eviction on a byte-level model is, in our experiments, not better than random or recency at equal bytes, across three seeds, four settings, ten times more training, longer contexts and a test split.
2. Attention-guided eviction, 4-bit quantization and bounded recency caches do work on patch caches, with the numbers listed in `README.md`.
3. These conclusions hold for 13.3M-parameter models that do not retrieve; larger models and retrieval tasks are untested.
