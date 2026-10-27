# The Part 1 proposal, weighed against what was delivered

The commitments are those of the Part 1 proposal (27 Aug 2026) and the Part 2 report (15 Oct 2026); the evidence for each row resides in the file named beside it. A promise is only as good as its audit, and this is ours.

| Commitment | Delivered | Evidence |
|---|---|---|
| Reuse the BLT patching entropy to manage the KV cache of an autoregressive byte-level model | Done. Result: the entropy rule is not better than random or recency (worse in 68 and 70 of 72 comparisons); attention-guided eviction, 4-bit quantization and bounded caches work | `report/main.pdf`, `docs/results_summary.md` |
| Classical baseline: byte n-gram with Kneser-Ney smoothing | Done in Part 2 (interpolated, one discount per order instead of the three of modified Kneser-Ney, as stated in the Part 2 report): 1.8085 and 2.4103 BPB | `results/phase2.md` |
| Modern baseline: BPE Transformer with an uncompressed cache | Done in Part 2: 13.8M parameters, 0.6939 and 1.5200 BPB | `results/phase2.md` |
| Byte-level baseline: BLT-style model with a separately trained entropy patcher, uncompressed cache | Done: 13,333,632 parameters, 3.3M-parameter entropy model, entropy and fixed-stride patching, three seeds | `docs/block_2_3.md`, `results/part3/seeds.md` |
| Our system compared with StreamingLLM-style recency and random eviction at the same budget | Done, plus keep-low-entropy, H2O, SnapKV-style, quantization and eviction combined | `results/part3/comparison.md`, `online_policies.md`, `deploy_combo.md` |
| Models of 10 to 30 million parameters, trained from scratch | 13.3M and 13.8M; cluster GPUs used for the long runs | `CHALLENGES.md` |
| Datasets: TinyStories, WikiText-103, The Stack Smol, PG-19 | All four used. Sizes differ from the proposal: 30,000 TinyStories train stories (not about 2.1M), all 29,002 WikiText-103 train articles, 28,229 code files, 100 PG-19 test books (20 streamed) | report Table 2, `data/processed/*/stats.json` |
| Metric: bits per byte, comparable across tokenizers | Done | report section 5.1 |
| Efficiency: peak KV memory at fixed contexts, decode latency, throughput, FLOPs per byte | Memory computed and measured on a GPU, time per byte measured, FLOPs per byte analytic (new) | `results/part3/deploy_memory.md`, `deploy_bench.md`, `flops.md` |
| Retrieval check: needle in a haystack | Done as a planted-name test; uninformative because the models do not retrieve | `results/part3/deploy_needle.md` |
| Error analysis: BPB by entropy region | Done: entropy terciles, byte classes, kept-patch length | `docs/error_analysis_part3.md` |
| Error analysis: code against prose | Done: models retrained with a 15 percent code mix | `results/part3/code_domain.md` |
| Error analysis: which patches had been evicted where a generation went wrong | Done: kept and evicted patches shown, and greedy generations compared across policies | `docs/error_analysis_part3.md`, `results/part3/generation_divergence.md` |
| Risk: variable-length patching hard to implement; fall back to fixed stride | Entropy patching worked; fixed strides 4 and 5 kept as controls | `results/part3/seeds.md` |
| Plan: Part 3 report, demo and final results by 21 Nov | Experiments finished 25 Oct; report, supplement, slides, demo script and viva notes written | `report/`, `docs/demo_script.md`, `docs/presentation_notes.md` |

Departures from the proposal, confessed without ceremony: the entropy hypothesis was not supported; TinyStories was used at 30,000 stories because of compute; Kneser-Ney uses one discount per order; and the planted-name retrieval probe cannot discriminate between policies at this scale.
