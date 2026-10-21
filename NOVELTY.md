# Novelty and related work

Core claim: in byte-level models, next-byte entropy is already computed to place patch boundaries, so using it to decide which cache entries to evict or quantize costs nothing extra, and patches give a natural eviction unit. We test whether entropy-guided retention beats random and recency eviction at matched cache bytes, judged on BPB against cache bytes with bootstrap confidence intervals. If it does not, the report says so.

## Closest published work

All of the following target subword LLMs. Nothing found so far applies entropy-guided eviction or quantization to byte-level or BLT-style models.

| Work | What it does | How we differ |
|---|---|---|
| InfoKV (arXiv 2606.26875) | Retains entries by predictive entropy times layer drift, mixed 10 percent with attention. No random control. | Not read in full by us; the summary is from the project notes. Entropy is one ingredient of a mixed score. We use entropy alone on patches and include a random control. |
| CONF-KV (arXiv 2605.24786, Li and Miao) | Turns the next-token distribution (normalised entropy, log-probability margin, top-token mass) into a confidence score that picks a tight or loose per-step cache budget. Tokens inside the budget are ranked by EMA attention mass and recency. FP16/INT8 mixed storage. Random eviction with the same schedule is a control (36.54 PPL against 30.92 for full CONF-KV on GPT-2 WikiText-2). Subword models only (GPT-2, Qwen). | Entropy sets how much to keep, not which entries to keep; ranking still uses attention and recency. We use entropy to rank which entries (patches) to keep, at a fixed budget, with no attention scores needed. Their random-schedule control is the closest precedent for ours and shows the control is expected. |
| Counter-Causal Surprise (arXiv 2607.27600, Gould and van den Hengel) | Scores past tokens by how well later tokens predict them (a backward, counter-causal pass over cached keys and values). Low surprise means redundant, so evict. Baselines are full cache, sliding window, H2O; no random control found. | Backward predictability needs an extra pass over the cache. Our signal is forward next-byte entropy, already available from decoding. The two signals are different and could be compared as an extra baseline if time allows. |
| Don't Waste Bits (arXiv 2604.04722) | Learned bit-width controller; entropy is one input feature. No ablation of its role. | Learned controller on subword models. We test entropy alone as the bit-width rule. |
| Scratchpad Patching (arXiv 2605.09630) | Uses next-byte entropy to spend compute. | Compute allocation, not cache eviction. |

CONF-KV also cites AdaKV, ZigZagKV, UNComp (matrix entropy) and an entropy-based adaptive precision method; these adapt budgets by attention or layer statistics rather than next-token entropy over retained entries.

## What the full-text reading changed

Read in full on 2026-10-21: CONF-KV and Counter-Causal Surprise. CONF-KV is closer than the project notes implied: it already uses next-token entropy and already has a random-eviction control. The honest difference is the role of entropy (budget size versus which entries to keep) and the model family (byte-level). We should not claim that entropy-guided cache management is new. We can claim that it has not been tested on byte-level models, where the signal is free and patches define the unit.

Not read in full: InfoKV and Don't Waste Bits. Open both before citing them as different in the report.

## Claims we can defend

1. Entropy-ranked patch eviction, evaluated on a byte-level model with patch-granular caches, against random and recency at equal cache bytes.
2. Whether entropy computed by a separate small model (as in BLT) is as good as the main model's own entropy.
3. Whether the result holds on two domains (stories and Wikipedia plus code).

If entropy eviction does not beat the controls, claim 1 becomes a negative result with confidence intervals.
