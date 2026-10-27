# Presentation notes (slides in `report/slides.pdf`)

Speaker notes for the 22 slides. All slides take about 14 minutes; skipping the slides marked optional gives about 12 minutes. Every number is in `docs/results_summary.md` or `results/part3/`. Source: `report/slides.tex`; build with `cd report && pdflatex slides`.

| # | Slide | Say (30 to 45 seconds each) | Time |
|---|---|---|---|
| 1 | Title | Group 12, byte-level KV cache compression. | 0:10 |
| 2 | Why this problem | Chat models read subword tokens. A byte model has no unknown words but needs four times more positions, so its KV cache is four times larger, and the cache limits batch size and context. The BLT already computes entropy to place patches. Question: can that free signal also pick what to keep? | 0:45 |
| 3 | Relation to public LLMs | Llama 3 8B uses grouped-query attention with 8 KV heads to shrink the cache and still reaches 17 GB at 128k tokens, above its 16 GB of weights. Our models are 600 times smaller; we compare cost structure and methods, not chat quality. | 0:45 |
| 4 | Task and metric | Language modelling in bits per byte so a 4-gram, a BPE model and a byte model are scored on identical text. Datasets, baselines, three seeds, test split once. | 0:40 |
| 5 | System | Entropy model sets patches; local encoder; global Transformer over patches holds the cache; policy decides what stays. 13.3M parameters. | 0:45 |
| 6 | Patches shrink the cache | 43 bytes become 4 entries. 67 percent of patch starts are at word starts. | 0:30 |
| 7 | Policies | Recency, random, entropy, H2O, SnapKV-style, plus 4-bit. First patch and 8 newest always protected so the signal is tested, not the window. | 0:40 |
| 8 | Baselines (optional) | Everything beats the 4-gram; BPE is ahead at equal training; ten times more training puts our WikiText BLT at 1.29 to 1.31. State the caveat. | 0:40 |
| 9 | Seeds (optional) | Seed spread equals the gaps; we retired a 3.3 percent claim. | 0:30 |
| 10 | Main test | 72 paired comparisons: entropy is worse than random in 68, than recency in 70, never better; holds on the test split; SnapKV-style beats recency 24 of 24. | 0:50 |
| 11 | More training | The penalty grows about tenfold with ten times more training, so it is not a small-model artefact. | 0:35 |
| 12 | Why | The rule keeps one- and two-byte fragments (98 to 99.7 percent) and drops patches with words; attention to old high-entropy patches is lower. Correlation, not proof. | 0:50 |
| 13 | Where loss falls (optional) | High-entropy bytes are not protected and easy bytes pay; 1 percent of bytes carry 18 to 30 percent of the extra loss. | 0:30 |
| 14 | What works | 128 patches in 4-bit: 7.4 percent of the cache for +0.0012 BPB. | 0:45 |
| 15 | Streaming | Fixed recency cache beats context reset in 3 of 3 seeds and on 20 PG-19 books. | 0:40 |
| 16 | Memory and time | 0.59 MB against 136.5 MB at 65,536 bytes; 0.9 ms per byte against 2.0. | 0:30 |
| 17 | Compute (optional) | 2.8 times fewer FLOPs than a byte Transformer, but more than BPE; the case is memory. Analytic. | 0:30 |
| 18 | Error analysis | Real divergence of generations and per-byte examples; anecdotes. | 0:40 |
| 19 | Challenges | Pick two: seed variance and the accounting error; both were found by our own checks and corrected. | 0:45 |
| 20 | Demo | Option 6 live training, 4 compress, 7 stream; see `docs/demo_script.md`. | 3:00 |
| 21 | Limits | 13.3M parameters, models do not retrieve, cache sizes computed, PG-19 confounded. | 0:30 |
| 22 | Conclusion | New, improved, did not work, open. | 0:30 |

Answers to likely questions: `docs/viva_by_rubric.md`.
