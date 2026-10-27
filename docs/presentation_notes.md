# Notes for the speaker (slides in `report/slides.pdf`)

Counsel for the delivery of the twenty-two slides, with timings. The whole sequence occupies about fourteen minutes; omitting the slides marked optional brings it to about twelve. Every figure quoted is in `docs/results_summary.md` or `results/part3/`. The source is `report/slides.tex`; build it with `cd report && pdflatex slides`.

| # | Slide | Say (thirty to forty-five seconds apiece) | Time |
|---|---|---|---|
| 1 | Title | Group 12; the compression of the KV cache in byte-level language models. | 0:10 |
| 2 | Why this problem | Chat models read subword tokens. A byte model has no unknown words but needs four times the positions, so its KV cache is four times larger, and the cache limits batch size and context. The BLT already computes an entropy to place its patches; can that gratuitous signal also choose what to keep? | 0:45 |
| 3 | Relation to public LLMs | Llama 3 8B employs grouped-query attention with 8 KV heads to shrink the cache and still reaches 17 GB at 128k tokens, more than its 16 GB of weights. Our models are 600 times smaller; we compare cost structure and methods, not conversational quality. | 0:45 |
| 4 | Task and metric | Language modelling in bits per byte, so that a 4-gram, a BPE model and a byte model are scored on identical text. Datasets, baselines, three seeds, the test split used once. | 0:40 |
| 5 | System | The entropy model sets the patches; a local encoder; a global Transformer over patches holds the cache; a policy decides what remains. 13.3M parameters. | 0:45 |
| 6 | Patches shrink the cache | Forty-three bytes become four entries. Sixty-seven percent of patch starts fall at word starts. | 0:30 |
| 7 | Policies | Recency, random, entropy, H2O, SnapKV-style, with 4-bit besides. The first patch and the eight newest are always spared, so that the signal is tested and not the window. | 0:40 |
| 8 | Baselines (optional) | Everything beats the 4-gram; BPE leads at equal training; tenfold training brings our WikiText BLT to 1.29 to 1.31. State the caveat. | 0:40 |
| 9 | Seeds (optional) | The seed spread equals the gaps; we retired a 3.3 percent claim. | 0:30 |
| 10 | The main trial | Seventy-two paired comparisons: entropy is worse than random in 68, than recency in 70, and never better; it holds on the test split; SnapKV-style beats recency 24 of 24. | 0:50 |
| 11 | More training | The penalty grows about tenfold with tenfold training, so it is no artefact of small models. | 0:35 |
| 12 | Why | The rule keeps one- and two-byte fragments (98 to 99.7 percent) and discards the patches that carry words; attention to old high-entropy patches is lower. A correlation, not a proof. | 0:50 |
| 13 | Where the loss falls (optional) | High-entropy bytes are not protected and the easy bytes pay; one percent of the bytes carry 18 to 30 percent of the extra loss. | 0:30 |
| 14 | What works | 128 patches in 4-bit: 7.4 percent of the cache for +0.0012 BPB. | 0:45 |
| 15 | Streaming | A fixed recency cache beats a context reset in 3 of 3 seeds and on 20 PG-19 books. | 0:40 |
| 16 | Memory and time | 0.59 MB against 136.5 MB at 65,536 bytes; 0.9 ms per byte against 2.0. | 0:30 |
| 17 | Compute (optional) | 2.8 times fewer FLOPs than a byte Transformer, but more than BPE; the case rests on memory. Analytic. | 0:30 |
| 18 | Error analysis | Real divergences of generations and per-byte examples; anecdotes, honestly labelled. | 0:40 |
| 19 | Challenges | Choose two: seed variance and the accounting error; both were found by our own checks and corrected. | 0:45 |
| 20 | Demonstration | Option 6 live training, 4 compress, 7 stream; see `docs/demo_script.md`. | 3:00 |
| 21 | Limits | 13.3M parameters; the models do not retrieve; cache sizes computed; PG-19 confounded. | 0:30 |
| 22 | Conclusion | What is new, what improved, what failed, what remains open. | 0:30 |

Answers to the questions likely to be put: `docs/viva_by_rubric.md`.
