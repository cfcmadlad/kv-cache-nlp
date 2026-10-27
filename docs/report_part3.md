# The Part 3 report in brief, with a map to its evidence

The submitted report is `report/main.tex` (built as `report/main.pdf`: ten pages, references included), accompanied by `report/supplement.pdf` and the slides `report/slides.pdf`. What follows is a companion for the reader who wants the findings on one sheet, and the examiner who wants to know where each assessed item is substantiated. Every figure below appears in `docs/results_summary.md` or `results/part3/`. Validation data throughout; the WikiText test split was consulted once, and TinyStories has none.

## The findings

**Apparatus.** A BLT-style byte model of 13,333,632 parameters (entropy or fixed-stride patching, one KV-cache entry per patch) with patch-granular cache policies: recency with a sink, random, keep high entropy (the proposal), keep low entropy, H2O and a SnapKV-style rule, together with uniform and tiered quantization. Three seeds per setting, paired bootstrap intervals over documents, and 71 passing tests.

**What improved.**
* Eviction joined to 4-bit quantization retains 7.4 percent of the WikiText cache bytes at a cost of +0.0012 BPB (SnapKV-style, 128 patches; +0.0019 with recency). Uniform 4-bit alone costs 0.0003 to 0.0014 BPB at 27 percent of the bytes.
* An online SnapKV-style policy, scored on the model's own attention, beats recency in 24 of 24 comparisons (by 0.0002 to 0.0077 BPB); online H2O is worse than recency in 22 of 24.
* A cache of fixed size follows documents four times longer than the training context better than resetting the context does (recency, 64 patches: 0.0030, 0.0032 and 0.0008 BPB better in three seeds), and 20 PG-19 books by 0.0086 BPB (256 patches). Measured: a global cache of 0.59 to 2.36 MB against 136.5 MB at 65,536 bytes, and a time per byte constant at 0.9 ms against a climb to 2.0 ms.
* Entropy patching is lower in BPB than fixed stride in 3 of 3 same-seed pairs on WikiText (validation and test); 67 percent of its patch starts fall at word starts, against 16 to 20 percent by chance. The gap is about one seed range, so we call it likely, not established.
* The BLT needs 2.8 times fewer FLOPs per byte than a byte-level Transformer of equal width (analytic; `results/part3/flops.md`).

**What did not work.** Keeping high-entropy patches is worse than random in 68 and than recency in 70 of 72 comparisons, better in none; the penalty against random at 64 patches grows from +0.002 (3,000 steps) to +0.024 to +0.029 (30,000 steps). It persists for fixed-stride models, with the code mix, at 16,384 bytes, on the test split, for a recency-protected variant, and under every scoring-rule ablation. The error analysis shows that the rule keeps chiefly patches of one or two bytes and forfeits on the easy bytes (`docs/error_analysis_part3.md`).

**Limits.** Models of 13.3M parameters; at equal training the BPE Transformer scores lower BPB than this BLT (0.6939 against 0.7980 on TinyStories, 1.5200 against 1.6801 on WikiText); the models do not retrieve (planted-name test); one seed for the 16k and fixed-stride 30,000-step models; PG-19 confounds the entropy signal with distance.

## Where each assessed item is substantiated

| Item | What the report shows | Files |
|---|---|---|
| Model novelty | A controlled trial of entropy-ranked retention on a byte-level patch model; online attention policies over patch caches; measured bounded-memory streaming; an analytic compute count | `NOVELTY.md`, `report/main.pdf` sections 1, 2 and 5, `tests/test_blt.py` |
| Improvement over the previous technique | Table 1 of the report: against the Part 2 baselines, a byte-level Transformer, recency, and context resets | `results/part3/deploy_combo.md`, `online_policies.md`, `stream_16k.md`, `pg19_stream.md`, `deploy_bench.md`, `flops.md` |
| Results and error analysis | Tables with intervals, three seeds, the test split, tenfold training; quantitative error analysis (terciles, byte classes, patch length) and real examples | `results/part3/seeds.md`, `test_split.md`, `scale_and_tuning.md`, `error_analysis.md`, `generation_divergence.md`, `docs/error_analysis_part3.md` |
| Challenges | A table of eleven challenges with actions and outcomes in the report; twenty entries in full | `CHALLENGES.md` |
| Demonstration and viva | A CPU demonstration (options 1 to 7) verified from a fresh clone; a rehearsal script; questions with file-backed answers | `docs/demo_script.md`, `docs/viva_by_rubric.md`, `docs/viva_notes.md` |

## Regeneration

```bash
python report/make_figures.py
python scripts/flops.py
python scripts/error_analysis_part3.py
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
