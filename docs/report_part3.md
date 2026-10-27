# Part 3 report: summary and evidence map

The submitted report is `report/main.tex` (built `report/main.pdf`, 10 pages with references). This file is a short companion for readers and examiners: the findings in one page, and a map from each assessed item to the evidence that backs it. Every number below is in `docs/results_summary.md` or `results/part3/`. Validation data throughout; the WikiText test split was evaluated once, and TinyStories has no test split.

## Findings

**Setup.** A 13,333,632-parameter BLT-style byte model (entropy or fixed-stride patching, one KV cache entry per patch) with patch-granular cache policies: recency with a sink, random, keep high entropy (the proposal), keep low entropy, H2O and a SnapKV-style rule, plus uniform and tiered quantization. Three seeds per setting, paired bootstrap intervals over documents, 70 passing tests.

**What improved.**
- Eviction plus 4-bit quantization keeps 7.4 percent of the WikiText cache bytes for +0.0012 BPB (SnapKV-style, 128 patches; +0.0019 with recency). Uniform 4-bit alone costs 0.0003 to 0.0014 BPB at 27 percent of the bytes.
- An online SnapKV-style policy, scored on the model's own attention, beats recency in 24 of 24 comparisons (by 0.0002 to 0.0077 BPB). Online H2O is worse than recency in 22 of 24.
- A fixed-size cache follows documents four times longer than the training context better than resetting the context (recency, 64 patches: 0.0030, 0.0032 and 0.0008 BPB better in three seeds), and 20 PG-19 books by 0.0086 BPB (256 patches). Measured: global cache 0.59 to 2.36 MB against 136.5 MB at 65,536 bytes, time per byte constant at 0.9 ms against a rise to 2.0 ms.
- Entropy patching is lower in BPB than fixed stride in 3 of 3 same-seed pairs on WikiText (validation and test); 67 percent of its patch starts are at word starts against 16 to 20 percent by chance. The gap is about one seed range, so we call it likely, not established.
- The BLT needs 2.8 times fewer FLOPs per byte than a byte-level Transformer of the same width (analytic, `results/part3/flops.md`).

**What did not work.** Keeping high-entropy patches is worse than random in 68 and than recency in 70 of 72 comparisons, better in none; the penalty against random at 64 patches grows from +0.002 (3,000 steps) to +0.024 to +0.029 (30,000 steps). It holds for fixed-stride models, with the code mix, at 16,384 bytes, on the test split, for a recency-protected variant, and under every scoring-rule ablation. The error analysis shows the rule keeps mostly one- and two-byte fragments and loses on the easy bytes (`docs/error_analysis_part3.md`).

**Limits.** 13.3M-parameter models; at equal training the BPE Transformer scores lower BPB than this BLT (0.6939 against 0.7980 on TinyStories, 1.5200 against 1.6801 on WikiText); the models do not retrieve (planted-name test); one seed for the 16k and fixed-stride 30,000-step models; PG-19 confounds the entropy signal with distance.

## Evidence map by assessed item

| Item | What the report shows | Files |
|---|---|---|
| Model novelty | A controlled test of entropy-ranked retention on a byte-level patch model; online attention policies over patch caches; measured bounded-memory streaming; analytic compute count | `NOVELTY.md`, `report/main.pdf` sections 1, 2 and 5, `tests/test_blt.py` |
| Improvement over the previous technique | Table 1 of the report: against Part 2 baselines, the byte-level Transformer, recency, and context resets | `results/part3/deploy_combo.md`, `online_policies.md`, `stream_16k.md`, `pg19_stream.md`, `deploy_bench.md`, `flops.md` |
| Results and error analysis | Tables with intervals, three seeds, test split, ten times more training; quantitative error analysis (tercile, byte class, patch length) and real examples | `results/part3/seeds.md`, `test_split.md`, `scale_and_tuning.md`, `error_analysis.md`, `docs/error_analysis_part3.md` |
| Challenges | Table of 11 challenges with actions and outcomes in the report; 20 entries in full | `CHALLENGES.md` |
| Demo and viva | CPU demo (options 4, 6, 7, 5) verified from a fresh clone; rehearsal script; questions with file-backed answers | `docs/demo_script.md`, `docs/viva_by_rubric.md`, `docs/viva_notes.md` |

## Regenerating

```bash
python report/make_figures.py
python scripts/flops.py
python scripts/error_analysis_part3.py
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
