# A chronicle of the project, and its plan

Official deadline for Part 3: 21 November 2026; the team's own target was 12 November, the intervening nine days serving as a cushion against misadventure. The assessment for Part 3 (30 marks) rewards Novelty (3), Improvement (3), Results and error analysis (2), Challenges (2) and Demonstration and viva (20); any block that served none of these was to be cut first.

## Where matters stand (27 Oct 2026)

All experiments are complete and committed. The deliverables are in place: the paper-style report (`report/main.tex`, ten pages with references), a supplement, a 22-slide deck with notes, an error analysis (`docs/error_analysis_part3.md`), an analytic FLOPs table, a demonstration script verified on CPU from a fresh clone, anticipated viva questions, a verified literature survey (`NOVELTY.md`) and the full record of difficulties (`CHALLENGES.md`). What remains is human: rehearse the demonstration on the machine from which it will be given, read the two long documents once more with a cold eye, and read the Scratchpad Patching paper in full.

## How the weeks unfolded

| Date | Event |
|---|---|
| 27 Aug | Part 1 proposal submitted |
| 15 Oct | Part 2 submitted: data pipeline, Kneser-Ney byte 4-gram and BPE Transformer baselines |
| 21 Oct | Byte datasets, entropy model, BLT-style model, cache policies, quantization; first four BLT runs; the headline result already negative for the entropy rule |
| 21 Oct (evening) | Seeds, stride-5 control, ablation tooling, the Part 3 demonstration |
| 21 Oct (night) | Three seeds, controls and ablations on the cluster; code dataset built; code-mix retraining; policy evaluation rerun after an accounting fix |
| 25 Oct | Online H2O and SnapKV-style; eviction with 4-bit quantization; recency-protected entropy; bounded-memory streaming; measured decoding; planted-name recall; 30,000-step models; 16k tuning; requeue test; PG-19 streaming on 20 books |
| 26 Oct to 27 Oct | Report, supplement, slides, error analysis, FLOPs, demo rehearsal, verified literature |

Known gaps, stated without embroidery: positions beyond the training range were not renumbered inside the cache; there is no retrieval conclusion because the models do not retrieve; long TinyStories training overfits; the 16k and fixed-stride 30,000-step models have one seed each.

## The original plan, by phase

### Phase 2: byte models

| Block | Done when | Rubric |
|---|---|---|
| 2.0 Infrastructure | pytest passes; a smoke job under five minutes writes `meta.json`; a killed job resumes at the saved step | Challenges, Demo |
| 2.1 Byte datasets | byte counts match `stats.json`; 100 documents round-trip; no duplicate code hashes across splits | Results |
| 2.2 Entropy model | validation BPB in a plausible range; at least 60 percent of WikiText patch starts at word starts; causality test passes | Novelty |
| 2.3 BLT-style model | incremental decode equals the full forward within 1e-5; leakage test; parameter count matches a hand calculation | Novelty, Demo |
| 2.4 Training on the cluster | four BLT runs (two datasets, entropy and fixed stride) plus two entropy models | Results |
| 2.5 Gate | `scripts/results.py --phase 2` prints the five-row table; the Part 2 rows reproduce exactly | Results |

Decision points: if incremental decoding were not green by 25 Oct, fixed stride would become the main model; if entropy patching proved unstable or more than 10 percent worse in loss than fixed stride at equal steps by 26 Oct, likewise. Neither occurred.

### Phases 3 to 9

* **3. Cache policies.** A patch-granular cache with byte accounting; full, recency (a sliding window with a sink), random and entropy-ranked eviction under one budget interface.
* **4. Quantization.** Per-patch precision by entropy against uniform and random precision at matched bytes.
* **5. Main evaluation.** BPB against cache bytes on TinyStories and WikiText plus code validation, tuned on validation only; three seeds; bootstrap intervals over documents; the test split touched once per final configuration.
* **6. Ablations.** Own entropy against a separate entropy model; patch against byte-level eviction; sink size; a fixed-stride model under the same policies.
* **7. Error analysis.** Where eviction hurts: per document, per domain, and which kinds of bytes are evicted.
* **8. Demonstration.** `scripts/demo.py`, `run.bat` and `run.sh` on a CPU laptop from a clean clone, with new release assets and a full rehearsal.
* **9. Comparative study and report.** All policies and baselines in one view, a written comparison with related work, the report sections, and viva notes collated from `docs/`.

## Risks, as foreseen

1. Incremental decoding at patch boundaries (see the decision points above).
2. One GPU job at a time, with queueing behind other users; Phase 2.4 was the long pole.
3. A crowded calendar; if Phase 2 slipped, quantization would be cut first, then ablations.
4. The core result might be negative, and the plan did not depend upon a positive one. It was, in the event, negative.

## Rules we held ourselves to

Configurations rather than hard-coded values; fixed seeds; every reported number regenerable from a single command; tuning on validation only; no overwriting of `checkpoints/` or the Part 2 results; a pause, with options laid out, whenever a block exceeded one and a half times its estimate.
