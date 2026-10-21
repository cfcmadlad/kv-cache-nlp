# Plan to 12 November 2026

Written 21 Oct 2026. Official deadline 21 Nov; 12 Nov is our target, the nine days after are buffer. Rubric for Part 3 (30 marks): Novelty 3, Improvement 3, Results and error analysis 2, Challenges 2, Demo and viva 20. A block that maps to none of these is cut first.

Phases 0 and 1 are Parts 1 and 2 (done). Phases 3 to 9 below are my outline from the project notes, not a spec you gave me; please correct them.

## Status on 21 Oct (end of day)

Done and tested: 2.1 (except the code dataset, which is gated), 2.2, 2.3, 2.4 (four BLT runs, two entropy models, laptop GPU), 2.5 gate table, Phase 3 policies, Phase 4 quantization, an attention mechanism analysis from Phase 7. Numbers: `docs/results_summary.md`. Not done: 2.0 cluster verification, code dataset, three seeds, Phase 8 demo update, report text. The headline result is negative for the entropy rule; see the summary.

## Status on 21 Oct (evening)

Added since: seed 1 and 2 configs and a fixed-stride-5 control (`scripts/run_seeds.sh`; seed 1 of TinyStories entropy done and the sign holds), ablation tooling (`--signal own`, `--agg first|sum`, sink size, byte-level model with stride 1; `scripts/run_ablations.sh`), the Part 3 demo (option 4, `scripts/part3_assets.py`, tested from a clean copy on CPU), `scripts/compare.py seeds` and `scripts/collate_viva.py`. Waiting on the GPU queue: remaining seeds, ablations. Needs the HPC or the user: Block 2.0 verification, code dataset, upload of the `v0.3-part3` release asset (zip built by `part3_assets.py build`, SHA-256 already in the script), Part 1 and 2 gaps. Phase 9 tables and report text come after the runs finish.

## Status on 21 Oct (night)

Done on the cluster: three seeds, stride-5 control, ablations (own entropy, scoring rules, sink, byte-level), policy evaluation rerun after an accounting fix, code dataset built and verified. Tables: `results/part3/` (`seeds.md`, `ablations.md`, `comparison.md`, `code_domain.md`). Written: `docs/results_summary.md`, `docs/phase_6_ablations.md`, `docs/report_part3.md`, `docs/viva_notes.md` (collated). Phase 8 demo done and tested from a clean copy. Code-mix WikiText models trained and evaluated (`scripts/run_mix.sh`). Open: Block 2.0 requeue test, release asset delivery for the demo, Part 1 and 2 gaps, test split evaluation (once per final configuration), attention analysis on more seeds.

## Phase 2: byte models (21 Oct to 31 Oct)

| Block | Dates | Done when | Rubric |
|---|---|---|---|
| 2.0 Infrastructure | 21 to 22 Oct | pytest passes; smoke job under 5 min writes meta.json; killed job resumes at saved step | Challenges, Demo |
| 2.1 Byte datasets | 22 to 23 Oct | byte counts match stats.json; 100 docs round-trip; no duplicate code hashes across splits | Results |
| 2.2 Entropy model | 23 to 24 Oct | val BPB in plausible range; at least 60 percent of WikiText patch starts at word starts; causality test passes | Novelty |
| 2.3 BLT-style model | 24 to 25 Oct | incremental decode equals full forward within 1e-5; leakage test; parameter count matches hand calculation | Novelty, Demo |
| 2.4 Training on HPC | 25 to 30 Oct | four BLT runs (two datasets, entropy and fixed stride) plus two entropy models | Results |
| 2.5 Gate | 31 Oct | `scripts/results.py --phase 2` prints the five-row table, Part 2 rows reproduce exactly | Results |

Decision points: 25 Oct, if 2.3 incremental decoding is not green, fixed stride becomes the main model. 26 Oct, if entropy patching is unstable or more than 10 percent worse in loss than fixed stride at equal steps, fixed stride becomes the main model.

## Phase 3: cache policies (1 to 3 Nov)

Patch-granular cache with byte accounting (cache bytes per 1000 input bytes). Policies: full, recency (sliding window with sink), random, entropy-ranked patch eviction. Same budget interface for all.

## Phase 4: quantization (3 to 4 Nov)

Per-patch precision by entropy (high entropy keeps more bits) against uniform and random precision at matched bytes.

## Phase 5: main evaluation (4 to 7 Nov)

BPB against cache bytes curves on TinyStories and WikiText plus code validation, tuned on validation only. Three seeds for the two entropy-patched BLT models. Bootstrap confidence intervals over documents. Test split touched once per final configuration.

## Phase 6: ablations (7 to 8 Nov)

Own-entropy versus separate entropy model signal; patch eviction versus byte-level eviction; sink size; fixed-stride model with the same policies (does the entropy signal matter without entropy patching).

## Phase 7: error analysis (8 to 9 Nov)

Where eviction hurts: per-document gaps, per-domain (prose versus code), what kinds of bytes get evicted.

## Phase 8: demo (9 to 10 Nov)

Extend `scripts/demo.py`, `run.bat`, `run.sh` for CPU laptop from a clean clone after `fetch_assets.py`; new release assets; full rehearsal on a fresh unzip.

## Phase 9: comparative study and report (10 to 12 Nov)

Comparison table of all policies and baselines in one view (cache bytes against BPB, with CIs), written comparison against related work from NOVELTY.md, report sections, viva notes collated from `docs/`.

## Risks

1. 2.3 incremental decoding at patch boundaries (see decision points).
2. One GPU job at a time (half a GPU, 2-day wall time, queueing from other users). Phase 2.4 is the long pole: six trainings in five days. If queue time is long, drop the second entropy model and reuse the TinyStories one, or cut steps.
3. Phases 3 to 9 get 11 days with writing included. If Phase 2 slips past 31 Oct, cut Phase 4 (quantization) first, then Phase 6.
4. The core result may be negative. The plan does not depend on a positive result.

## Rules

Configs not hard-coded values; fixed seeds; every reported number regenerates from one command; tune on validation only; never overwrite `checkpoints/` or Part 2 results; stop and give options if a block passes 1.5 times its estimate.
