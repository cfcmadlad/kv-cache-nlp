# kv-cache-nlp

Patch-aware KV cache compression for byte-level language models.
CS F429 Natural Language Processing, BITS Pilani Hyderabad, Group 12.

Byte-level language models read raw UTF-8 and avoid tokenizer failure modes, but a sentence spans about four times as many positions as in BPE tokens, so the key-value (KV) cache, the main deployment bottleneck, grows about four times faster. The Byte Latent Transformer (BLT) groups bytes into patches using next-byte entropy, so the cache holds one entry per patch. This project builds a BLT-style model and evaluates patch-granular cache compression against recency (StreamingLLM), attention-score methods (H2O, SnapKV-style), random eviction and quantization at an equal number of cache bytes.

Part 2 (done) built the data pipeline and two baselines scored in bits per byte (BPB). Part 3 adds the byte-level model, the cache policies and their evaluation, with three seeds per setting, paired bootstrap intervals, a test split evaluated once, and tested code.

## Headline results (Part 3)

Small models (13.3M parameters). BPB is bits per byte, lower is better. Tables are in `results/part3/`; the full account is `docs/results_summary.md` and the paper-style report is `report/main.pdf`.

| Result | Number | Evidence |
|---|---|---|
| Eviction plus 4-bit quantization | 7.4 percent of the WikiText cache bytes for +0.0012 BPB (SnapKV-style, 128 patches) | `results/part3/deploy_combo.md` |
| SnapKV-style policy, scored online | Better than recency in 24 of 24 comparisons, by 0.0002 to 0.0077 BPB | `results/part3/online_policies.md` |
| Bounded-memory streaming | A fixed recency cache beats resetting the context in 3 of 3 seeds on long WikiText documents and by 0.0086 BPB on 20 PG-19 books | `results/part3/stream_16k.md`, `pg19_stream.md` |
| Measured memory and speed | Global cache 0.59 MB against 136.5 MB at 65,536 bytes; 0.9 ms per byte against a rise to 2.0 ms | `results/part3/deploy_bench.md` |
| Uniform 4-bit K and V | Within 0.0003 to 0.0014 BPB of fp16 at 27 percent of the bytes | `docs/results_summary.md` |
| Compute per byte (analytic) | BLT needs 2.8 times fewer FLOPs than a byte-level Transformer of the same width, but more than the BPE Transformer | `results/part3/flops.md` |
| Entropy-guided retention (the proposal) | Negative: worse than random in 68 and than recency in 70 of 72 comparisons, better in none; the penalty grows about tenfold with ten times more training | `results/part3/seeds.md`, `scale_and_tuning.md`, `docs/error_analysis_part3.md` |

See `docs/results_summary.md` for caveats: small models, no retrieval ability, one seed for the 16k and fixed-stride 30,000-step models.

## Baseline results

BPB on identical documents, lower is better.

| Evaluation set | Kneser-Ney byte 4-gram | BPE Transformer (13.8M) |
|---|---|---|
| TinyStories val, first 500 docs | 1.8085 | 0.6939 |
| WikiText-103 val, 60 docs | 2.4103 | 1.5200 |
| WikiText-103 test, 62 docs | 2.4381 | 1.5309 |

The Transformer is 6 layers, 6 heads, width 384, 8k byte-level BPE vocabulary, 256-token context, 3,000 AdamW steps at lr 3e-4 without gradient clipping. The first token of each 256-token chunk is context only while its bytes are counted, which makes its BPB 0.4 to 0.5 percent optimistic on these sets (much more on very short inputs).

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/fetch_assets.py
```

`fetch_assets.py` downloads the trained checkpoints and processed data from the `v0.2-part2` release (138 MB, SHA-256 checked) into `checkpoints/` and `data/processed/`. The release copy of WikiText-103 train is cut to 3,000 documents; rebuild the full corpus with `pip install datasets` and `python scripts/prepare_data.py --dataset all`. On Windows, `run.bat` (or `run.sh` elsewhere) sets up Python and starts the demo in one step.

## Usage

```bash
python scripts/demo.py
python scripts/train_ngram.py --dataset tinystories --order 4 --max-train-docs 25000 --eval-max-docs 2000
python scripts/train_bpe_transformer.py --dataset tinystories --max-steps 3000 --grad-clip 0
python scripts/bpb_harness.py
python scripts/error_analysis.py --dataset wikitext103 --eval-max-docs 60
```

For WikiText-103 the reported runs used `--max-train-docs 4500 --eval-max-docs 60` for the n-gram and `--max-steps 3000 --grad-clip 0` with the full training set for the Transformer.

## Layout

| Path | Contents |
|---|---|
| `models/ngram.py` | Interpolated Kneser-Ney byte n-gram |
| `models/transformer.py` | Decoder-only Transformer |
| `scripts/prepare_data.py` | Download, clean, deduplicate and split the corpora |
| `scripts/train_ngram.py`, `scripts/train_bpe_transformer.py` | Training and BPB evaluation |
| `scripts/bpb_harness.py` | Comparison table from `checkpoints/*_results.json` |
| `scripts/error_analysis.py` | Per-document comparison, written to `results/part2/` |
| `scripts/demo.py` | CLI demo: live training, results table, scoring and generation |
| `scripts/fetch_assets.py` | Release download and verified extraction |

## Data

TinyStories (30,000 train stories, 21,990 validation) and WikiText-103 (articles rebuilt from the raw line export, placeholder tokens such as `@-@` restored), Unicode NFC normalised, exact duplicates removed. Part 3 adds The Stack Smol for code and PG-19 (the `emozilla/pg19` mirror of the test split, 100 books) for long-book streaming; long contexts otherwise use the long WikiText-103 documents.

## Part 3

Byte-level BLT-style model with patch-granular KV cache policies. Runs write to `runs/<id>/`; byte data to `data/bytes/`.

```bash
python scripts/prepare_bytes.py && python scripts/prepare_bytes.py --verify
python scripts/train_entropy.py configs/entropy_tinystories.json
python scripts/patch_entropy.py --run entropy_tinystories --max-docs 500
python scripts/cache_entropy.py --run entropy_tinystories
python scripts/train_blt.py configs/blt_tinystories_entropy.json
python scripts/eval_policies.py --run blt_tinystories_entropy --max-docs 500 --budgets 12 16 32 64 128
python scripts/eval_quant.py --run blt_tinystories_entropy --max-docs 500 --hi-bits 4 --lo-bits 2
python scripts/results.py --phase 2
python -m pytest
```

Demo on a CPU laptop (option 4 of `scripts/demo.py` runs the policies and quantization on any text; option 6 trains a small byte-level model live in about two minutes and compares cache policies; option 7 decodes a long text one byte at a time with a real fixed-size cache and prints BPB, final cache size and time): `python scripts/fetch_assets.py`, then `run.bat` or `./run.sh`. Option 4 fetches the Part 3 models (124 MB, SHA-256 checked) from the `v0.3-part3` release on first use with `scripts/part3_assets.py fetch`; `scripts/part3_assets.py build` rebuilds the zip.

Seeds, control and ablations (run on the cluster with `sbatch slurm/job.sbatch bash scripts/<name>.sh`; each resumes finished runs):

```bash
bash scripts/run_seeds.sh
bash scripts/run_ablations.sh
bash scripts/run_policies.sh
bash scripts/run_code_eval.sh
python scripts/compare.py seeds --out results/part3/seeds.md
python scripts/compare.py ablations --out results/part3/ablations.md
python scripts/compare.py table --out results/part3/comparison.md
python scripts/compare.py code --out results/part3/code_domain.md
python scripts/collate_viva.py
```

Extra runs, all on the cluster (`bash scripts/run_tsmix.sh`, `run_controls.sh`, `run_long.sh`, `run_attn_policies.sh`, chained by `run_extra.sh`; the test split by `run_test.sh`):

```bash
python scripts/compare.py attention --out results/part3/attention_policies.md
python scripts/compare.py long --out results/part3/long_context.md
python scripts/compare.py test --out results/part3/test_split.md
```

Deployment-oriented runs (`run_deploy.sh`: measured decoding, eviction plus quantization, recency-protected entropy, planted-name recall; `run_online.sh`: online H2O and SnapKV; `run_ablations_seeds.sh`; `run_scale.sh` and `run_scale2.sh`: 30,000-step models and 16k tuning; `run_stream.sh`: bounded-memory streaming; `run_pg19.sh`: PG-19 download, entropy cache and byte-by-byte streaming on long books):

```bash
python scripts/compare_deploy.py online --out results/part3/online_policies.md
python scripts/compare_deploy.py combo --out results/part3/deploy_combo.md
python scripts/compare_deploy.py stream --out results/part3/stream_16k.md
python scripts/compare_deploy.py bench --out results/part3/deploy_bench.md
python scripts/compare_deploy.py hybrid --out results/part3/deploy_hybrid.md
python scripts/compare_deploy.py needle --out results/part3/deploy_needle.md
python scripts/compare_deploy.py ablseeds --out results/part3/ablations_seeds.md
python scripts/compare_deploy.py scale --out results/part3/scale_and_tuning.md
python scripts/compare_deploy.py memory --out results/part3/deploy_memory.md
python scripts/compare_deploy.py pg19 --out results/part3/pg19_stream.md
```

All experiments are done; tables are in `results/part3/` and the report draft is `docs/report_part3.md`. Details in `PLAN.md`.

Deliverables for the Part 3 submission:

```bash
python report/make_figures.py
python scripts/flops.py
python scripts/error_analysis_part3.py
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

```bash
python scripts/check_demo.py
python scripts/generation_divergence.py
python scripts/render_mermaid.py --chromium /path/to/chromium
cd report && pdflatex supplement && pdflatex slides
```

`check_demo.py` verifies Python, packages, assets and the demo code paths in seconds; run it on the presentation machine first.

The LaTeX report is `report/main.tex` (references in `report/references.bib`, figures in `report/figures/`, built `report/main.pdf`). The supplement (`report/supplement.pdf`), the slides (`report/slides.pdf`, notes in `docs/presentation_notes.md`) and Mermaid diagrams (`docs/presentation_diagrams.md`, PNGs in `report/presentation/mermaid/`) support the presentation; `docs/proposal_vs_delivered.md` maps the Part 1 proposal to the evidence. The error analysis is `docs/error_analysis_part3.md` (raw output `results/part3/error_analysis.md`), the analytic compute table is `results/part3/flops.md`, the demo rehearsal script is `docs/demo_script.md` and the viva questions by rubric item are in `docs/viva_by_rubric.md` (collated into `docs/viva_notes.md`).

Plan in `PLAN.md`, related work in `NOVELTY.md`, problems and fixes in `CHALLENGES.md`, per-block notes and results in `docs/`.
