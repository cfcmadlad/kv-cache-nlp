# kv-cache-nlp

*Patch-aware compression of the key-value cache in byte-level language models.*
CS F429 Natural Language Processing, BITS Pilani Hyderabad Campus. Group 12.

A language model that consumes raw bytes dispenses with the tokenizer, that fastidious intermediary which stumbles over every word it has not previously met. The dispensation is not free. A sentence of a hundred subword tokens swells, when spelled in bytes, to something near four hundred positions, and the memory in which a decoder hoards its keys and values (the KV cache, the true bottleneck of deployment) swells in lockstep. The Byte Latent Transformer softens the blow by gathering bytes into patches, one cache entry apiece, and it does so by consulting the entropy of the next byte, a quantity that is computed and then, with a certain prodigality, thrown away.

This repository asks whether that discarded entropy deserves a second career as a verdict on which cache entries to keep, and answers with a small BLT-style model, a half-dozen eviction and quantization policies, three seeds of everything, and a conscience. The answer is a plain no, delivered with some precision; the same apparatus, however, uncovered four results of real and practical value (below). Part 2 supplied the data pipeline and two baselines; Part 3, which this repository chiefly documents, supplies the byte model, the cache policies and their trial.

## Findings in brief

Small models (13.3M parameters). BPB is bits per byte; lower is better. The full ledger is `docs/results_summary.md`, the tables are in `results/part3/`, and the paper-style report is `report/main.pdf`.

| Finding | Figure | Evidence |
|---|---|---|
| Eviction combined with 4-bit quantization | 7.4 percent of the WikiText cache bytes for +0.0012 BPB (SnapKV-style, 128 patches) | `results/part3/deploy_combo.md` |
| SnapKV-style policy, scored online | Better than recency in 24 of 24 comparisons, by 0.0002 to 0.0077 BPB | `results/part3/online_policies.md` |
| Bounded-memory streaming | A fixed recency cache beats resetting the context in 3 of 3 seeds on long WikiText documents, and by 0.0086 BPB on 20 PG-19 books | `results/part3/stream_16k.md`, `pg19_stream.md` |
| Measured memory and speed | Global cache 0.59 MB against 136.5 MB at 65,536 bytes; 0.9 ms per byte against a climb to 2.0 ms | `results/part3/deploy_bench.md` |
| Uniform 4-bit keys and values | Within 0.0003 to 0.0014 BPB of fp16 at 27 percent of the bytes | `docs/results_summary.md` |
| Compute per byte (analytic) | 2.8 times fewer FLOPs than a byte-level Transformer of equal width, though more than the BPE Transformer | `results/part3/flops.md` |
| Entropy as the keep signal (the original hypothesis) | Refuted: worse than random in 68 and than recency in 70 of 72 comparisons, better in none; the penalty grows about tenfold with tenfold training | `results/part3/seeds.md`, `scale_and_tuning.md`, `docs/error_analysis.md` |

The caveats are as honest as the figures: the models are small, they do not retrieve (a planted-name probe could not tell any policy from another), the 16k and fixed-stride 30,000-step models have a single seed, and at equal training the BPE Transformer still outscores our byte model.

## Baselines (Part 2)

BPB on identical documents; lower is better.

| Evaluation set | Kneser-Ney byte 4-gram | BPE Transformer (13.8M) |
|---|---|---|
| TinyStories val, first 500 docs | 1.8085 | 0.6939 |
| WikiText-103 val, 60 docs | 2.4103 | 1.5200 |
| WikiText-103 test, 62 docs | 2.4381 | 1.5309 |

The Transformer has 6 layers, 6 heads, width 384, an 8k byte-level BPE vocabulary and a 256-token context, and was trained for 3,000 AdamW steps at lr 3e-4 without gradient clipping. The first token of every 256-token chunk serves as context only while its bytes are still counted, which flatters its BPB by 0.4 to 0.5 percent on these sets (far more on very short inputs); corrected values are in `docs/CHALLENGES.md`.

## Requirements

Python 3.10 or newer; PyTorch 2.x; `numpy`, `tokenizers`, `tqdm` (listed in `requirements.txt`; `pytest` in `requirements-dev.txt`). A GPU is not needed for the demo, the analyses or the figures. The training runs behind the tables used a laptop GPU (seed 0 of the prose-only models) and a cluster GPU (everything else). Rebuilding the report needs LaTeX (on Debian we installed `texlive-latex-recommended`, `texlive-latex-extra`, `texlive-fonts-recommended` and `texlive-science`).

## Running the trained models

This is all that is needed to see the final models at work; no GPU, no training and no data download beyond the two release archives.

```bash
git clone https://github.com/cfcmadlad/kv-cache-nlp.git && cd kv-cache-nlp
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/check_demo.py
```

On Windows you may instead double-click `run.bat`; on Linux or macOS run `./run.sh`. Either launcher finds Python (installing it if absent), reuses an existing PyTorch or builds a private environment, installs the requirements and opens the menu below. The first start needs the internet: the Part 2 assets (138 MB, release `v0.2-part2`) are fetched at once and the Part 3 models (124 MB, release `v0.3-part3`) on first use of options 4 and 7. With no connectivity, download `kv-cache-nlp-part2-assets.zip` and `kv-cache-nlp-part3-assets.zip` from the releases page beforehand and place them beside the repository, in its parent folder or in `~/Downloads`; the demo looks there first.

`scripts/check_demo.py` audits the interpreter, the packages, the assets, a forward pass and the code paths of options 4 and 6 in about five seconds, printing `[ok]` or `[FAIL]` for each.

### Scoring a text of your own

```bash
python scripts/infer.py --dataset wikitext103 --budget 32 --text "Any paragraph you like."
python scripts/infer.py --dataset tinystories --budget 16 --file story.txt --stream
```

`--dataset` selects the trained model (a TinyStories or a WikiText-103 byte model, 13.3M parameters), `--budget` the number of patches the global cache may hold (at least 10, because one sink patch and the eight newest are always kept), and `--stream` decodes one byte at a time with real eviction, so that the memory it reports is the memory the cache actually holds. The output lists the patch boundaries the model chose and, for the full cache and each policy (recency, random, keep-high-entropy, keep-low-entropy, uniform 4 and 3 bits), the bits per byte and the mean KV cache size. A single short text is an anecdote: differences of a few thousandths of a bit are noise, and the evidence is in `results/part3/`.

### The interactive demonstration

```bash
python scripts/demo.py
```

| Option | What it does | Typical time on 4 CPU cores |
|---|---|---|
| 1 | Train a Part 2 baseline live (n-gram or small Transformer) | seconds to minutes |
| 2 | Print the baseline comparison table | instant |
| 3 | Score and continue your own sentence with a baseline | seconds |
| 4 | Compress the KV cache of any text under each policy; shows patch boundaries, BPB and cache size | 2 to 5 s |
| 5 | Print the Part 3 results (`docs/results_summary.md`) | instant |
| 6 | Train a small byte-level model from scratch, live, and compare policies | about 80 s |
| 7 | Decode a long text one byte at a time with a real fixed-size cache | about 36 s |

The ten-minute running order for a presentation, with expected output for every option and the fallbacks, is `docs/demo_script.md`; the slides are `report/slides.pdf`.

## Reproducing the experiments

Assets and data:

```bash
python scripts/fetch_assets.py              # checkpoints and processed data, release v0.2-part2
python scripts/part3_assets.py fetch        # the four small Part 3 models, release v0.3-part3
```

The release copy of WikiText-103 train is cut to 3,000 documents to keep the download small (our choice, not a platform limit); rebuild the full corpus with `pip install datasets` and `python scripts/prepare_data.py --dataset all`. The code corpus (`bigcode/the-stack-smol`) is gated: accept its terms on Hugging Face and log in before `python scripts/prepare_code.py`. PG-19 is fetched by `scripts/campaign/run_pg19.sh` from the `emozilla/pg19` mirror.

Part 2 baselines:

```bash
python scripts/train_ngram.py --dataset tinystories --order 4 --max-train-docs 25000 --eval-max-docs 2000
python scripts/train_bpe_transformer.py --dataset tinystories --max-steps 3000 --grad-clip 0
python scripts/bpb_harness.py
python scripts/error_analysis.py --dataset wikitext103 --eval-max-docs 60
```

For WikiText-103 the reported runs used `--max-train-docs 4500 --eval-max-docs 60` for the n-gram, and `--max-steps 3000 --grad-clip 0` with the full training set for the Transformer.

Part 3 pipeline (runs are written to `runs/<id>/`, byte data to `data/bytes/`):

```bash
python scripts/prepare_bytes.py && python scripts/prepare_bytes.py --verify
python scripts/train_entropy.py configs/entropy/entropy_tinystories.json
python scripts/patch_entropy.py --run entropy_tinystories --max-docs 500
python scripts/cache_entropy.py --run entropy_tinystories
python scripts/train_blt.py configs/blt/blt_tinystories_entropy.json
python scripts/eval_policies.py --run blt_tinystories_entropy --max-docs 500 --budgets 12 16 32 64 128
python scripts/eval_quant.py --run blt_tinystories_entropy --max-docs 500 --hi-bits 4 --lo-bits 2
python scripts/results.py --phase 2
```

The full campaign (seeds, controls, ablations, scaling, deployment, streaming) was run on a SLURM cluster with `sbatch slurm/job.sbatch bash scripts/campaign/<name>.sh`; every script skips finished runs, so an interrupted chain resumes where it stopped. Edit the account, partition and QOS lines of `slurm/job.sbatch` and `slurm/train.sbatch` for your own site, and create the environment with `slurm/setup_env.sh` (a Python 3.11 virtual environment in `~/venvs/kvc` with the CUDA 12.8 build of PyTorch; override with `KVC_VENV` and `PYTHON`).

| Script (in `scripts/campaign/`) | Purpose |
|---|---|
| `run_seeds.sh`, `run_controls.sh`, `run_tsmix.sh`, `run_mix.sh` | Three seeds, stride controls, code-mix retraining |
| `run_ablations.sh`, `run_ablations_seeds.sh` | Scoring rules, own entropy, sink size, byte-level cache |
| `run_policies.sh`, `run_attn_policies.sh`, `run_online.sh` | Eviction policies, offline and online H2O and SnapKV-style |
| `run_code_eval.sh`, `run_test.sh` | Code domain; the WikiText test split (touched once) |
| `run_long.sh`, `run_scale.sh`, `run_scale2.sh` | 16,384-byte models; 30,000-step models and tuning |
| `run_deploy.sh`, `run_stream.sh`, `run_pg19.sh` | Measured decoding, quantization combined with eviction, planted names, streaming, PG-19 |
| `run_extra.sh` | Chains the extra runs |

Tables are regenerated from the run outputs:

```bash
python scripts/compare.py seeds --out results/part3/seeds.md
python scripts/compare.py ablations --out results/part3/ablations.md
python scripts/compare.py table --out results/part3/comparison.md
python scripts/compare.py code --out results/part3/code_domain.md
python scripts/compare.py attention --out results/part3/attention_policies.md
python scripts/compare.py long --out results/part3/long_context.md
python scripts/compare.py test --out results/part3/test_split.md
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

## Analyses, figures and documents

These need no GPU and no training runs, only the tables in `results/part3/` (and, for the first two commands, the fetched release models):

```bash
python scripts/error_analysis_part3.py      # entropy terciles, byte classes, kept patches (about 2 min, CPU)
python scripts/generation_divergence.py     # greedy continuations under each policy (about 3 min, CPU)
python scripts/flops.py                     # analytic FLOPs per byte, configs/analysis/flops.json
python report/make_figures.py               # every chart, parsed from the tables
python scripts/render_mermaid.py --chromium /path/to/chromium   # optional; needs `pip install playwright`
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd report && pdflatex supplement && pdflatex slides
python -m pytest                            # 71 tests
```

## Layout

```
run.bat, run.sh        one-step demo launchers (Windows, Linux and macOS)
requirements*.txt      runtime and test dependencies
models/                Kneser-Ney byte n-gram, GPT-style Transformer, entropy model, BLT-style model with incremental decoding, rotary embeddings
kvc/                   cache policies and masks, quantization, entropy patching, preemption-safe trainer, run logging, byte datasets
scripts/               data preparation, training, evaluation, comparison tables, analyses, the demo and infer.py
  campaign/            the shell chains that produced every table (run_*.sh)
configs/
  blt/, entropy/       one JSON per trained model
  analysis/            inputs of the analytic FLOPs and public-model comparisons
  ops/                 smoke test and the cluster requeue test
slurm/                 cluster submission and environment scripts
tests/                 71 tests: causality, incremental decoding equals full forward, every policy equals real eviction, resume equals uninterrupted run
results/               generated tables: part3/ for Part 3, phase2.md for the Part 2 baselines
report/                main.tex (the report), supplement.tex, slides.tex, make_figures.py, figures/, presentation/ (diagram sources and PNGs), built PDFs
docs/                  results ledger, error analysis, demo script, novelty, challenges, proposal against delivery, design/ (one note per component)
```

Git-ignored and fetched or generated on demand: `checkpoints/` and `data/processed/` (release `v0.2-part2`), `runs/` (trained models, release `v0.3-part3`, and the outputs of every run), `data/bytes/` (built by `scripts/prepare_bytes.py`).

## Data

TinyStories (30,000 train stories, 21,990 validation) and WikiText-103 (articles rebuilt from the raw line export, placeholder tokens such as `@-@` restored), Unicode NFC normalised, exact duplicates removed. Part 3 adds The Stack Smol for code (28,229 files) and PG-19 (the `emozilla/pg19` mirror of the test split, 100 books) for long-book streaming. Corpora keep the licences of their sources; consult them before redistributing any derived data.

## Where to read next

| Document | Subject |
|---|---|
| `report/main.pdf`, `report/supplement.pdf`, `report/slides.pdf` | The report (10 pages), its extra charts, the slides |
| `docs/results_summary.md` | Every finding with exact figures and caveats |
| `docs/error_analysis.md` | Where eviction hurts, with real examples |
| `docs/NOVELTY.md` | What is new, what is not, and the verified related work |
| `docs/CHALLENGES.md` | Twenty difficulties and what became of each |
| `docs/proposal_vs_delivered.md` | The Part 1 promises against the evidence |
| `docs/demo_script.md` | The running order of the demonstration |
| `docs/design/` | One note per component: datasets, entropy patching, the BLT model, cache policies, quantization, ablations, infrastructure |

## Troubleshooting

* *The demo cannot find the models.* Run it once with the internet available, or place the two release zips as described above; `python scripts/fetch_assets.py --force` re-downloads the first set.
* *Option 7 is glacial.* It is compute-bound; close other heavy programs (with a competing process it ran more than an order of magnitude slower).
* *Antivirus quarantines `run.bat`.* Allow it in the antivirus history, or install the requirements by hand as in the quick start.
* *`playwright` cannot launch a browser.* Pass an existing Chromium with `--chromium`; the PNGs for the slides are already committed.
* *Tests or scripts import errors.* Run them from the repository root so that `kvc`, `models` and `scripts` resolve.
