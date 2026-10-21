# kv-cache-nlp

Patch-aware KV cache compression for byte-level language models.
CS F429 Natural Language Processing, BITS Pilani Hyderabad, Group 12.

Byte-level models read raw UTF-8, so a sentence spans about four times as many positions as in BPE tokens, and the KV cache grows with it. The Byte Latent Transformer (BLT) already computes next-byte entropy to place patch boundaries. This project tests whether that same entropy can decide which cache entries to evict or quantize, compared with recency (StreamingLLM), attention-score methods (H2O, SnapKV) and random eviction at an equal number of cache bytes.

Part 2 (done) built the data pipeline and two baselines scored in bits per byte (BPB). Part 3 (in progress) adds the byte-level model, the cache policies and their evaluation. Headline so far, on one seed and small models: using next-byte entropy as the keep signal is negative. Keeping high-entropy patches, or giving them more bits, does worse than random and recency at the same cache bytes, and uniform 4-bit KV is within 0.001 BPB of fp16 at 27 percent of the bytes. See `docs/results_summary.md`.

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

TinyStories (30,000 train stories, 21,990 validation) and WikiText-103 (articles rebuilt from the raw line export, placeholder tokens such as `@-@` restored), Unicode NFC normalised, exact duplicates removed. Part 3 adds PG-19 for long contexts and The Stack Smol for code.

## Part 3 (in progress)

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

Demo on a CPU laptop (option 4 of `scripts/demo.py` runs the policies and quantization on any text): `python scripts/fetch_assets.py`, then `run.bat` or `./run.sh`. Option 4 fetches the Part 3 models on first use with `scripts/part3_assets.py fetch`; the release asset `v0.3-part3` is built by `scripts/part3_assets.py build` and still has to be uploaded.

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

Seeds, ablations, the comparison table and the report text are done (cluster runs, `slurm/job.sbatch`); tables are in `results/part3/` and the report draft is `docs/report_part3.md`. The WikiText models were also retrained with a 15 percent code mix (`bash scripts/run_mix.sh`). Not done: the Block 2.0 requeue test, the test split. Details in `PLAN.md`.

Plan in `PLAN.md`, related work in `NOVELTY.md`, problems in `CHALLENGES.md`, per-block notes and results in `docs/`.
