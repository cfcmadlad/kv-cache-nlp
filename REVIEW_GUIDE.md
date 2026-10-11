# Review guide

Preparation notes for the Part 2 code review: the pipeline in brief, where each important idea
lives in the code, the questions an examiner is most likely to ask about each file with short
answers, and a suggested split of the files across the five of us.

## The pipeline in 10 lines

1. `prepare_data.py` downloads TinyStories and WikiText-103 from HuggingFace.
2. It rebuilds WikiText articles from `= Title =` headers and undoes the `@-@ @,@ @.@` placeholders.
3. It applies Unicode NFC, trims whitespace, drops exact duplicates (SHA-256), writes `data/processed/<dataset>/<split>.jsonl`.
4. `train_ngram.py` counts byte 4-grams on the first N training documents (`models/ngram.py`).
5. It derives Kneser-Ney continuation counts and fits one discount per order from count-of-counts.
6. `train_bpe_transformer.py` trains an 8,000-token byte-level BPE tokenizer on the training split.
7. It packs all documents into one token stream and trains a 13.8M-parameter GPT for 3,000 AdamW steps.
8. Both models are scored in bits per byte on whole documents; the Transformer scores long documents in 256-token chunks.
9. `bpb_harness.py` tabulates the results files; `error_analysis.py` compares the models document by document.
10. `demo.py` trains live, shows the table, and scores and continues text; `check_setup.py --full` re-derives the headline numbers.

## Where the key ideas live

| Idea | File | Function or class |
|---|---|---|
| Bits per byte, n-gram | `models/ngram.py` | `KneserNeyByteLM.bits_per_byte`, `log2_prob_doc` |
| Bits per byte, Transformer | `scripts/train_bpe_transformer.py` | `bits_per_byte`, `per_document_bpb` |
| Byte counting per chunk (character offsets, not decoding) | `scripts/train_bpe_transformer.py` | `_doc_chunks` |
| Batched, masked chunk scoring | `scripts/train_bpe_transformer.py` | `_score_chunks` |
| Kneser-Ney interpolation formula | `models/ngram.py` | `KneserNeyByteLM._prob` |
| Continuation counts | `models/ngram.py` | `KneserNeyByteLM.fit` (second loop) and `_table` |
| Discount fitting | `models/ngram.py` | `KneserNeyByteLM._fit_discounts` |
| Causal attention mask | `models/transformer.py` | `CausalSelfAttention.forward` (`is_causal=True`) |
| Keys and values (what a KV cache would store) | `models/transformer.py` | `CausalSelfAttention.forward`: `k`, `v` from `self.qkv` |
| Generation without a KV cache (recomputes the whole context each step) | `models/transformer.py` | `GPT.generate` |
| Context window limit | `models/transformer.py` | `GPT.forward` assert and `GPT.generate` (`idx[:, -self.block_size:]`) |
| Weight tying | `models/transformer.py` | `GPT.__init__` (`self.head.weight = self.tok_emb.weight`) |
| Demo output isolation | `scripts/demo.py` | `DEMO_CHECKPOINTS_DIR`, `train_live` |
| Training loop (AdamW, warm-up, optional clipping) | `scripts/train_bpe_transformer.py` | `train` |
| Tokenizer reuse and naming | `scripts/train_bpe_transformer.py` | `load_tokenizer`, `get_or_train_tokenizer` |
| Guard against the truncated release file | `scripts/data_utils.py` | `check_not_truncated` |

KV-cache relevance, in one paragraph: in `CausalSelfAttention.forward`, every position's key and
value vectors are computed from `self.qkv`. During generation, the keys and values of earlier
positions never change, so a real system stores them (the KV cache) instead of recomputing them.
Our `GPT.generate` deliberately does not cache; it reruns the full forward pass over up to 256
tokens for every new token. The cache would hold `2 x n_layer x context x n_embd` numbers per
sequence (2 x 6 x 256 x 384 = 1.18M values here), and for a byte-level model the context is
about four times longer for the same text. Part 3 is about deciding which of those entries to keep.

## Showing that the results are real

Examiners may want evidence that the numbers came from running this code, not from a generated
write-up. Each of these takes under a minute and can be done live:

1. **Re-derive the headline numbers from the weights.** Demo option 4 (or
   `python scripts/check_setup.py --full`) loads the four released models and re-scores the evaluation documents from scratch. All six
   reported BPB values come out again to four decimals (about 30 seconds on a GPU, 2 minutes on a CPU).
2. **Regenerate a committed report.** Run `python scripts/error_analysis.py --dataset wikitext103`,
   which rewrites `eval/error_analysis_wikitext103.md` from the models, then `git status`: the file
   is unchanged, every number and example document included (about 40 seconds).
3. **Train live.** Demo option 1 runs the real training script: the loss falls step by step, and
   the n-gram prints the discounts it fits from the data.
4. **Change something and watch the result move.** Score a sentence in demo option 3, then the
   same sentence with a misspelt or unusual word: its BPB rises. Or train a lower-order n-gram,
   which needs fewer counts but predicts worse, and compare its BPB with the order-4 demo run:
   `python scripts/train_ngram.py --dataset tinystories --order 3 --max-train-docs 3000 --eval-max-docs 300 --out checkpoints/demo`.
   Always pass `--out checkpoints/demo` for experiments like this: the default output folder is
   `checkpoints/`, where a new run would replace or shadow the released models.
5. **Point at internal consistency.** The results files record token counts and timings that
   agree with the data: 6,539,016 TinyStories training tokens for 26.7 MB of text is 4.08 bytes
   per token, and 13,817,856 parameters follows from the architecture term by term (see below).

## Likely questions, file by file

### `scripts/prepare_data.py`

- **Why rebuild WikiText articles?** The raw export is one line of text per row, with `= Title =`
  rows marking where articles start. Treating lines as documents would cut articles into
  fragments. `load_wikitext103_split` starts a new document at each top-level header; section
  headers (`= = History = =`) do not match the regex and stay inside their article.
- **Why no lower-casing or stop-word removal?** We model the exact text. Any change to the text
  changes what is predicted and breaks bits per byte as a measure of the real text.
- **How is deduplication done?** SHA-256 of the normalised text, exact matches only, within each
  split (`dedup`). It removed 442 WikiText training documents and none elsewhere.

### `scripts/data_utils.py`

- **Why does importing it change `sys.path`?** Scripts in `scripts/` import `models.*`, which lives
  one folder up. Doing this once in the module every script already imports avoids repeating it.
- **How does it pick a checkpoint?** `_find_latest` globs a pattern such as
  `transformer_tinystories_*.pt` and takes the newest file. The tokenizer is found by the
  vocabulary size stored in the checkpoint's config.
- **What is `safe_print` for?** Windows consoles use a legacy codepage; printing WikiText's
  non-ASCII text crashed scripts. It replaces characters the console cannot show.

### `models/ngram.py`

- **What makes this Kneser-Ney and not plain backoff?** Lower orders use continuation counts, the
  number of distinct contexts a byte follows, rather than raw frequency. The top order uses raw
  counts. Interpolation mixes every order with weight `D * N(context) / c(context)`.
- **How are the discounts chosen?** `_fit_discounts`: `D = n1 / (n1 + 2 * n2)` per order from the
  number of (context, byte) pairs whose count in that order's table is exactly 1 and exactly 2,
  clamped to [0.1, 0.9].
- **Can a probability be zero?** No. The recursion ends in a uniform 1/256, so every byte has
  positive probability; the 1e-12 floor in `log2_prob_doc` is only a guard.

### `scripts/train_ngram.py`

- **Why only 25,000 and 4,500 training documents?** Exact counting in pure Python scales with bytes.
  4,500 WikiText documents are already 78 MB, more than the 22 MB of 25,000 TinyStories documents.
- **What is `train_bpb_sample`?** BPB on the first 200 training documents, as an overfitting check.
- **Why is TinyStories val BPB 1.8477 here but 1.8085 in the README table?** This script scored
  2,000 validation documents; the matched table uses the 500 the Transformer was scored on.

### `models/transformer.py`

- **Where is the causal mask?** `F.scaled_dot_product_attention(..., is_causal=True)` in
  `CausalSelfAttention.forward`; it masks the upper triangle so position t attends to positions up
  to t only.
- **Why weight tying, and why std 0.02 initialisation?** Tying saves 3.07M of 13.8M parameters and
  is standard for small LMs. PyTorch's default embedding init (std 1.0) makes the tied output
  logits far too large at the start; std 0.02 starts the loss near ln(8000), as expected.
- **How do you get 13,817,856 parameters?** 3,072,000 token embedding + 98,304 positions + 6 blocks
  x 1,774,464 + 768 final LayerNorm. The head adds nothing because it is tied.

### `scripts/train_bpe_transformer.py`

- **Why bits per byte and not perplexity?** Perplexity depends on the vocabulary; a byte model and a
  BPE model are not comparable that way. Total bits divided by UTF-8 bytes is tokenizer-free.
- **How do you score documents longer than 256 tokens?** `_doc_chunks` splits each document into
  non-overlapping 256-token chunks and records the exact bytes each chunk covers using the
  tokenizer's character offsets; `_score_chunks` batches 32 chunks, masks the padding, and sums the
  bits. The first token of each chunk is only context: it is never predicted, but its bytes are
  counted. That lowers BPB by 0.37 to 0.55 percent through the byte count alone, and probably a
  little more in truth, since those tokens have the least context. It is documented, far smaller
  than the 35 percent gap between the models, and fixed in Part 3 (prepend `<|endoftext|>`, score
  with overlapping windows).
- **What does `--grad-clip 0` mean and why does it matter?** No gradient clipping. The reported
  models were trained before clipping was added; `--grad-clip 0` reproduces them to four decimals
  (0.6939).
  The default 1.0 gives 0.6826 on TinyStories, a different model.

### `scripts/bpb_harness.py`

- **What does it read?** Every `checkpoints/*_results.json`; both training scripts write the same
  flat keys, so the table is built the same way for both models.
- **What if two results exist for one model and dataset?** It uses the newest and prints a note.

### `scripts/error_analysis.py`

- **What does it show?** Per-document BPB statistics, the correlation between the models' scores
  (0.74 TinyStories, 0.73 WikiText), BPB by length tercile, and the best, worst and most
  disputed documents.
- **What does the high correlation mean?** Both models find the same documents hard; difficulty is
  mostly in the text (rare words). The Transformer is better on every single document analysed: its
  smallest per-document gain is 0.74 bpb on TinyStories and 0.69 bpb on WikiText (bottom of each
  `eval/error_analysis_*.md`).

### `scripts/demo.py`

- **Is the live training real?** Yes, it runs the same training scripts as a subprocess with
  smaller settings (3,000 documents, 300 steps, vocabulary 2,000 for the Transformer).
- **What does `run.bat` / `run.sh` do?** Finds Python 3.10+ (installing it if missing). If that
  Python already has PyTorch it is used as is; otherwise one shared environment is created in the
  user folder and PyTorch (GPU build if `nvidia-smi` works, else CPU) and `requirements.txt` are
  installed into it, once. Then it starts the demo, which fetches the models and data on its first
  run. The submitted code has no data, so this is how a fresh copy becomes runnable.
- **Can the demo damage the results?** No. Live runs write to `checkpoints/demo/`; options 2 and 3
  only read.
- **Why does a short phrase get an absurdly low BPB in option 3?** The Transformer never predicts
  the first token of its input, but that token's bytes still count. On "Once upon a time" that
  free token is a quarter of the bytes, giving 0.010 BPB. Use full sentences in the demo; on the
  evaluation sets the effect is under 0.6 percent.

### `scripts/fetch_assets.py` and `scripts/check_setup.py`

- **How do we know the downloaded models are the reported ones?** The zip's SHA-256 is checked
  against a constant, and `check_setup.py --full` recomputes all six headline BPB values to four
  decimals.
- **Why is the WikiText training file truncated in the release?** Only the first 3,000 documents are
  ever read without retraining, and `read_jsonl_texts` stops reading at that count, so the result is
  identical while the download is 138 MB instead of about 300 MB.

## Suggested split across five members

| Member | Files | Topics to own |
|---|---|---|
| 1 | `scripts/prepare_data.py`, `scripts/data_utils.py` | datasets, cleaning, dedup, statistics, why these corpora |
| 2 | `models/ngram.py`, `scripts/train_ngram.py` | Kneser-Ney, continuation counts, discounts, training-size choice |
| 3 | `models/transformer.py` | attention and causal mask, block structure, tying, parameter count, KV-cache link to Part 3 |
| 4 | `scripts/train_bpe_transformer.py` | tokenizer, training loop and settings, gradient clipping, BPB and chunked scoring |
| 5 | `scripts/bpb_harness.py`, `scripts/error_analysis.py`, `scripts/demo.py`, `scripts/fetch_assets.py`, `scripts/check_setup.py` | results, error analysis, running the demo, setup |
