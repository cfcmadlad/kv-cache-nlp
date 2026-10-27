# Viva demo script (10 to 12 minutes)

Uses `scripts/demo.py` options 6, 4, 7 and 5. All outputs below were produced on CPU (4 cores, no GPU) on 27 Oct 2026 from a fresh clone of commit 84f9b5d, with the release assets downloaded by the demo itself. Numbers will differ slightly on another CPU or PyTorch build; the ranking of policies on a single text is an anecdote, not the result (see "What to say about single runs").

## Running it on a desktop

Windows: double-click `run.bat`. Linux or macOS: `./run.sh`. Both find Python 3.10 or newer (installing it if missing), reuse an existing PyTorch or build a private environment in `%LOCALAPPDATA%\kv-cache-nlp\venv` (`~/.cache/kv-cache-nlp/venv`), install the requirements and start the menu. The first start needs internet (PyTorch about 200 MB for CPU, plus 138 MB and 124 MB of release assets). Before the viva run `python scripts/check_demo.py`: it checks Python, packages, assets, a forward pass and the option 4 and option 6 code paths in about 5 seconds and prints `[ok]` or `[FAIL]` per item.

Verified on 27 Oct 2026 (Linux, 4 CPU cores, no GPU): a fresh clone with no PyTorch in the system Python ran `run.sh`, built the private environment (about 2 minutes, first time only) and completed option 4; options 1, 2, 3, 4, 5, 6 and 7 all ran from the menu. Not verified by me: `run.bat` on Windows and the launcher on macOS (`CHALLENGES.md` records a Part 2 rehearsal from a fresh unzip that passed). Test both on the machine you will present from. The menu now rejects cache budgets below 10 patches and returns to the menu if an option fails, because a budget of 9 or less used to stop the demo with a traceback.

## Before the viva (5 minutes, once)

1. `pip install torch --index-url https://download.pytorch.org/whl/cpu` and `pip install -r requirements.txt` (or run `run.bat` / `./run.sh`, which set Python up). The demo itself does not install packages.
2. `python scripts/demo.py`, enter `8`. The first run downloads the Part 2 assets (138 MB) and, on the first use of option 4 or 7, the Part 3 models (124 MB). No internet on the day: place `kv-cache-nlp-part2-assets.zip` and `kv-cache-nlp-part3-assets.zip` in the repo folder, its parent or `~/Downloads`; the demo looks there first.
3. Close other heavy programs. With another CPU-heavy job running, option 7 took more than 10 minutes instead of 36 seconds in our test (thread contention on 4 cores).
4. Have `docs/results_summary.md` and `report/main.pdf` open in another window.

## Timeline

| Time | Step | What you do |
|---|---|---|
| 0:00 to 1:00 | Framing | Problem (byte models need about 4 times more KV cache), idea (keep high-entropy patches), outcome (negative; attention and quantization work). |
| 1:00 to 3:30 | Option 6, live training | Start it first, talk while it trains. |
| 3:30 to 6:00 | Option 4, compress a text | Paste a sentence chosen by the examiner. |
| 6:00 to 8:00 | Option 7, stream with real eviction | Memory and time on a 4,000-byte text. |
| 8:00 to 10:00 | Option 5, results | Scroll to the headline tables. |
| 10:00 to 12:00 | Questions | `docs/viva_notes.md`, `docs/error_analysis_part3.md`. |

## Option 6: train a small byte-level model live (about 80 seconds on 4 cores)

Input: `6`, `tinystories`, `300`, `16`.

Say: a 1.84M-parameter byte-level model with fixed 4-byte patches, trained from scratch on 1.3 MB of stories; the loss falls from 8 bits per byte (random guess) as it learns.

Expected (the loss values were identical in two runs, so seeding works):

```
1.84M parameters, 1,302,911 training bytes, 300 steps of 16 x 512 bytes
step    1  loss 5.557 nats  = 8.017 bits per byte  (1s)
step  100  loss 2.007 nats  = 2.896 bits per byte  (25s)
step  200  loss 1.631 nats  = 2.353 bits per byte  (49s)
step  300  loss 1.532 nats  = 2.211 bits per byte  (74s)

Cache setting                            BPB   Mean KV cache
full cache                             2.147        201.9 KB
keep 16 patches: recency               2.146         35.2 KB
keep 16 patches: random                2.148         35.2 KB
keep 16 patches: entropy               2.147         35.2 KB
```

Say: the policies are indistinguishable here, and the demo prints why: a model this small and this briefly trained barely uses its global cache. This is the honest result of a 2-minute model. The fully trained models in options 4 and 7 are the evidence.

Fallbacks: if the machine is slow, enter `150` steps (about 40 seconds). If it fails, skip to option 4 and say option 6 was verified on a fresh clone (this file).

## Option 4: compress the KV cache of a text (2 to 5 seconds)

Input: `4`, `tinystories`, then paste a sentence (or press Enter for a validation story), then `32`.

Expected for the default story (1,201 bytes, 298 patches, 4.0 bytes per patch):

```
Once upon a time|, in a |big |forest, there li|ved a |r|hinoceros |named |R|o|xy. |Ro|x|y |loved to |c|limb. ...

Cache setting                            BPB   Mean KV cache
full cache (fp16)                      0.734       1377.8 KB
keep 32 patches: recency               0.744        279.6 KB
keep 32 patches: random                0.746        279.6 KB
keep 32 patches: entropy               0.751        279.6 KB
keep 32 patches: entropy_inv           0.743        279.6 KB
uniform 4-bit                          0.735        366.0 KB
uniform 3-bit                          0.744        279.9 KB
```

With `wikitext103`, Enter, `64` (4,096 bytes, 927 patches, 4.4 bytes per patch): full 1.576 (4276.2 KB), recency 1.580, random 1.582, entropy 1.588, entropy_inv 1.586 (all 569.8 KB), uniform 4-bit 1.577 (1135.9 KB), uniform 3-bit 1.580 (868.6 KB).

Say: boundaries fall before surprising letters (`c|limb`, `R|o|xy`), which is how entropy patching works. Uniform 4-bit costs 0.001 BPB at 27 percent of the bytes; eviction to 32 patches costs 0.01 at 20 percent. Keeping high-entropy patches (`entropy`) is the worst eviction rule, as in the paper. Point to `docs/error_analysis_part3.md`: the entropy rule keeps one-byte fragments.

Caution on very short texts: a pasted 94-byte sentence (30 patches, budget 16) gave full 1.276, recency 1.280, random 1.274, entropy 1.268, entropy_inv 1.269. Differences of 0.01 BPB on one short text are noise; do not read a policy ranking from them.

## Option 7: stream a long text with a real fixed-size cache (about 36 seconds)

Input: `7`, `tinystories`, Enter, `4000`, `64`.

Expected (default text is 40 validation stories joined by spaces, 978 patches):

```
4000 bytes in 978 patches, decoded one byte at a time with real cache eviction.

Cache setting                            BPB  Final KV cache   Seconds
full cache                             0.890       9004.0 KB      9.1
keep 64 patches: recency               0.844        589.8 KB      8.4
keep 64 patches: random                0.870        589.8 KB      8.9
keep 64 patches: entropy               0.862        589.8 KB      8.7
```

With `wikitext103`: 940 patches, full 1.722 (8653.8 KB), recency 1.729, random 1.738, entropy 1.756 (all 589.8 KB), about 8 seconds per policy.

Say: the cache is really bounded (590 KB against 9 MB at 4,000 bytes, and the gap grows with length: `results/part3/deploy_bench.md` measured 136.5 MB against 0.59 MB at 65,536 bytes on a GPU). On TinyStories recency is better than the full cache. We think this is because the demo text joins unrelated stories so old stories are noise; we have not tested that. The paper's version of this finding is in `results/part3/stream_16k.md` and `results/part3/pg19_stream.md`.

## Option 5: results (print takes a few seconds, 250 lines)

Input: `5`. It prints `docs/results_summary.md`; scroll to the top.

Show, in this order: the gate table (BLT does not beat the BPE Transformer at equal training); the eviction table (keeping high entropy is worst); the 10x-training table (penalty grows from +0.002 to +0.029); the combination table (7.4 percent of the cache for +0.0012 BPB); the PG-19 table. If the printout is too long, open `docs/results_summary.md` instead.

## What to say about single runs

Examiners may point at a line where entropy beats random (it does in option 7 on TinyStories: 0.862 against 0.870). Answer: one text of 4,000 bytes is an anecdote. The evidence is 72 paired comparisons over three seeds, four settings and two budgets, where keeping high entropy is worse than random in 68 and better in none (`results/part3/seeds.md`), plus the error analysis (`docs/error_analysis_part3.md`).

## Test record and failures

Run from a fresh clone (`git clone`, then `python scripts/demo.py` with stdin choices): options 4, 6 and 7 completed with no error and no change to the repository. No failure was found. What was not tested: Windows (`run.bat`), a machine without Python or PyTorch, and a GPU. Observed issues, not failures: the first run needs about 262 MB of downloads; the demo needs `torch`, `numpy`, `tokenizers` and `tqdm` installed beforehand; option 7 slows by more than an order of magnitude when another CPU-heavy process shares the cores.

Rehearsal without typing (CPU, about 2 minutes in total, three options in one go):

```bash
printf '4\ntinystories\n\n32\n6\ntinystories\n300\n16\n7\ntinystories\n\n4000\n64\n8\n' | python scripts/demo.py
```
