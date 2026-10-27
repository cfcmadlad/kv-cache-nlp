# The demonstration: a script of ten to twelve minutes

The demonstration employs `scripts/demo.py`, options 6, 4, 7 and 5, in that order. Every output printed below was produced on CPU (four cores, no GPU) on 27 Oct 2026 from a fresh clone, the release assets having been fetched by the demo itself. Figures will deviate slightly on another processor or PyTorch build; and the ranking of policies on a single text is an anecdote, not a result (see "On single runs", below).

## Arranging it on a desktop

On Windows, double-click `run.bat`; on Linux or macOS, run `./run.sh`. Each launcher locates Python 3.10 or newer (installing it if absent), reuses an existing PyTorch or builds a private environment (`%LOCALAPPDATA%\kv-cache-nlp\venv` on Windows, `~/.cache/kv-cache-nlp/venv` elsewhere), installs the requirements and opens the menu. The first start needs the internet: PyTorch for CPU is about 200 MB, and the release assets are 138 MB and 124 MB. Before the viva, run `python scripts/check_demo.py`; in about five seconds it audits Python, the packages, the assets, a forward pass and the code paths of options 4 and 6, and prints `[ok]` or `[FAIL]` against each.

What was verified on 27 Oct 2026 (Linux, four CPU cores, no GPU): a fresh clone whose system Python lacked PyTorch ran `run.sh`, built the private environment (about two minutes, once only) and completed option 4; options 1 to 7 all ran from the menu. What was not verified: `run.bat` on Windows and the launcher on macOS (`CHALLENGES.md` records a Part 2 rehearsal from a fresh unzip that passed). Test both on the machine from which you will present. The menu now refuses cache budgets below ten patches and returns to the menu should an option fail, since a budget of nine or less formerly ended the demonstration with a traceback.

## Before the viva (five minutes, once)

1. `pip install torch --index-url https://download.pytorch.org/whl/cpu` and `pip install -r requirements.txt`, or simply run `run.bat` or `./run.sh`, which attend to Python. The demo itself installs no packages.
2. `python scripts/demo.py`, then enter `8`. The first run fetches the Part 2 assets (138 MB) and, on first use of option 4 or 7, the Part 3 models (124 MB). With no connection on the day, place `kv-cache-nlp-part2-assets.zip` and `kv-cache-nlp-part3-assets.zip` in the repository folder, its parent or `~/Downloads`; the demo looks there first.
3. Close every other heavy program. With a competing CPU-bound job, option 7 took more than ten minutes instead of 36 seconds (thread contention on four cores).
4. Keep `docs/results_summary.md` and `report/main.pdf` open in another window.

## The order of proceedings

| Time | Step | What you do |
|---|---|---|
| 0:00 to 1:00 | Framing | The problem (byte models need about four times the KV cache), the idea (keep high-entropy patches), the outcome (negative; attention and quantization succeed). |
| 1:00 to 3:30 | Option 6, live training | Start it first and talk while it trains. |
| 3:30 to 6:00 | Option 4, compress a text | Paste a sentence of the examiner's choosing. |
| 6:00 to 8:00 | Option 7, stream with real eviction | Memory and time on a 4,000-byte text. |
| 8:00 to 10:00 | Option 5, results | Scroll to the headline tables. |
| 10:00 to 12:00 | Questions | `docs/viva_notes.md`, `docs/error_analysis_part3.md`. |

## Option 6: a small byte-level model, trained before their eyes (about 80 seconds on four cores)

Input: `6`, `tinystories`, `300`, `16`.

Say: a 1.84M-parameter byte-level model with fixed four-byte patches, trained from nothing on 1.3 MB of stories; the loss descends from 8 bits per byte (a blind guess) as it learns.

Expected (the loss values were identical in two runs, so the seeding holds):

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

Say: the policies are indistinguishable here, and the demo itself says why: a model so small and so briefly trained scarcely consults its global cache. That is the honest yield of a two-minute model; the fully trained models of options 4 and 7 are the evidence.

Fallbacks: on a sluggish machine enter `150` steps (about 40 seconds). Should it fail outright, pass to option 4 and mention that option 6 was verified on a fresh clone (this file).

## Option 4: compressing the KV cache of a text (two to five seconds)

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

Say: the boundaries fall before surprising letters (`c|limb`, `R|o|xy`), which is how entropy patching works. Uniform 4-bit costs 0.001 BPB at 27 percent of the bytes; eviction to 32 patches costs 0.01 at 20 percent. Keeping high-entropy patches (`entropy`) is the worst eviction rule, as in the paper. Point to `docs/error_analysis_part3.md`: the entropy rule hoards one-byte fragments.

A caution on very short texts: a pasted sentence of 94 bytes (30 patches, budget 16) gave full 1.276, recency 1.280, random 1.274, entropy 1.268, entropy_inv 1.269. Differences of 0.01 BPB on one brief text are noise; read no ranking into them.

## Option 7: a long text with a genuinely bounded cache (about 36 seconds)

Input: `7`, `tinystories`, Enter, `4000`, `64`.

Expected (the default text is 40 validation stories joined by spaces, 978 patches):

```
4000 bytes in 978 patches, decoded one byte at a time with real cache eviction.

Cache setting                            BPB  Final KV cache   Seconds
full cache                             0.890       9004.0 KB      9.1
keep 64 patches: recency               0.844        589.8 KB      8.4
keep 64 patches: random                0.870        589.8 KB      8.9
keep 64 patches: entropy               0.862        589.8 KB      8.7
```

With `wikitext103`: 940 patches, full 1.722 (8653.8 KB), recency 1.729, random 1.738, entropy 1.756 (all 589.8 KB), about eight seconds per policy.

Say: the cache is truly bounded (590 KB against 9 MB at 4,000 bytes, and the disparity widens with length: `results/part3/deploy_bench.md` measured 136.5 MB against 0.59 MB at 65,536 bytes on a GPU). On TinyStories recency beats the full cache; we suspect that this is because the demo text joins unrelated stories, whose older members are mere noise, but we have not tested the suspicion. The report's version of the finding is in `results/part3/stream_16k.md` and `results/part3/pg19_stream.md`.

## Option 5: the results (a few seconds to print, 250 lines)

Input: `5`. It prints `docs/results_summary.md`; scroll to the top.

Show, in this order: the gate table (the BLT does not beat the BPE Transformer at equal training); the eviction table (keeping high entropy is worst); the tenfold-training table (the penalty grows from +0.002 to +0.029); the combination table (7.4 percent of the cache for +0.0012 BPB); the PG-19 table. If the printout proves unwieldy, open `docs/results_summary.md` instead.

## On single runs

An examiner may point to a line where entropy beats random (it does in option 7 on TinyStories: 0.862 against 0.870). Answer: one text of 4,000 bytes is an anecdote. The evidence is 72 paired comparisons over three seeds, four settings and two budgets, in which keeping high entropy is worse than random in 68 and better in none (`results/part3/seeds.md`), together with the error analysis (`docs/error_analysis_part3.md`).

## The record of testing, and of failure

From a fresh clone (`git clone`, then `python scripts/demo.py` fed by stdin), options 1 to 7 completed without error and without altering the repository; the launcher `run.sh` also completed on a machine whose Python lacked PyTorch. No failure remained after the fix to cache-budget validation. Untested: Windows (`run.bat`), macOS, and a GPU. Observed frictions, which are not failures: the first run needs about 262 MB of downloads; and option 7 slows by more than an order of magnitude when another CPU-bound process shares the cores.

A rehearsal without typing (CPU, about two minutes, three options in one breath):

```bash
printf '4\ntinystories\n\n32\n6\ntinystories\n300\n16\n7\ntinystories\n\n4000\n64\n8\n' | python scripts/demo.py
```
