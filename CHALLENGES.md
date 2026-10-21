# Challenges

Each entry: problem, impact, strategy, outcome. Entries marked "to fill" need information I cannot see; the questions are at the end.

## Part 1

To fill. No Part 1 material was available to me.

## Part 2

### Reported BPB was slightly optimistic

Problem: the first token of every 256-token chunk is context only, but its bytes were still counted in the denominator.
Impact: transformer BPB is 0.4 to 0.5 percent too low on these sets (much more on very short inputs). Corrected values are 0.6978 (TinyStories val), 1.5257 (WikiText val), 1.5370 (WikiText test).
Strategy: kept the reported numbers unchanged for traceability and documented the correction in the README; Part 3 scoring counts every byte it scores.
Outcome: disclosed; Part 3 comparisons use corrected scoring.

### Training without gradient clipping

Problem: the two reported checkpoints were trained before clipping was added.
Impact: a rerun with clipping 1.0 gives 0.6826 on TinyStories instead of 0.6939, so the baseline is slightly weaker than it could be.
Strategy: added a `--grad-clip` flag where 0 reproduces the reported checkpoints exactly; the reported runs keep clip 0.
Outcome: reported numbers reproduce; the weaker baseline favours the Part 3 model if anything, noted in the report.

### Building WikiText-103 documents

Problem: the raw line export has no document boundaries and uses placeholder tokens such as `@-@`.
Impact: wrong text would distort byte counts and BPB.
Strategy: rebuilt articles from the line export, restored placeholders, NFC normalisation, exact deduplication.
Outcome: documents usable; byte counts recorded in stats.json.

### Release size limit

Problem: the full WikiText-103 train set is too large for the release bundle.
Impact: demo training on WikiText would not match the original run.
Strategy: the release copy cuts train to 3,000 documents (138 MB zip, SHA-256 checked); the full corpus rebuilds with `prepare_data.py`.
Outcome: demo works offline from the bundle.

### Demo robustness

Problem: the examiner runs our code from LMS on an unknown machine.
Impact: a missing Python or PyTorch would fail the demo.
Strategy: `run.bat` and `run.sh` find or install Python, reuse an existing PyTorch or build a shared venv; `demo.py` fetches assets; loading hardened (restricted unpickler, `weights_only=True`).
Outcome: full offline rehearsal from a fresh unzip passed. Norton Antivirus killed or quarantined installer-starting scripts and test wrappers during testing.

### Cluster access

Problem: on 2026-09-21 the QOS had MaxSubmitJobsPerUser=0, so no jobs could be submitted.
Impact: Part 2 training ran on a laptop RTX 4060.
Strategy: on 2026-10-09 we found `sbatch --test-only` succeeds once the job requests `--gres=mps:50`; limits are 4 CPUs, half a GPU, 80 GB RAM, 2-day wall time.
Outcome: cluster usable for Part 3.

## Part 3

### Cluster limits shape the design (Block 2.0)

Problem: one job at a time, 2-day wall time, queueing behind other users, shared CUDA modules only up to 12.1.
Impact: a lost or timed-out job could waste days.
Strategy: every run checkpoints periodically, handles SIGUSR1 five minutes before the time limit, requeues itself, and resumes automatically; PyTorch installs from a cu128 wheel in a home venv.
Outcome: resume verified locally by test (killed run matches an uninterrupted run to 1e-6). Not yet verified on the cluster.

### Code data is gated (Block 2.1)

Problem: `bigcode/the-stack-smol` needs a Hugging Face login with the dataset terms accepted; this machine has no token.
Impact: the code split cannot be built or checked on real data yet. The splitting and verification logic is tested on synthetic files only.
Strategy: `prepare_code.py` takes the repository name and a per-language cap so a small trial can run first.
Outcome: open; needs one team member to run `hf auth login` after accepting the terms.

### Entropy model BPB came out below the planned range (Block 2.2)

Problem: the planned plausible range was 0.9 to 1.3 (TinyStories) and 1.7 to 2.2 (WikiText); we measured 0.8438 and 1.6983.
Impact: a BPB that looks too good can mean a leak. Nothing else would catch it, because the model is also used to place patch boundaries.
Strategy: causality tests (changing later bytes never changes earlier logits, windows limit reach, documents isolated), and a sanity bound: the 13.8M BPE transformer scores 0.694 and 1.520 on the same documents, the 4-gram 1.81 and 2.41.
Outcome: no leak found; the estimate was conservative. Reported as measured.

### Training and evaluation must see the same patches (Block 2.3)

Problem: training samples windows from the middle of documents, while evaluation scores whole documents from a BOS token.
Impact: a mismatch would make evaluation BPB look different from training loss for reasons unrelated to the model.
Strategy: documents are isolated by an attention mask built from BOS positions, patch boundaries come from a cached per-byte entropy array (so training and evaluation use identical boundaries), and evaluation scores 4096-byte chunks that start exactly like training windows.
Outcome: consistent; the cache matches direct computation to 1e-3 (float16 storage).

### A CUDA out-of-memory error silently corrupted a cache (Block 2.4)

Problem: the WikiText entropy cache crashed with CUDA out of memory while another job used the GPU, but the queue carried on and the partly written array sat on disk.
Impact: the entropy-patched WikiText model would have trained on zeros for part of the data.
Strategy: caught by reading the queue log; the cache is deleted and rebuilt, and the next queue stops if the cache script fails.
Outcome: fixed before use. Lesson: GPU jobs run one at a time.

### My speed estimate was wrong in both directions (Block 2.4)

Problem: a 60-step test suggested 0.3 s per step; while other GPU work ran, real steps took 1.5 to 2.4 s on the power-throttled laptop GPU.
Impact: wrong time estimates for the user's planning.
Strategy: measure with the GPU idle. Two TinyStories BLT runs of 3,000 steps took about 10 minutes each when run alone.
Outcome: the laptop is sufficient for the planned runs if one job runs at a time.

### Entropy-guided eviction did not beat the controls on TinyStories (Phase 3)

Finding: keeping high-entropy patches is worse than random and than recency at every budget from 12 to 128 patches (paired differences +0.003 to +0.015 BPB, intervals exclude zero). Keeping low-entropy patches is level with random and slightly worse than recency. Even a 12-patch cache costs only 0.04 BPB over a full cache of about 200 patches, so there is little to gain on documents this short.
Impact: the core claim is not supported on this dataset. WikiText (longer documents) is the real test.
Strategy: report as measured; check WikiText, other aggregations (max) and the fixed-stride model before drawing any conclusion.

## Questions for you

1. Part 1: what were the main problems, and what did you do about them?
2. Any problem from Part 2 that I have missed (data, time, team, tooling)?
3. Release size limit: is 3,000 documents the real reason, or was it a GitHub limit?

### Seed variance is the size of some effects

Problem: the first extra seed of the TinyStories entropy-patched model has full-cache BPB 0.8125 against 0.7980 for seed 0.
Impact: the 3.3 percent gap between entropy patching and fixed stride in `docs/results_summary.md` is within about two seed spreads and is not established. The eviction sign comparisons are paired on the same model and are not affected.
Strategy: three seeds for the entropy-patched models and a stride-5 control; report per-seed numbers.
Outcome: in progress.
