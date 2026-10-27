# Challenges

Each entry: problem, impact, strategy, outcome.

## Part 1

No separate Part 1 problems were recorded. The team confirmed (27 Oct 2026) that the Part 2 entries below stand as the overall problems for Parts 1 and 2.

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

Problem: the full WikiText-103 train set makes a large release bundle and a slow first download for the demo. The team confirmed (27 Oct 2026) that the cut was our choice of bundle size, not a GitHub limit.
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
Strategy: every run checkpoints periodically, handles SIGUSR1 five minutes before the time limit, requeues itself and resumes; PyTorch installs from a cu128 wheel in a home venv; run scripts skip finished outputs so a failed chain restarts where it stopped.
Outcome: resume verified locally (killed run matches an uninterrupted run to 1e-6) and on the cluster. One job (416139) was signalled at step 1,864, requeued under the same id, resumed from 1,864 and finished at 24,000. Three earlier attempts to show this failed for my own reasons (a signalling script that never fired twice, and a reused run name that overwrote a smoke run's config); the docs record both.

### Code data is gated (Block 2.1)

Problem: `bigcode/the-stack-smol` needs a Hugging Face login with the dataset terms accepted.
Impact: the code split could not be built until a team member logged in on the cluster.
Strategy: `prepare_code.py` takes the repository name and a per-language cap; the splitting and verification logic was tested on synthetic files first.
Outcome: built (28,229 files, split and hash checks pass). A token pasted into chat was not used and must be revoked.

### Entropy model BPB came out below the planned range (Block 2.2)

Problem: the planned plausible range was 0.9 to 1.3 (TinyStories) and 1.7 to 2.2 (WikiText); we measured 0.8438 and 1.6983.
Impact: a BPB that looks too good can mean a leak, and the model also places patch boundaries.
Strategy: causality tests (changing later bytes never changes earlier logits, windows limit reach, documents isolated) and a sanity bound from the BPE transformer (0.694 and 1.520) and the 4-gram (1.81 and 2.41).
Outcome: no leak found; the estimate was conservative.

### Training and evaluation must see the same patches (Block 2.3)

Problem: training samples windows from the middle of documents, while evaluation scores whole documents from a BOS token.
Impact: a mismatch would make evaluation BPB differ from training loss for reasons unrelated to the model.
Strategy: documents are isolated by an attention mask built from BOS positions, patch boundaries come from a cached per-byte entropy array so training and evaluation use identical boundaries, and evaluation scores 4,096-byte chunks that start exactly like training windows.
Outcome: consistent; the cache matches direct computation to 1e-3 (float16 storage).

### A CUDA out-of-memory error silently corrupted a cache (Block 2.4)

Problem: the WikiText entropy cache crashed with out of memory while another job used the GPU, but the queue carried on and the partly written array stayed on disk.
Impact: the entropy-patched WikiText model would have trained on zeros for part of the data.
Strategy: caught by reading the queue log; the cache was deleted and rebuilt, and the next queue stops if the cache script fails.
Outcome: fixed before use. The same limit returned later: two 16,384-byte trainings (about 45 GB each) could not share one 95 GB GPU, so those runs went one at a time.

### My speed estimate was wrong in both directions (Block 2.4)

Problem: a 60-step test suggested 0.3 s per step on the laptop; real steps took 1.5 to 2.4 s while other GPU work ran on the power-throttled GPU.
Impact: wrong time estimates for planning.
Strategy: measure with the GPU idle. On the cluster a 3,000-step run takes about 90 seconds, which made ten times longer training cheap. The online evaluations I estimated at 7 hours took about 3.

### Seed variance is the size of some effects

Problem: seed 0 of TinyStories had full-cache BPB 0.7980 against 0.8544 for seed 2 (seed spread 0.056; 0.135 for fixed stride).
Impact: the 3.3 percent gap between entropy patching and fixed stride that I first reported was within seed noise; it was withdrawn.
Strategy: three seeds for every patching scheme in all four settings, a stride-5 control, and paired document bootstrap intervals so eviction comparisons on one model are not affected by seed.
Outcome: the eviction sign holds in every seed; patching comparisons are stated as likely, not established.

### Cache size accounting error in the policy evaluation

Problem: `eval_policies.py` counted a full-cache chunk as P entries but a masked chunk as about (P + 1) / 2, so full-cache bytes were doubled.
Impact: the first quoted shares of full cache bytes were half the true shares (32 and 15 percent instead of 16 and 7). BPB was not affected.
Strategy: found when the comparison table put eviction and quantization rows side by side and the full-cache sizes disagreed (1,669 KB against 839 KB). Fixed, reran the policy evaluations on the cluster and regenerated every table.
Outcome: corrected.

### Code dataset arrived late and the prose models cannot read it

Problem: the prose-trained models scored 7 to 8 BPB on code, so the first code evaluation said nothing.
Strategy: a mix option in the entropy and BLT trainers (15 percent of each batch from code), retraining on WikiText and then TinyStories.
Outcome: the mix costs about 0.05 BPB on WikiText and 0.09 on TinyStories; the eviction conclusion is unchanged but smaller on WikiText.

### Entropy-guided eviction did not beat the controls

Finding: keeping high-entropy patches is worse than random and than recency in 68 to 70 of 72 comparisons on three seeds, in the code-mix models, on the WikiText test split, with models trained ten times longer (where the penalty grows), at 16,384 bytes, and for a recency-protected variant.
Impact: the core claim of the proposal is not supported.
Strategy: report as measured, test the obvious escape routes (other aggregations, the model's own entropy, sink size, byte-level cache, more training, longer context, a protected recency window) and add an attention-based rule that does beat recency.
Outcome: a reproducible negative result plus a working SnapKV-style policy; the report says so plainly.

### Attention policies were first scored offline

Problem: the mask simulation cannot see the evicted cache, so the first H2O and SnapKV-style scores came from a full-cache pass refined once.
Impact: the numbers might not hold for a real online policy.
Strategy: built `BLT.forward_online` (global layers run patch by patch with a real cache, attention accumulated as produced) and tested it against step-by-step decoding for the static policies and for causality of the attention ones.
Outcome: online SnapKV-style is never worse than the offline score and beats recency in 24 of 24 comparisons; online H2O is worse than recency in 22 of 24.

### A wrong conclusion from under-trained long-context models

Problem: the first 16,384-byte models (batch size 2) scored 1.89 BPB, worse than the 4,096-byte models, and suggested the entropy penalty fades with context.
Impact: a wrong conclusion about long context.
Strategy: tuned learning rate and batch size on validation; batch 8 gave 1.5843 BPB.
Outcome: with the tuned models the penalty does not fade (+0.016 to +0.021 BPB against random); the earlier statement is withdrawn.

### Ten times longer training overfits TinyStories

Problem: 30,000 steps on the 26.7 MB TinyStories file are 37 epochs; training loss fell to 0.14 nats while validation BPB rose to 1.65.
Impact: the TinyStories scale-up is unusable.
Strategy: kept the WikiText scale-up (1.9 epochs of a 526 MB file, no overfitting) as the evidence and reported the TinyStories rows only for completeness.

### The models cannot retrieve

Problem: the planted-name test could not show any policy helping, because the model scores the same with and without the earlier mention, even when trained ten times longer.
Impact: nothing about retrieval-sensitive use can be concluded.
Strategy: reported as a limit of models this small; kept the retention numbers (the entropy rule keeps the planted name more often than random does).

### PG-19 is not loadable the standard way

Problem: the `deepmind/pg19` dataset ships as a loading script, which current `datasets` no longer runs.
Strategy: used the `emozilla/pg19` community mirror of the same test split (100 books, 40.5 MB, NFC normalised, split and hash checks pass) and recorded the substitution.
Outcome: streamed 20 books byte by byte with a real cache. Recency is the only policy that beats resetting the context; the whole-book full cache is worse than the reset baseline because positions run 24 times past the training range, and that confounds the entropy comparison on old patches (see `docs/results_summary.md`).
