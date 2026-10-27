# Challenges, and what became of them

Every entry follows one cadence: the problem, its impact, our action, the outcome. Most of these obstacles were discovered by our own checks (a cross-table comparison, a causality test, a perusal of logs, a repetition of seeds) before they could corrupt a published conclusion. Where a conclusion had to be retracted it is marked **corrected**; such retractions are the process working as intended, and the figures in `docs/results_summary.md` reflect the amended state.

## Overview

| # | Challenge | Phase | Outcome |
|---|---|---|---|
| 1 | Part 2 BPB slightly flattering | Part 2 | Disclosed; corrected values published |
| 2 | Baseline trained without gradient clipping | Part 2 | Reproducible; baseline a trifle weaker, as stated |
| 3 | WikiText-103 lacks document boundaries | Part 2 | Articles rebuilt and verified |
| 4 | Release bundle size | Part 2 | 138 MB bundle; full corpus rebuildable |
| 5 | Demo must run on an unknown machine | Part 2 | Fresh-clone rehearsals pass |
| 6 | Cluster submission blocked | Part 2 | Workaround found; cluster used for Part 3 |
| 7 | One job at a time, two-day limit | Part 3 | Checkpoint, requeue and resume verified end to end |
| 8 | Gated code dataset | Part 3 | 28,229 files built and verified |
| 9 | Entropy-model BPB below the planned range | Part 3 | Causality tests found no leak |
| 10 | Training and evaluation must see the same patches | Part 3 | Identical boundaries by construction |
| 11 | An out-of-memory error silently corrupted a cache | Part 3 | Caught, rebuilt, queue hardened |
| 12 | Seed variance as large as some effects | Part 3 | **Corrected**: three seeds, paired intervals |
| 13 | Cache byte accounting error | Part 3 | **Corrected**: all tables regenerated |
| 14 | Prose-trained models cannot read code | Part 3 | Retrained with a code mix |
| 15 | Attention policies first scored offline | Part 3 | Online implementation; conclusions confirmed |
| 16 | Under-trained 16,384-byte models | Part 3 | **Corrected**: tuned; the penalty does not fade |
| 17 | TinyStories overfits at 30,000 steps | Part 3 | WikiText used as the evidence |
| 18 | The models do not retrieve | Part 3 | Scope stated; no retrieval claim |
| 19 | PG-19 loading script unsupported | Part 3 | Mirror of the same split, recorded |
| 20 | The central hypothesis was refuted | Part 3 | Reproducible negative result with a mechanism, plus working alternatives |

## Part 1

No separate Part 1 difficulties were recorded; the team confirmed (27 Oct 2026) that the Part 2 entries below stand as the overall record for Parts 1 and 2.

## Part 2

### 1. The reported BPB was slightly flattering

- Problem: the first token of every 256-token chunk is context only, yet its bytes were still counted in the denominator.
- Impact: Transformer BPB is 0.4 to 0.5 percent too low on these sets (far more on very short inputs). Corrected values: 0.6978 (TinyStories val), 1.5257 (WikiText val), 1.5370 (WikiText test).
- Action: kept the reported numbers unaltered for traceability, documented the correction in the README, and made Part 3 scoring count every byte it scores.
- Outcome: disclosed; Part 3 comparisons use the corrected scoring.

### 2. Training without gradient clipping

- Problem: the two reported checkpoints were trained before clipping was added.
- Impact: a rerun with clipping at 1.0 gives 0.6826 on TinyStories instead of 0.6939, so the baseline is slightly weaker than it might have been.
- Action: added a `--grad-clip` flag, where 0 reproduces the reported checkpoints exactly; the reported runs keep clip 0.
- Outcome: the reported numbers reproduce. The handicapped baseline favours the Part 3 model if it favours anything, and the report says so.

### 3. Building WikiText-103 documents

- Problem: the raw line export has no document boundaries and employs placeholder tokens such as `@-@`.
- Impact: corrupted text would distort byte counts and BPB.
- Action: rebuilt articles from the line export, restored the placeholders, applied NFC normalisation and exact deduplication.
- Outcome: documents usable; byte counts recorded in `stats.json`.

### 4. Release bundle size

- Problem: the full WikiText-103 train set makes a ponderous release bundle and a slow first download for the demo.
- Impact: demo training on WikiText would not match the original run.
- Action: the release copy cuts train to 3,000 documents (138 MB zip, SHA-256 checked); the full corpus rebuilds with `prepare_data.py`. The team confirmed (27 Oct 2026) that this was our choice of bundle size, not a platform limit.
- Outcome: the demo works offline from the bundle.

### 5. Demo robustness

- Problem: the examiner runs our code on a machine of unknown provenance.
- Impact: a missing Python or PyTorch would sink the demonstration.
- Action: `run.bat` and `run.sh` find or install Python and reuse or build a PyTorch environment; `demo.py` fetches its own assets; model loading is hardened (restricted unpickler, `weights_only=True`).
- Outcome: an offline rehearsal from a fresh unzip passed, and options 1 to 7 were rerun on CPU from a fresh clone on 27 Oct 2026 (`docs/demo_script.md`). Antivirus software killed or quarantined installer-starting scripts and test wrappers during earlier testing.

### 6. Cluster access

- Problem: on 2026-09-21 the QOS carried `MaxSubmitJobsPerUser=0`, so that no job could be submitted.
- Impact: Part 2 training ran on a laptop RTX 4060.
- Action: on 2026-10-09 we found that `sbatch --test-only` succeeds once the job requests `--gres=mps:50`; the limits are 4 CPUs, half a GPU, 80 GB of RAM and a two-day wall time.
- Outcome: the cluster became usable for Part 3.

## Part 3

### 7. Cluster limits shape the design

- Problem: one job at a time, a two-day wall time, queueing behind other users, shared CUDA modules only up to 12.1.
- Impact: a lost or timed-out job could squander days.
- Action: every run checkpoints periodically, handles SIGUSR1 five minutes before the time limit, requeues itself and resumes; PyTorch installs from a cu128 wheel in a home venv; run scripts skip finished outputs, so a failed chain restarts where it stopped.
- Outcome: resume verified locally (a killed run matches an uninterrupted one to 1e-6) and on the cluster. Job 416139 was signalled at step 1,864, requeued under the same id, resumed from 1,864 and finished at 24,000. Three earlier attempts to demonstrate this failed for avoidable reasons of ours (a signalling script that never fired twice, and a reused run name that overwrote a smoke run's config); the docs record both.

### 8. Code data is gated

- Problem: `bigcode/the-stack-smol` requires a Hugging Face login with the dataset terms accepted.
- Impact: the code split could not be built until a team member logged in on the cluster.
- Action: `prepare_code.py` takes the repository name and a per-language cap; the splitting and verification logic was first tested on synthetic files.
- Outcome: built (28,229 files; split and hash checks pass).

### 9. Entropy-model BPB fell below the planned range

- Problem: the plausible range had been set at 0.9 to 1.3 (TinyStories) and 1.7 to 2.2 (WikiText); we measured 0.8438 and 1.6983.
- Impact: a BPB that looks too good may betoken a leak, and this model also places the patch boundaries.
- Action: causality tests (changing later bytes never changes earlier logits, windows limit reach, documents are isolated) and a sanity bound from the BPE Transformer (0.694 and 1.520) and the 4-gram (1.81 and 2.41).
- Outcome: no leak found; the planned range had been conservative.

### 10. Training and evaluation must see the same patches

- Problem: training samples windows from the middle of documents, whereas evaluation scores whole documents from a BOS token.
- Impact: a mismatch would make evaluation BPB differ from training loss for reasons unconnected with the model.
- Action: documents are isolated by an attention mask built from BOS positions; patch boundaries come from a cached per-byte entropy array, so training and evaluation share them; evaluation scores 4,096-byte chunks that begin exactly like training windows.
- Outcome: consistent; the cache matches direct computation to 1e-3 (float16 storage).

### 11. An out-of-memory error silently corrupted a cache

- Problem: the WikiText entropy cache crashed for lack of memory while another job occupied the GPU, but the queue carried on and the half-written array remained on disk.
- Impact: the entropy-patched WikiText model would have trained on zeros for part of its data.
- Action: caught by reading the queue log; the cache was deleted and rebuilt, and the next queue halts if the cache script fails.
- Outcome: repaired before use. The same limit returned later: two 16,384-byte trainings (about 45 GB each) could not share one 95 GB GPU, so those runs went one at a time.
- Lesson: our own time estimates were also awry (a 60-step laptop test suggested 0.3 s per step, whereas real steps took 1.5 to 2.4 s). We now measure with the GPU idle; on the cluster a 3,000-step run takes about 90 seconds, which made tenfold training cheap.

### 12. Seed variance rivals some effects (corrected)

- Problem: seed 0 of TinyStories had a full-cache BPB of 0.7980 against 0.8544 for seed 2 (seed spread 0.056; 0.135 for fixed stride).
- Impact: the 3.3 percent gap between entropy patching and fixed stride that we first reported lay within seed noise.
- Action: three seeds for every patching scheme in all four settings, a stride-5 control, and paired document bootstrap intervals, so that eviction comparisons on one model are unaffected by seed. The single-seed claim was retired.
- Outcome: the eviction sign holds in every seed; patching comparisons are stated as likely, not established.

### 13. A cache size accounting error (corrected)

- Problem: `eval_policies.py` counted a full-cache chunk as P entries but a masked chunk as about (P + 1) / 2, so that full-cache bytes were doubled.
- Impact: the first quoted shares of full-cache bytes were half the true shares (32 and 15 percent instead of 16 and 7). BPB was unaffected.
- Action: found when the comparison table set eviction and quantization rows side by side and the full-cache sizes disagreed (1,669 KB against 839 KB). Fixed, the policy evaluations rerun on the cluster, every table regenerated.
- Outcome: corrected; all tables in `results/part3/` use the repaired accounting.

### 14. Prose-trained models cannot read code

- Problem: models trained on prose scored 7 to 8 BPB on code, so the first code evaluation said nothing.
- Impact: the code-domain portion of the plan was uninformative.
- Action: a mix option in the entropy and BLT trainers (15 percent of each batch from code), with retraining on WikiText and then TinyStories.
- Outcome: the mix costs about 0.05 BPB on WikiText and 0.09 on TinyStories; the eviction conclusion is unchanged, though smaller in size on WikiText.

### 15. Attention policies were first scored offline

- Problem: the mask simulation cannot see the evicted cache, so the first H2O and SnapKV-style scores came from a full-cache pass refined once.
- Impact: the numbers might not survive a genuinely online policy.
- Action: built `BLT.forward_online` (global layers run patch by patch with a real cache, attention accumulated as produced) and tested it against step-by-step decoding for the static policies and for causality of the attention ones.
- Outcome: online SnapKV-style is never worse than the offline score and beats recency in 24 of 24 comparisons; online H2O is worse than recency in 22 of 24.

### 16. Under-trained long-context models (corrected)

- Problem: the first 16,384-byte models (batch size 2) scored 1.89 BPB, worse than the 4,096-byte models, and suggested that the entropy penalty fades with context.
- Impact: a wrong conclusion about long context.
- Action: tuned learning rate and batch size on validation; batch 8 gave 1.5843 BPB. The tuned models also see four times more bytes, so context length and amount of training are not disentangled.
- Outcome: with the tuned models the penalty does not fade (+0.016 to +0.021 BPB against random); the earlier statement was retired.

### 17. Tenfold training overfits TinyStories

- Problem: 30,000 steps on the 26.7 MB TinyStories file are 37 epochs; training loss fell to 0.14 nats while validation BPB rose to 1.65.
- Impact: the TinyStories scale-up is unusable.
- Action: used the WikiText scale-up (1.9 epochs of a 526 MB file, no overfitting) as the evidence and reported the TinyStories rows for completeness only.
- Outcome: scale conclusions rest on WikiText.

### 18. The models cannot retrieve

- Problem: the planted-name test could not show any policy helping, because the model scores the same with and without the earlier mention, even when trained tenfold longer.
- Impact: nothing about retrieval-sensitive use can be concluded.
- Action: reported as a limit of models this small; kept the retention numbers (the entropy rule keeps the planted name more often than random does).
- Outcome: the scope of every claim is stated.

### 19. PG-19 is not loadable the standard way

- Problem: the `deepmind/pg19` dataset ships as a loading script, which current `datasets` no longer runs.
- Impact: the long-book evaluation named in the proposal was blocked.
- Action: used the `emozilla/pg19` community mirror of the same test split (100 books, 40.5 MB, NFC normalised, split and hash checks pass) and recorded the substitution.
- Outcome: 20 books streamed byte by byte with a real cache. Recency is the only policy that beats resetting the context; the whole-book full cache is worse than the reset baseline because positions run 24 times past the training range, which also confounds the entropy comparison on old patches (`docs/results_summary.md`).

### 20. The central hypothesis was refuted

- Problem: keeping high-entropy patches is worse than random in 68 and than recency in 70 of 72 comparisons, in every seed, in the code-mix models, on the WikiText test split, with models trained tenfold longer (where the penalty grows), at 16,384 bytes, and for a recency-protected variant.
- Impact: the core claim of the proposal is not supported.
- Action: reported as measured; tried the obvious escape routes (other aggregations, the model's own entropy, sink size, a byte-level cache, more training, longer context, a protected recency window); added an attention-based rule; and wrote an error analysis showing what the entropy rule keeps (`docs/error_analysis_part3.md`).
- Outcome: a reproducible negative result with a mechanism, accompanied by working compression results (SnapKV-style eviction, 4-bit quantization, bounded-memory streaming).
