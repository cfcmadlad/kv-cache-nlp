# Viva notes

Collated from the `## Viva notes` section of each file in `docs/`. Regenerate with `python scripts/collate_viva.py`.

## Block 2.0: infrastructure (`docs/block_2_0.md`)

Why checkpoint so often? The cluster allows one job at a time with a 2-day limit, so a lost job costs days.
Why is the resume test a comparison with an uninterrupted run? Loading a checkpoint without error proves nothing; matching weights proves the optimizer and data order were restored.
What was verified on the cluster? One job (416139) was signalled with SIGUSR1 at step 1,864, saved a checkpoint, was requeued under the same job id, resumed from step 1,864 and finished at step 24,000. What is not verified there is equality of the resumed run with an uninterrupted one, which only the local test covers (Windows has no SIGUSR1).

## Block 2.1: byte datasets (`docs/block_2_1.md`)

Why bytes in one file? Memory-mapping gives random access to any window without loading 526 MB of WikiText, and the offsets let evaluation score whole documents.
Why split code by hash, not randomly? The split is reproducible from the content alone, and a file and its duplicate cannot be on both sides.
Why not remove near-duplicates? It needs fuzzy hashing and was out of the block's scope; stated as a limitation.

## Block 2.2: entropy model and patching (`docs/block_2_2.md`)

Why a sliding window? The entropy model is a cache cost too; a 512-byte window bounds it.
Why must boundaries ignore the future? At generation time the next byte does not exist yet, and the patch decision has to be made before it is read. The causality test changes later bytes and checks that earlier boundaries stay the same.
Why tune the threshold on validation? Mean patch length is a property of the text distribution, not of the loss; only a single scalar is fitted.

## Block 2.3: BLT-style model with incremental decoding (`docs/block_2_3.md`)

Why mean pooling? It is the simplest pooling that needs no extra parameters beyond one linear layer; BLT's cross-attention pooling is a possible later upgrade.
Why is the BOS patch separate? Without it, the first byte would have no global state to read.
What is the compression target? The per-layer key and value tensors of the global model, one entry per patch.

## Phase 3: patch-level cache policies (`docs/phase_3_cache_policies.md`)

Why a protected recent window for all policies? Without it, random and entropy would sometimes drop the byte context needed right now, and the comparison would test the window rather than the signal.
Why include entropy_inv? If keeping low-entropy patches did equally well, the entropy signal would not be doing real work.
Why mask instead of real eviction for evaluation? Same numbers, tested equal, and the whole chunk runs in one pass instead of thousands of single-byte steps.

## Phase 4: per-patch KV precision (`docs/phase_4_quantization.md`)

Why re-quantize recency tiers as entries age? A real cache would keep a recent fp window and compress behind it; the simulation does the same.
Why is entropy_inv a control? If low-entropy patches are the ones the model reads, keeping them precise should beat keeping high-entropy ones, and it does.
Why is a 4-bit uniform cache the practical answer here? It is within 0.001 BPB of fp16 at about a quarter of the bytes, and no tiering rule beats it per byte.

## Phase 6: ablations, seeds and controls (`docs/phase_6_ablations.md`)

Why test fixed-stride models with the same policies? If the entropy signal only failed because entropy patching makes patches uneven, fixed strides would show a positive result; they show the same sign.
Why compare the model's own entropy with the entropy model? The proposal could fail because a separate 3.3M model is a poor proxy for what the big model finds surprising. It makes no difference.
Why does a sum rule tie random? Unknown. A sum favours long patches, so it mixes a length preference with the entropy preference; I have not separated the two.
Why is the sink irrelevant here? Probably because the local layers see a 256-byte window of raw bytes and the documents are short; I did not test longer contexts, where a sink may matter.
What might change the conclusion? Untested here: a task where rare surprising spans are queried later, such as retrieval or long-range copying, rather than next-byte prediction on short documents.
Why was online scoring built later, and does it change the result? The first tables scored attention offline (one pass, refined once under the mask) because the mask simulation cannot see the evicted cache. `BLT.forward_online` then ran the global layers patch by patch with a real cache and accumulated attention as it came; snap online is never worse than offline and beats recency in 24 of 24 comparisons, h2o online is worse than recency in 22 of 24, so the conclusions held. What remains untested is a step-decoding test for the attention policies themselves; the online loop is checked equal to step decoding for recency and random only.
Why does snap help and h2o not? Unknown. A guess: snap scores recent attention, which stays close to recency while correcting it, whereas accumulated attention favours old patches that were popular early. I have not tested that.
Why did the first 16k models score worse than the 4k ones, and what does the tuned result say? They were under-trained: batch 8 with learning rate 0.001 gave 1.5843 BPB against 1.8920 at batch 2, and a lower rate was worse at both sizes. With the tuned models the entropy penalty does not shrink (+0.016 to +0.021 against random at 64 to 512 patches). The batch-8 runs also see four times more bytes than the 4k models, so I cannot separate context length from training amount.
Does a better-trained model change the entropy result? It makes it larger: WikiText code-mix models trained ten times longer score 1.29 to 1.31 BPB and keeping high entropy costs +0.024 to +0.029 BPB against random at 64 patches in all three seeds (+0.002 at 3,000 steps). The TinyStories version overfit (37 epochs of a 26.7 MB file) and is not used.
Why can a fixed-size cache beat the full cache? The baseline resets the context every 4,096 bytes, so it forgets everything at each boundary; a recency window of 64 patches carries on across the boundary and scores 0.0008 to 0.0032 BPB better. Against a full cache over the whole 16,384-byte chunk (patch positions beyond training, 1.7201, 1.7178 and 1.8149 BPB for the three seeds) the 64-patch recency window scores 1.7180, 1.7109 and 1.8154, so it is better in two seeds and 0.0005 worse in the third; all of this is one dataset and 46 documents.
Do the models retrieve anything from the cache? No. In the planted-name test the score with the cue equals the score without it, even for the 30,000-step models, so nothing about retrieval can be concluded and the entropy rule's higher chance of keeping the name does not help.
Is the memory saving real? Measured on the GPU with real eviction: the global cache stays at 0.59 to 2.36 MB for 64 to 256 patches while a full cache reaches 137 MB at 65,536 bytes, and per-byte decoding time stays at 0.9 ms against 2.0 ms. The fixed byte-window caches (786 KB) and the entropy model (about 2.1 MB of cache) are not reduced, so at short contexts they dominate.

## Viva questions by rubric item (`docs/viva_by_rubric.md`)

### Novelty (3)

What is new here? A test of entropy-ranked cache retention on a byte-level model with patch-granular caches, against random and recency at equal cache bytes, with three seeds per setting, paired bootstrap intervals and a test split (`results/part3/seeds.md`, `results/part3/test_split.md`). Also an online, tested implementation of H2O and SnapKV-style policies over patch caches (`BLT.forward_online`, `tests/test_blt.py`), and measured streaming results with a bounded cache (`results/part3/stream_16k.md`, `results/part3/pg19_stream.md`). The result of the main test is negative.

What is not new? The BLT, StreamingLLM, H2O, SnapKV, KV quantization and the use of entropy for cache management on subword models are published (`NOVELTY.md`, `report/references.bib`). We do not claim the idea of entropy-guided caching; CONF-KV already uses next-token entropy and already has a random-eviction control.

How does your model differ from the BLT? Mean pooling of byte states instead of cross-attention, no n-gram hash embeddings, 13.3M parameters, 3,000 training steps (`docs/block_2_3.md`). It does not beat the BPE Transformer at equal training (`docs/results_summary.md`, gate table).

Why is a negative result worth reporting? Because the hypothesis is cheap and plausible and others may try it. We removed the usual escape routes: three seeds, fixed-stride and stride-5 controls, the model's own entropy, four aggregation rules, sink sizes, a byte-level cache, ten times more training, 16,384-byte contexts, a recency-protected variant and a held-out test split (`docs/results_summary.md`, `docs/phase_6_ablations.md`).

How does your work differ from CONF-KV, InfoKV and Scratchpad Patching? CONF-KV uses entropy to set how much to keep and ranks entries by attention and recency; we rank which patches to keep by entropy alone. InfoKV mixes entropy with attention. Scratchpad Patching uses byte-level uncertainty to add slots inside patches. We read CONF-KV and the counter-causal paper in full and only the abstracts of InfoKV, Scratchpad Patching and Don't Waste Bits, so for those three our statement of difference is provisional (`NOVELTY.md`, `report/main.tex` section 2).

### Improvement over the previous technique (3)

Does the byte model beat the Part 2 baselines? Against the Kneser-Ney 4-gram, yes by a wide margin. Against the BPE Transformer at equal training (3,000 steps), no: 0.7980 against 0.6939 (TinyStories) and 1.6801 against 1.5200 (WikiText). Trained ten times longer the WikiText code-mix BLT scores 1.2949 to 1.3118, below the BPE number, but that comparison gives the BLT ten times more training and a code mix, so it is not like for like (`docs/results_summary.md`, gate table and scale section).

Is entropy patching better than fixed stride? Likely on WikiText (lower in 3 of 3 same-seed pairs, mean gap 0.039 against seed ranges of 0.03 to 0.04), tied with stride 4 on TinyStories. Three seeds do not establish it (`docs/results_summary.md`, seed section).

Where is the improvement then? Cache memory and streaming, not accuracy: eviction plus 4-bit keeps 7.4 percent of the WikiText cache bytes for +0.0012 BPB (`results/part3/deploy_combo.md`); a fixed recency cache beats resetting the context on PG-19 by 0.0086 BPB with 256 patches (`results/part3/pg19_stream.md`); SnapKV-style beats recency in 24 of 24 online comparisons by 0.0002 to 0.0077 BPB (`results/part3/online_policies.md`).

Is the SnapKV-style gain meaningful? It is small. At 3,000 steps it recovers a fraction of the eviction cost (0.0041 of 0.0098 for seed 0 of TinyStories at 32 patches); at 30,000 steps it is 0.005 BPB below recency at 64 patches in all three seeds (`results/part3/scale_and_tuning.md`). The intervals exclude zero; the practical gain is modest.

Does the BLT save compute? Against a byte-level Transformer of the same width, yes (11.1 against 30.9 MFLOPs per byte at 1,024 bytes). Against the BPE Transformer, no: 11.1, or 19.6 with the entropy model, against 7.4. The case for the BLT here is cache memory, not FLOPs (`results/part3/flops.md`; analytic, not measured).

Does the patch cache save memory without compression? Not against BPE: at full precision it is about 0.9 times the BPE cache and a byte-level Transformer needs about 4 times more (`results/part3/deploy_memory.md`). The savings come from eviction and quantization.

### Results and error analysis (2)

Why bits per byte? It is tokenizer-independent, so the 4-gram, BPE and byte models are scored on identical text. Its limits: it does not measure retrieval, and the Part 2 BPE number is 0.4 to 0.5 percent optimistic (`README.md`).

How do you know differences are not noise? Every cache comparison is paired by document with a 95 percent bootstrap interval; the sign holds in every seed. Patching comparisons are not established because seed spread (0.03 to 0.14 BPB) is as large as the gaps (`docs/results_summary.md`).

Why does keeping high-entropy patches fail? Unknown for certain. Attention received by old patches correlates negatively with patch entropy in middle and upper layers of most models (`results/part3/attention_*.md`). In the error analysis the entropy rule keeps almost only the highest-entropy third of patches and loses on the lowest-entropy third, which is 55 to 57 percent of bytes, and even the highest-entropy third gets worse than under recency (`results/part3/error_analysis.md`, `docs/error_analysis_part3.md`). This is a correlation; no intervention tests it.

Where does eviction hurt most? Per-byte loss increases are heavy-tailed: the top 1 percent of bytes carry 30.4 percent (TinyStories) and 18.0 percent (WikiText) of the extra loss under entropy against recency, and they are often bytes that the full cache predicts well (`docs/error_analysis_part3.md`).

Was the test split used properly? Once, for 18 WikiText models with the policies and budgets fixed on validation. Keeping high entropy is worse than random and recency in all 36 cells (`results/part3/test_split.md`). TinyStories has no test split.

Does the result hold with more training? It gets larger: +0.0287 against random at 64 patches at 30,000 steps against +0.0020 at 3,000 (`results/part3/scale_and_tuning.md`). The TinyStories version overfits and is not used.

Why can a fixed cache beat the full cache? The baseline resets the context every 4,096 bytes; a small recency window carries on across the boundary. On PG-19 the whole-book full cache is worse than the reset because positions run 24 times past the training range (`results/part3/stream_16k.md`, `results/part3/pg19_stream.md`).

What can you not conclude? Anything about retrieval (the planted-name test shows the models do not use the earlier mention, `results/part3/deploy_needle.md`), anything about models much larger than 13M parameters, and on PG-19 whether the entropy signal or long-distance attention causes the loss (keeping old patches confounds the two).

### Challenges (2)

What went wrong, and what did you do? Full list with strategy and outcome in `CHALLENGES.md`. The ones that changed conclusions: single-seed patching claim withdrawn after seed variance showed it was noise; 16,384-byte conclusion withdrawn after the models were found under-trained; cache byte shares were doubled by an accounting error and all tables were regenerated; offline attention scores were replaced by an online implementation.

What did you get wrong and withdraw? The 3.3 percent TinyStories patching gap, the claim that the entropy penalty fades at long context, the claim that the code mix reverses the attention correlation (one outlier seed), and the early cache shares (`docs/results_summary.md`, `CHALLENGES.md`).

How did you cope with the cluster limits? Runs checkpoint, catch SIGUSR1, requeue and resume. The path was verified under one job id (416139: signalled at step 1,864, resumed, finished at step 24,000). Three earlier attempts failed for avoidable reasons that are recorded (`docs/results_summary.md`, requeue section).

### Demo and viva (20)

Does the demo need a GPU? No. Options 4, 6 and 7 were run on CPU from a fresh clone (`docs/demo_script.md`).

Why does option 6 show almost no difference between policies? A 1.8M-parameter model trained for 300 steps barely uses its global cache (BPB 2.147 full against 2.146 to 2.148). The demo says so. Options 4 and 7 use the fully trained models.

In option 7 the entropy policy beat random. Does that contradict the paper? One 4,000-byte text is an anecdote. The evidence is 72 paired comparisons with intervals (`results/part3/seeds.md`). In that run recency was also better than the full cache, probably because the demo joins 40 unrelated validation stories into one text so old stories are noise; that explanation is untested (`docs/demo_script.md`).

Why is option 4's cost table so close between policies? On one document of about 1,200 bytes with a budget of 32 patches the differences are 0.01 to 0.02 BPB, the same size as in the tables (`results/part3/comparison.md`).

How do I regenerate the figures and the FLOPs table? `python report/make_figures.py` parses `results/part3/*.md`; `python scripts/flops.py` writes `results/part3/flops.md`; `python scripts/error_analysis_part3.py` writes `results/part3/error_analysis.md` (CPU, about 6 minutes).
