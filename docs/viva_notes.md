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
