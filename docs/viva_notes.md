# Viva notes

Collated from the `## Viva notes` section of each file in `docs/`. Regenerate with `python scripts/collate_viva.py`.

## Block 2.0: infrastructure (`docs/block_2_0.md`)

Why checkpoint so often? The cluster allows one job at a time with a 2-day limit, so a lost job costs days.
Why is the resume test a comparison with an uninterrupted run? Loading a checkpoint without error proves nothing; matching weights proves the optimizer and data order were restored.
What is not verified yet? The SIGUSR1 and requeue path, which only exists on the cluster (Windows has no SIGUSR1).

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
Why score the attention policies offline? The tested simulation applies one mask per chunk, so attention scores cannot come from the evicted cache; I refine them once under the resulting mask. An online version would need score updates in the step API, which I have not built, so the numbers may differ.
Why does snap help and h2o not? Unknown. A guess: snap scores recent attention, which stays close to recency while correcting it, whereas accumulated attention favours old patches that were popular early. I have not tested that.
Why did the 16k models score worse than the 4k ones? Unknown. They saw four times fewer sequences per step and were not tuned; I did not test either.
