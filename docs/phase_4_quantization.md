# Phase 4: per-patch KV precision

Rubric: Novelty (3), Improvement (3), Results and error analysis (2), Demo and viva (20).

## Setup

Every global key and value vector is quantized when it is inserted, with symmetric absmax scaling per vector (per head and patch) and one fp16 scale. At 2 bits the scheme has three levels (-1, 0, 1). A patch is stored either at a high or a low bit width. Bytes per entry follow `kvc/quant.py`: 9,216 at fp16, 4,752 at 8 bits (52 percent), 2,448 at 4 bits (27 percent) and 1,296 at 2 bits (14 percent), scales included.

Tier policies decide which patches get the high width. All four use the same fraction f of patches at the high width, so cache bytes are close:

| Policy | High-precision patches |
|---|---|
| recency | the newest ceil(f x P) patches (and the sink), older ones are re-quantized to the low width as they age |
| random | each patch with probability f, seeded, 3 seeds averaged |
| entropy | a patch whose mean entropy ranks in the top fraction f of patches seen so far (online, uses only the past) |
| entropy_inv | the opposite: the lowest-entropy fraction |

Uniform 8, 4, 3 and 2 bit rows are the reference points. Two configurations were run on both datasets: high 8 / low 2 and high 4 / low 2. `tests/test_quant.py` checks that the tier-mask forward pass equals real step-by-step decoding with a quantized cache, to 1e-5, for every policy.

## Results (validation, matched documents, 95 percent bootstrap intervals over documents)

Files in `results/part3/`: `quant_tinystories_hi4_lo2.md`, `quant_wikitext103_hi4_lo2.md` and the hi 8 variants.

- Uniform 4-bit costs +0.0010 BPB on TinyStories and +0.0004 on WikiText at 27 percent of fp16 bytes. Uniform 3-bit costs +0.0048 and +0.0019 at 20 percent. Uniform 2-bit costs +0.072 and +0.024.
- Mixing in high-precision patches with the entropy rule does not help: at f = 0.5 (about 20 percent of bytes) entropy costs +0.042 (TinyStories) and +0.017 (WikiText), random +0.036 and +0.014, entropy_inv +0.028 and +0.010, recency +0.003 and +0.001.
- Paired differences: entropy is worse than random, worse than recency and worse than entropy_inv at every fraction on both datasets; every interval excludes zero.
- A recent window at high precision with older patches at low precision is the best mixed rule and about equal to uniform 3-bit at the same bytes.

## Reading the result

The claim that entropy-guided precision beats random and recency at matched bytes is not supported here; the opposite holds. `docs/results_summary.md` has the mechanism (high-entropy patches receive less attention).

## Limits

The entropy rule uses an online rank threshold, not a threshold tuned on training data. Quantization is simulated (values are rounded and kept in fp32); byte counts are computed, not measured from a packed cache. One seed per model. The 2-bit width is coarse, so results at low f mostly measure how well each rule protects the few high-precision slots.

## Viva notes

Why re-quantize recency tiers as entries age? A real cache would keep a recent fp window and compress behind it; the simulation does the same.
Why is entropy_inv a control? If low-entropy patches are the ones the model reads, keeping them precise should beat keeping high-entropy ones, and it does.
Why is a 4-bit uniform cache the practical answer here? It is within 0.001 BPB of fp16 at about a quarter of the bytes, and no tiering rule beats it per byte.
