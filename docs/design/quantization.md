# Per-patch KV precision

## Setup

Each global key and value vector is quantized at the moment of its insertion, with symmetric absmax scaling per vector (per head and patch) and one fp16 scale. At 2 bits the scheme has but three levels (-1, 0, 1). A patch is stored at a high width or at a low one, never between. Bytes per entry follow `kvc/quant.py`: 9,216 at fp16, 4,752 at 8 bits (52 percent), 2,448 at 4 bits (27 percent) and 1,296 at 2 bits (14 percent), scales included.

Tier policies decide which patches receive the high width. All four grant it to the same fraction f of the patches, so that their cache bytes are nearly equal:

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
- Admixing high-precision patches by the entropy rule does not help: at f = 0.5 (about 20 percent of bytes) entropy costs +0.042 (TinyStories) and +0.017 (WikiText), random +0.036 and +0.014, entropy_inv +0.028 and +0.010, recency +0.003 and +0.001.
- Paired differences: entropy is worse than random, worse than recency and worse than entropy_inv at every fraction on both datasets; every interval excludes zero.
- A recent window at high precision with older patches at low precision is the best mixed rule and about equal to uniform 3-bit at the same bytes.

## Reading the result

The claim that entropy-guided precision beats random and recency at matched bytes finds no support here; the opposite holds. `docs/results_summary.md` gives the mechanism (high-entropy patches receive less attention).

## Limits

The entropy rule uses an online rank threshold, not one tuned on training data. Quantization is simulated (values are rounded and kept in fp32), and the byte counts are computed, not measured from a packed cache. One seed per model. The 2-bit width is coarse, so that the results at low f chiefly measure how well each rule guards the few high-precision slots.
