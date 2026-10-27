# Results summary (validation only, 25 Oct 2026)

All numbers are bits per byte (BPB) on matched validation documents: TinyStories first 500 documents (405,231 bytes) and WikiText-103 all 60 documents (1,129,194 bytes). The WikiText test split was evaluated once (section below); TinyStories has none. Intervals are 95 percent bootstrap intervals over documents. Regenerate any table with the command in its section; the raw tables are in `results/part3/` and `results/phase2.md`.

Training hardware: seed 0 of the entropy-patched and fixed-stride-4 models was trained on a laptop RTX 4060; seeds 1 and 2, the stride-5 control and the byte-level model were trained on the cluster (RTX PRO 6000). Same code and configs, different GPU, so seed 0 differs from the others in more than the seed.

## Gate table (`python scripts/results.py --phase 2`)

| Model | TinyStories | WikiText | Bytes per patch or token | Global KV bytes per 1000 input bytes |
|---|---|---|---|---|
| Kneser-Ney byte 4-gram | 1.8085 | 2.4103 | n/a | n/a |
| BPE transformer 13.8M, reported | 0.6939 | 1.5200 | 4.1 / 4.0 | 2.22 MB / 2.31 MB |
| BPE transformer, corrected | 0.6978 | 1.5257 | same | same |
| Entropy model 3.3M | 0.8438 | 1.6983 | 4.5 | n/a |
| BLT 13.3M, fixed stride 4 | 0.7726 | 1.7600 | 4.0 | 2.31 MB |
| BLT 13.3M, entropy patching (seed 0) | 0.7980 | 1.6801 | 4.5 | 2.05 MB |

Part 2 rows reproduce exactly. The BLT-style models do not beat the BPE transformer; this is a smaller, shorter-trained model (3,000 steps, 100M bytes seen) without the n-gram hashing or cross-attention of the original BLT.

## Seed variance and entropy patching against fixed stride (`python scripts/compare.py seeds`)

Full-cache BPB for three seeds of every patching scheme (seeds 0, 1, 2; mean in brackets; range is the largest minus the smallest seed):

| Dataset | Entropy patching (4.5 bytes per patch) | Fixed stride 4 | Fixed stride 5 |
|---|---|---|---|
| TinyStories | 0.7980, 0.8026, 0.8544 (0.8183; range 0.056) | 0.7726, 0.7907, 0.9078 (0.8237; range 0.135) | 0.8071, 0.8214, 0.9278 (0.8521; range 0.121) |
| WikiText | 1.6801, 1.6994, 1.7080 (1.6958; range 0.028) | 1.7600, 1.7161, 1.7280 (1.7347; range 0.044) | 1.7768, 1.7473, 1.7645 (1.7629; range 0.030) |
| TinyStories, code mix | 0.9019, 0.8925, 0.9396 (0.9114; range 0.047) | 0.9150, 0.9298, 0.9890 (0.9446; range 0.074) | 0.9308, 1.0197, 0.9951 (0.9819; range 0.089) |
| WikiText, code mix | 1.7190, 1.7117, 1.8139 (1.7482; range 0.102) | 1.7541, 1.7564, 1.8100 (1.7735; range 0.056) | 1.7688, 1.8129, 1.8633 (1.8150; range 0.095) |

Seed spread is as large as most patching gaps, and larger for the fixed-stride models, so the earlier single-seed comparison was too weak. On WikiText entropy patching has lower BPB than both fixed strides in all three seed pairs (by 0.017 to 0.080 against stride 4 and 0.048 to 0.097 against stride 5), with fewer cache entries per byte than stride 4; the mean gap of 0.039 (stride 4) is about the size of the seed ranges, so I call it likely but three seeds do not establish it. On TinyStories entropy patching ties stride 4 (means 0.8183 and 0.8237, ranges of 0.056 and 0.135) and is lower than stride 5 in all three seed pairs. With the code mix entropy patching is lower in 11 of the 12 same-seed comparisons (the exception is WikiText seed 2 against stride 4, 1.8139 against 1.8100), but the seed ranges overlap the gaps. The earlier claim that entropy patching is 3.3 percent worse on TinyStories stays withdrawn. Seed 2 is the worst or near-worst seed for every scheme on TinyStories, which suggests the seed sets the data order more than the patching does. Seed 0 of the prose-only models was trained on a laptop GPU, the rest on the cluster.

## Entropy model checks

Patch starts at word starts: 67.8 percent on TinyStories (chance 19.9 percent) and 66.7 percent on WikiText (chance 16.3 percent). Patch starts that land on a space byte: 2.8 and 1.3 percent. Mean patch length 4.5 by construction (threshold tuned on validation: 1.09 nats TinyStories, 2.17 nats WikiText).

## Eviction at a fixed budget (global cache patches kept)

BPB, entropy-patched BLT seed 0. Full cache: TinyStories 0.7980, WikiText 1.6801. Protected for every policy: the first patch and the 8 newest.

| Dataset | Budget (patches, share of full cache bytes) | recency | random | keep low entropy | keep high entropy |
|---|---|---|---|---|---|
| TinyStories | 32 (32%) | 0.8078 | 0.8115 | 0.8089 | 0.8231 |
| TinyStories | 64 (56%) | 0.8016 | 0.8033 | 0.8026 | 0.8109 |
| WikiText | 64 (15%) | 1.6836 | 1.6860 | 1.6901 | 1.6922 |
| WikiText | 128 (28%) | 1.6814 | 1.6833 | 1.6864 | 1.6885 |

Keeping high-entropy patches (the proposal) is worse than random and than recency, in all three seeds, at every budget on both datasets. The paired difference against random (positive means worse) ranges over seeds from +0.0116 to +0.0245 (TinyStories, 32 patches), +0.0076 to +0.0162 (TinyStories, 64), +0.0062 to +0.0124 (WikiText, 64) and +0.0052 to +0.0120 (WikiText, 128); every interval excludes zero, as do those against recency. The fixed-stride-4 and stride-5 models show the same sign, so it does not depend on entropy patching. Effects are small, at most 0.03 BPB. Files: `results/part3/seeds.md`, `results/part3/policies_*`.

## Ablations (`python scripts/compare.py ablations`, seed-0 model)

| Change | TinyStories entropy minus random (32 / 64 patches) | WikiText (64 / 128) |
|---|---|---|
| Main setting: entropy model, mean over the patch | +0.0116 / +0.0076 | +0.0062 / +0.0052 |
| The model's own output entropy as the signal | +0.0110 / +0.0067 | +0.0060 / +0.0050 |
| Maximum over the patch | +0.0019 / +0.0006 | +0.0037 / +0.0030 |
| Entropy of the first byte of the patch | +0.0098 / +0.0050 | +0.0059 / +0.0047 |
| Sum over the patch (favours long patches) | -0.0007 / -0.0002 (interval includes zero) | +0.0048 / +0.0035 |

- Using the model's own entropy instead of the separate entropy model changes nothing, so the negative result is not caused by the separate model. The own-entropy score was computed from a full-cache pass; an online version would use the entropy under the evicted cache.
- No scoring rule beats random. Maximum and sum are closest to random; on TinyStories the sum rule is indistinguishable from random, and it is still worse than recency (+0.0030 and +0.0014).
- Sink size 0, 1 or 4 changes BPB by at most 0.001 for every policy, because the 256-byte local windows already carry the start of the context.
- Byte-level against patch cache (TinyStories, stride-1 model, one seed). The byte-level model has a better full-cache BPB (0.7220 against 0.7726 for stride 4) but 4 times the cache entries. At equal entry counts, which are equal bytes, eviction is far more damaging for bytes: with 32 entries recency costs +0.2286 over the full cache at byte level and +0.0133 with patches; with 64 entries, +0.1540 and +0.0050. Keeping high entropy is again worse than random at byte level, by +0.045 to +0.075. At 288 byte entries (4.5 times the cache of 64 patches) recency reaches 0.7494, better than 64 patches at 0.7775, so patches win per byte of cache and bytes win only with several times more cache.

## Precision tiers (global K and V, high 4 bits and low 2 bits, half of the patches at the high width, about 20 percent of fp16 bytes)

| Dataset | recency | random | keep low entropy | keep high entropy | uniform 3-bit | uniform 4-bit (27%) |
|---|---|---|---|---|---|---|
| TinyStories (fp16 0.7980) | 0.8006 | 0.8336 | 0.8258 | 0.8398 | 0.8028 | 0.7990 |
| WikiText (fp16 1.6801) | 1.6815 | 1.6937 | 1.6902 | 1.6972 | 1.6820 | 1.6805 |

Uniform 4-bit is within 0.001 BPB of fp16 at 27 percent of the bytes. Entropy-guided tiers lose to random and recency again. Quantization was run for seed 0 only.

## All policies in one view (`python scripts/compare.py table`)

`results/part3/comparison.md` lists every eviction and precision setting with its mean resident cache size (computed from entry counts and bit widths), BPB and interval, and marks the rows on the bytes-against-BPB frontier (no other row has equal or fewer bytes and a lower BPB). No row that keeps high-entropy patches is on a frontier. On WikiText the frontier is recency eviction at 16 to 64 patches, then recency-tier precision, uniform 3-bit, 4-bit and 8-bit. On TinyStories it is keep-low-entropy eviction at 12 and 16 patches (ahead of recency by 0.003 and 0.0015 BPB, inside overlapping intervals), then recency-tier precision, uniform 4-bit and 8-bit. Uniform 4-bit costs 27 percent of the fp16 bytes and 0.0010 BPB on TinyStories, 0.0004 on WikiText.

## Code mix (`python scripts/compare.py seeds`, `code`, `ablations`, `table`; run names `*wikitext103mix*`)

The code set was built from `bigcode/the-stack-smol` (28,229 files; split and hash checks pass). Models trained on prose alone score 7.64 (TinyStories) and 7.17 (WikiText) BPB on 200 validation code files, close to a uniform byte guess, so code needs training data. The WikiText entropy model and BLT were retrained with 15 percent of each batch drawn from code (`configs/*wikitext103mix*`, `scripts/run_mix.sh`): entropy model, three seeds of the entropy-patched BLT, and fixed strides 4 and 5 as controls. The patch threshold was tuned on WikiText validation only (4.5 bytes per patch there; 4.0 on code).

| Model (trained with code mix) | WikiText BPB | Code BPB (200 files) |
|---|---|---|
| Entropy model | 1.6883 (1.6983 without code) | not measured |
| BLT entropy patching, seeds 0, 1, 2 | 1.7190, 1.7117, 1.8139 (mean 1.7482; 1.6958 without code) | 1.7476, 1.7136, 2.1903 |
| BLT fixed stride 4 / 5 | 1.7541 / 1.7688 | 1.7591 / 1.7956 |

Adding code costs about 0.05 BPB on WikiText for the BLT (the entropy model improves slightly), and seed 2 is again the worst seed (as on TinyStories), so seed spread is 0.10 BPB here and the comparison with fixed stride is not established.

The eviction result on the mixed models:
- Keeping high-entropy patches is still worse than random and than recency in all three seeds, but the effect is about a third of the prose-only size: +0.0020 to +0.0027 BPB against random at 64 patches and +0.0017 to +0.0026 at 128, all intervals excluding zero.
- Keeping low-entropy patches is no longer better than keeping high-entropy ones: the difference is within noise in every seed (0 of 3 intervals above zero). Recency is the best policy.
- The ablations repeat the pattern (own entropy +0.0018, maximum +0.0040, first byte +0.0020, sum +0.0033 against random at 64 patches); sink size has no effect.
- On code validation files (seed 0, 64 patches) high entropy is worse than random by +0.0036 [+0.0030, +0.0044] and than recency by +0.0041.
- Precision: uniform 4-bit costs +0.0003 BPB at 27 percent of the bytes; recent-window tiers 1.7216 against high-entropy tiers 1.7409 and random tiers 1.7365 at half the patches at high width.

## Three seeds for every model, and TinyStories with code (`results/part3/seeds.md`)

Every patching scheme now has three seeds on all four settings (TinyStories and WikiText, with and without the 15 percent code mix; `scripts/run_controls.sh`, `scripts/run_tsmix.sh`). The TinyStories entropy model and BLT were retrained with the code mix as well, so both datasets are treated alike.
- Over the 72 comparisons (9 models in each of 4 settings, 2 budgets), keeping high-entropy patches is worse than recency with the interval above zero in 70 and indistinguishable in 2; it is worse than random in 68, indistinguishable in 4, and never better than either. The indistinguishable cases are the WikiText code-mix fixed-stride models against random (stride 4: 5 of 6 budget and seed cells worse; stride 5: 3 of 6) and the TinyStories code-mix stride-5 models against recency (4 of 6). The tallies are in `seeds.md`.
- Keeping low-entropy patches beats keeping high-entropy ones (interval above zero) in all 6 cells for entropy patching on TinyStories with and without code, on WikiText without code, and for fixed strides on TinyStories; for fixed strides on WikiText, and for all schemes on WikiText with code, it is 0 of 6. Over all 72 comparisons keeping low entropy is better in 42, worse in 19 and indistinguishable in 11. So the earlier statement that low-entropy retention is a good policy depends on the data and the patching.
- TinyStories with code: BLT 0.9114 mean against 0.8183 without code (seed spread 0.047), so the mix costs about 0.09 BPB on TinyStories (the corpus is simple, so 15 percent of the batches spent on code matters more) and 0.05 on WikiText. On code files the TinyStories-mix model scores 1.8842; keeping high entropy is worse than random by +0.0103 at 64 patches and +0.0077 at 128.

## Attention-score baselines (`python scripts/compare.py attention`, `results/part3/attention_policies.md`)

Two policies that use the model's own attention, implemented as scoring rules over the patch cache: h2o keeps the patches with the largest accumulated attention received so far (Zhang et al., averaged over heads and layers) and snap keeps the patches that the last 16 patches attend to most (SnapKV-style). Both protect the first patch and the 8 newest, like every other policy. In this section they are offline approximations: scores come from a full-cache pass refined once under the resulting mask (`--attn-iters 2`), not from attention computed online under the evicted cache. The online section below repeats them with attention accumulated during patch-by-patch decoding and finds the same ranking (snap online is never worse than snap offline).

Paired differences against recency (negative means the attention policy is better), over all models and both budgets of each set:
- Over the 72 comparisons (9 models in each of 4 settings, 2 budgets), snap is better than recency with the interval below zero in 67, indistinguishable in 5 (all at the larger budget, four of them on WikiText) and worse in none, by 0.0001 to 0.0069 BPB (for example TinyStories entropy seed 1 at 32 patches: -0.0048 [-0.0054, -0.0043]; WikiText seed 1 at 64: -0.0017 [-0.0023, -0.0011]). It is better than random in 68, indistinguishable in 3 and worse in 1, by up to 0.0134.
- h2o is worse than recency in 60 of the 72 comparisons (up to +0.0200), indistinguishable in 4 and better in 8 (by 0.0005 to 0.0038, nearly all on the TinyStories code-mix models and stride-5 models at 32 or 64 patches). Against random it is worse in 53, indistinguishable in 17 and better in 2. Accumulated attention favours old patches that were popular early, as the entropy-based rules do.
- At byte level (stride-1 model, three seeds) h2o is worse than random at 32 and 64 entries in every seed (seed 0: 1.1902 against 1.1110 at 32). snap beats recency at 32 and 64 entries in every seed (seed 0: 0.9298 against 0.9506 at 32) and loses to it at 288 entries in every seed (seed 0: 0.7617 against 0.7494).
- Effect sizes are small. For the seed-0 TinyStories entropy model at 32 patches, recency costs 0.0098 BPB over the full cache and snap recovers 0.0041 of it. A policy that uses the model's own recent attention helps a little where a static content score does not.

## Longer context (`python scripts/compare.py long`, `results/part3/long_context.md`; tuning in `results/part3/scale_and_tuning.md`)

WikiText models with the code mix trained and evaluated on 16,384-byte chunks (`configs/blt_wikitext103mix_L16k_*`, `scripts/run_long.sh`, `scripts/run_scale.sh`, `scripts/run_scale2.sh`), budgets 64 to 512 patches, one seed. The first 16k models (batch size 2, same bytes seen as the 4k models) scored worse than the 4k models (1.8920 and 1.8971 against 1.7190 and 1.7541). That was an optimisation problem. Tuning on validation (learning rate 0.001 or 0.0003, batch size 2 or 8):

| Run | Batch | Learning rate | Validation BPB |
|---|---|---|---|
| L16k entropy | 2 | 0.001 | 1.8920 |
| L16k entropy | 2 | 0.0003 | 2.0019 |
| L16k entropy | 8 | 0.001 | 1.5843 |
| L16k entropy | 8 | 0.0003 | 1.8213 |
| L16k fixed stride 4 | 2 | 0.001 | 1.8971 |
| L16k fixed stride 4 | 8 | 0.001 | 1.6048 |

Batch 8 with learning rate 0.001 gives 1.5843 (entropy patching) and 1.6048 (fixed), better than the 4k models, and I use these as the long-context models. The batch-8 runs see four times more bytes than the 4k models, so part of the gain is more training, not context length; I did not run a 4k model with the same bytes (the 10x-training section below does this for 4k). The earlier long-context conclusions were drawn on the under-trained models and are withdrawn: keeping high entropy does not become harmless at long context.
- Entropy patching, batch 8: keeping high entropy is worse than random by +0.0192 [+0.0176, +0.0209] at 64 patches, +0.0213 at 128 and 256 and +0.0161 at 512, and worse than recency by 0.041 BPB at 64 patches and 0.025 at 512 (differences of the means, 1.6414 against 1.6001 and 1.6104 against 1.5854).
- Fixed stride, batch 8: worse than random by +0.0186 (64), +0.0198 (128), +0.0174 (256), +0.0131 (512).
- snap beats recency at 64 patches (entropy patching -0.0016 [-0.0026, -0.0006], fixed -0.0020 [-0.0029, -0.0012]) and ties it from 128 patches up. h2o is worse than recency at every budget (+0.013 to +0.033).
- Even with 512 patches (about 2,300 bytes of text, a seventh of a 16,384-byte chunk) the high-entropy rule still loses 0.016 BPB to random.

## Ten times more training (`python scripts/compare_deploy.py scale`, `results/part3/scale_and_tuning.md`)

The main models are small and short-trained (3,000 steps, about 98M bytes), so I trained WikiText code-mix models for 30,000 steps (about 983M bytes, 1.9 epochs of the 526 MB training file; `*_x10` configs, `scripts/run_scale.sh`, `scripts/run_scale2.sh`) with three seeds of entropy patching and one of fixed stride 4, and evaluated them on the same 60 validation documents.

| Model | Full-cache BPB | entropy - random at 64 / 128 patches | entropy - recency at 64 | snap - recency at 64 / 128 |
|---|---|---|---|---|
| Entropy patching, 3,000 steps, seed 0 | 1.7190 | +0.0020 / +0.0017 | +0.0053 | -0.0012 / -0.0003 |
| Entropy patching, 30,000 steps, seed 0 | 1.2949 | +0.0287 / +0.0260 | +0.0562 | -0.0051 / -0.0020 |
| Entropy patching, 30,000 steps, seed 1 | 1.2983 | +0.0284 / +0.0258 | +0.0544 | -0.0057 / -0.0021 |
| Entropy patching, 30,000 steps, seed 2 | 1.3118 | +0.0238 / +0.0215 | +0.0478 | -0.0049 / -0.0019 |
| Fixed stride 4, 30,000 steps, seed 0 | 1.3571 | +0.0098 / +0.0078 | +0.0277 | -0.0042 / -0.0014 |

- The better-trained models score 1.29 to 1.31 BPB, better than the Part 2 BPE transformer (1.5200 on its WikiText validation set; the code mix is in training only). The seed spread is 0.017 (3,000 steps: 0.102).
- The entropy penalty grows more than tenfold with training (+0.002 to +0.029 against random at 64 patches) and every interval excludes zero in all three seeds, so a small, short-trained model is not what made the entropy rule look bad. The cost of any eviction also grows: recency at 64 patches costs 0.018 BPB over the full cache (3,000 steps: 0.002).
- snap beats recency in all three seeds and both budgets, by several times as much as before. Entropy patching is below fixed stride by 0.062 BPB (one fixed-stride seed; at 3,000 steps the gap is 0.035 for seed 0 and 0.025 for the three-seed means).
- The same experiment on TinyStories is not usable. The training set is 26.7 MB, so 30,000 steps are 37 epochs; the entropy-patched model reached a training loss of 0.14 nats and a validation BPB of 1.6493 (0.7980 at 3,000 steps), and the fixed-stride model 1.8765. Those rows are in `scale_and_tuning.md` for completeness; I draw no conclusions from them.

## Quantization and attention across seeds

Quantization on ten entropy-patched models (three seeds each of TinyStories, WikiText and WikiText with code, plus seed 0 of TinyStories with code): uniform 4-bit costs 0.0003 to 0.0014 BPB over fp16 at 27 percent of the bytes, uniform 3-bit 0.0014 to 0.0077, uniform 2-bit 0.022 to 0.111. For half the patches at 4 bits and the rest at 2 bits, the ranking from best to worst is the same in all ten models: recent-window tiers, then keep-low-entropy tiers, then random tiers, then keep-high-entropy tiers.

Attention correlation with patch entropy (Spearman, patches at least 8 patches old):
- TinyStories, three seeds: negative in all six layers in all three seeds (-0.04 to -0.55).
- WikiText without code, three seeds: negative in layers 2 to 5 in all three seeds (-0.007 to -0.39); layers 0 and 1 vary in sign (layer 0: -0.06, -0.06, +0.13).
- TinyStories with code, seed 0: negative in all six layers (-0.11 to -0.43).
- WikiText with code, three seeds: negative in layers 2 to 4 for seeds 1 and 2 (-0.15 to -0.39), and in layer 5 for seed 1; layers 0 and 1 positive for seeds 1 and 2 (+0.00 to +0.28). Seed 0 is the outlier found earlier: positive in every layer except layer 1 (-0.34).
So the observation that old patches with high entropy receive less attention holds in the middle and upper layers of most models; the earlier statement that the code mix reverses it rested on one seed (an outlier) and is withdrawn. The evidence is still correlational.

## Test split (`python scripts/compare.py test`, `results/part3/test_split.md`)

WikiText-103 test, 62 documents, evaluated once (`scripts/run_test.sh`) for the 18 WikiText models (entropy patching, fixed strides 4 and 5, three seeds each, with and without code) with the six policies and two budgets fixed on validation beforehand; nothing was tuned on it. TinyStories has no test split in the byte data, so its results are validation only.
- Keeping high-entropy patches is worse than random and than recency in all 36 model and budget cells, with every interval above zero (against random +0.0010 to +0.0124, against recency up to +0.0192).
- snap beats recency in 33 of 36 cells (the other three are indistinguishable), by 0.0001 to 0.0039 BPB. h2o is worse than recency in all cells.
- Full-cache test BPB is 0.02 to 0.03 higher than validation for the prose-only entropy models (1.6986, 1.7273, 1.7372 against 1.6801, 1.6994, 1.7080). Entropy patching is below both fixed strides in all three seed pairs for the prose-only models, and in 5 of 6 pairs with the code mix (the exception is seed 2 against stride 4, 1.8435 against 1.8318), the same pattern as on validation.

## Online attention policies (`python scripts/compare_deploy.py online`, `results/part3/online_policies.md`)

The offline h2o and snap scores were replaced by online ones: `BLT.forward_online` decodes the global stack patch by patch with a real cache, chooses the entry to drop before the new patch is attended to, and accumulates attention rows as they are produced (h2o: all rows so far; snap: the last 16). Tests check that the online path equals step-by-step decoding for recency and random, that it is causal for h2o and snap, and that it equals the full cache at a large budget. On the 12 entropy-patched models (300 TinyStories documents, 60 WikiText documents, two budgets each, 24 comparisons):
- snap online is better than recency in all 24 comparisons (by 0.0002 to 0.0077 BPB; WikiText with code, seed 0, 64 patches: -0.0013 [-0.0018, -0.0009]).
- h2o online is worse than recency in 22 of 24, tied in 1 and better in 1 (from -0.0018 to +0.0134), so the offline conclusion holds.
- snap online is never worse than snap offline and is at most 0.0009 BPB better (7 of 24 comparisons favour the online version, 17 tie), so the offline tables were a good approximation.
- recency through the online path equals the masked recency run to four decimals in all 24 comparisons (two have intervals that exclude zero at a difference below 0.0001), a check that the online loop reproduces the masks.

## Eviction with 4-bit quantization (`python scripts/compare_deploy.py combo`, `results/part3/deploy_combo.md`)

Eviction and uniform 4-bit K and V together, mean of three seeds per dataset (cache bytes computed, snap scored offline here):

| Setting | TinyStories cost over full cache (share of bytes) | WikiText cost over full (share of bytes) |
|---|---|---|
| recency, 64 patches, 4-bit | +0.0072 (15%) | +0.0046 (3.9%) |
| snap, 64 patches, 4-bit | +0.0049 (15%) | +0.0026 (3.9%) |
| recency, 128 patches, 4-bit | n/a | +0.0019 (7.4%) |
| snap, 128 patches, 4-bit | n/a | +0.0012 (7.4%) |
| keep high entropy, 64 patches, 4-bit | +0.0224 (15%) | +0.0196 (3.9%) |

Full-cache BPB is 0.8183 (TinyStories) and 1.6958 (WikiText). A WikiText cache of 128 patches at 4-bit uses 7.4 percent of the full cache and costs 0.0012 to 0.0019 BPB. Quantization and eviction costs add up roughly linearly; keeping high-entropy patches is the worst policy at every budget, with or without quantization.

## Bounded-memory streaming (`python scripts/compare_deploy.py stream`, `results/part3/stream_16k.md`)

The question for deployment is whether a cache of fixed size can follow a document longer than the context the model was trained on. Three WikiText code-mix models trained on 4,096-byte chunks decoded 46 validation documents longer than 8,192 bytes in 16,384-byte chunks with a fixed patch budget, so patch positions reach about 3,600 (the models saw about 900). The baseline resets the context every 4,096 bytes with a full cache.
- Recency with 64 patches beats the chunked full-cache baseline in all three seeds (by 0.0030, 0.0032 and 0.0008 BPB), and with 256 patches by 0.0051, 0.0058 and 0.0035. The reset loses the context at every chunk boundary; a small continuous window does not, and the model's extrapolated patch positions did not hurt.
- snap online is better still: at 64 patches it is 0.0014, 0.0008 and 0.0024 below recency with intervals excluding zero, and 0.0044, 0.0040 and 0.0032 below the chunked baseline. At 256 patches snap and recency tie.
- Random (about +0.003 to +0.005 against recency), h2o (+0.004 to +0.009) and keeping high entropy (+0.006 to +0.009) are all worse than recency at every budget, in all three seeds, with intervals above zero.

## Measured decoding (`python scripts/compare_deploy.py bench`, `results/part3/deploy_bench.md`)

Step-by-step decoding of one stream on the cluster GPU (batch 1, Python loop, recency eviction with real removal from the cache tensors; seed-0 WikiText code-mix model). The global cache after 65,536 bytes is 136.5 MB with a full cache against 0.59, 1.18 and 2.36 MB with 64, 128 and 256 patches, matching the computed sizes. Time per byte stays at 0.9 ms with a bounded cache and rises from 0.9 ms (2,048 bytes) to 2.0 ms (65,536) with a full cache; total time for 65,536 bytes was 102 s against 60 s. The byte window caches are a fixed 786 KB, so with a bounded cache they are larger than the global cache at 64 patches, and the separate entropy model adds about 2.1 MB of window cache and 3.3M parameters. Memory is the main saving and speed matters only at long contexts. `results/part3/deploy_memory.md` gives computed sizes per context: the patch cache at full precision is about 0.9 times a BPE transformer cache of the same width, and a byte-level transformer needs about 4 times more.

## Recency-protected entropy (`python scripts/compare_deploy.py hybrid`, `results/part3/deploy_hybrid.md`)

The budget is split: the newest half is protected, and the other half goes to high-entropy patches (hybrid entropy) or random patches (hybrid random).
- Hybrid entropy is worse than plain recency in 23 of 24 model and budget cells and tied in 1 (for example TinyStories seed 0, 32 patches: +0.0047 [+0.0040, +0.0054]), and worse than hybrid random in all 24.
- Hybrid random beats plain recency in 11 of 12 TinyStories cells (with and without code; seed 0: -0.0012 and -0.0013) and is worse in 5 of 12 and tied in 7 of 12 WikiText cells (with and without code). On TinyStories a few old patches chosen at random help more than a slightly longer recency window; entropy is not a better way to choose them.

## Ablations on seeds 1 and 2 (`python scripts/compare_deploy.py ablseeds`, `results/part3/ablations_seeds.md`)

- The main rule (mean over the patch), the model's own entropy, first-byte entropy and sink size 0 or 4 are worse than random in all three seeds on both datasets.
- Maximum over the patch: worse than random in 3 of 3 seeds on WikiText, 2 of 3 on TinyStories (seed 1: -0.0012, better than random).
- Sum over the patch: never worse than random on TinyStories (0 of 3 seeds; seed 1: -0.0024), worse in 3 of 3 on WikiText. Sum and maximum are the only rules that are ever better than random, in one seed each, and both only on TinyStories.

## Planted-name recall (`python scripts/compare_deploy.py needle`, `results/part3/deploy_needle.md`)

A random four-syllable name is introduced early and repeated after 400, 1,200 or 3,000 filler bytes; the score is the BPB of the name after its first two letters, 40 trials per distance. The models do not use the earlier mention: with a full cache the score equals the no-cue control (for example TinyStories seed 0 at 400 bytes: 9.24 bits per byte against 9.27 for the control), so no policy can be shown to help and nothing about retrieval can be concluded at this scale. The entropy rule does keep the planted name's patches more often than random does (45 percent against 31 percent at 64 patches and a distance of 400 bytes, 18 against 11 percent at 1,200), but the model gains nothing from them.

## Long books: PG-19 (`python scripts/compare_deploy.py pg19`, `results/part3/pg19_stream.md`)

The proposal named PG-19 for long-context cache evaluation. The first 100,000 bytes of each of the first 20 test books (`emozilla/pg19`, a community mirror of the same split, because `deepmind/pg19` ships as a loading script that current `datasets` no longer runs; 100 books, 40.5 MB, split and hash checks pass) were decoded one byte at a time with `BLT.step` and a real fixed-size cache (`scripts/eval_stream.py`, `scripts/run_pg19.sh`, `runs/<model>/stream_pg19_*.json`). Patch positions reach about 22,000, against roughly 900 in the 4,096-byte training chunks. The models are the WikiText code-mix models, so these pre-1919 books are out of domain and BPB is high; 95 percent intervals are bootstrapped over the 20 books and every comparison is paired.

| Setting (BPB) | 30,000-step model | 3,000-step model |
|---|---|---|
| Full cache, context reset every 4,096 bytes | 2.2097 | 2.5566 |
| Full cache for the whole book | 2.3467 | 2.5763 |
| Recency, 64 / 128 / 256 patches | 2.2265 / 2.2107 / 2.2012 | 2.5538 / 2.5522 / 2.5512 |
| Random, 64 / 128 / 256 patches | 2.3069 / 2.2870 / 2.2692 | 2.5578 / 2.5566 / 2.5567 |
| Keep high entropy, 64 / 128 / 256 patches | 2.3193 / 2.3160 / 2.3227 | 2.5789 / 2.5788 / 2.5795 |

- Recency is the only policy that is safe on long books. For the 30,000-step model, 256 patches beat the reset baseline by 0.0086 BPB [0.0079, 0.0093] and the whole-book full cache by 0.145; 128 patches tie the reset baseline (+0.0009) and 64 are 0.0168 worse. For the 3,000-step model recency beats the reset baseline at all three budgets, by 0.0028 to 0.0054.
- The whole-book full cache is worse than resetting the context for both models (+0.137 and +0.020 BPB), so the model does not use a context 24 times longer than it was trained on; a bounded recency cache avoids that.
- Keeping high-entropy patches is the worst policy at every budget: +0.093 to +0.122 against recency for the 30,000-step model and +0.025 to +0.028 for the 3,000-step model, all intervals above zero. Random patches are also worse than recency (+0.068 to +0.081 and +0.004 to +0.006).
- Caveat: keeping old patches (random or entropy) means attending across distances far beyond the training range, so this experiment cannot separate "entropy is a poor keep signal" from "old patches are hard for this model at these positions". The remedy of renumbering positions inside the cache (as StreamingLLM does) was not tried. SnapKV-style and H2O were not run on PG-19 because `BLT.step` has no attention policies.

## Requeue test (Block 2.0)

Verified on the cluster under a single job id. Job 416139 (`configs/requeue_cluster4.json`, 24,000 steps) was sent SIGUSR1 with `scancel --signal=USR1 --batch` while running. It saved a checkpoint at step 1,864, wrote the PREEMPTED marker and was requeued by the batch script (restart count 1). SLURM held it for its restart delay (about 10 minutes), then restarted the same job id; the log printed "resumed at step 1864" and training ran to step 24,000 (final loss 0.1845), with `meta.json` recording both attempts (`ended_step` 1864, then `resumed_from_step` 1864 and `ended_step` 24000, status done). Earlier smoke jobs (407961 and 408216, 7-minute limit) showed the same path across two job ids. Two attempts to repeat the single-id test were not signalled because my signalling script failed (jobs 415084 and 416117 ran straight through), and one earlier attempt reused the old run name `requeue_test`, which overwrote that smoke run's `config.json` (its log and checkpoint were kept; the smoke config is `configs/smoke.json`). The first smoke attempt recorded ending at step 1; I did not find out why. The local test (resume equals uninterrupted) covers equality of the resumed run.

## Why entropy does not work as a keep signal here

Attention received by patches at least 8 patches old correlates negatively with patch entropy in the middle and upper global layers of most models (section above; `results/part3/attention_*.md`). In the TinyStories seed-0 model the highest-entropy third of old patches gets about half the attention of the lowest third in layers 1 and 2. That suggests the model reads predictable patches more than surprising ones, with rare bytes resolved locally by the 256-byte windows, and it fits the result that keeping high-entropy patches is worse than random. Two things weaken it: it is a correlation (no intervention tests it), and the WikiText code-mix seed 0 model shows the opposite sign in most layers while its eviction result keeps the same sign. What is stable is the outcome: entropy is never better than random or recency. The attention policies that do help (snap) use the model's own recent attention, which is consistent with attention, not surprise, being what the model needs from old patches.

## Caveats

- Three seeds for every patching scheme and setting; quantization and attention analysis use three seeds only where stated, the ablations and the code-domain tables use seed 0, the byte-level model has three seeds, the 16k models one.
- Seed 0 of the prose-only models was trained on a laptop GPU, the rest on the cluster. Seed 2 is the worst or second-worst seed for most schemes.
- Small models (13M parameters). Most results use 3,000 training steps and 4,096-byte contexts; the 10x-training WikiText models (30,000 steps; three seeds of entropy patching, one of fixed stride) and the tuned 16,384-byte models (one seed each) give the same sign with larger effects. The TinyStories 10x models overfit and are not used. Budgets and gains are specific to these sizes.
- Eviction and quantization are simulated by masks and rounding, checked equal to step-by-step decoding in tests; cache bytes are computed, not measured. The offline attention policies (h2o, snap in the older tables) are scored from a full-cache pass refined once. The online versions run in `forward_online`, tested equal to step-by-step decoding for recency and random and tested for causality and equality to the full cache at a large budget for h2o and snap; the step-by-step `BLT.step` path does not implement attention policies, so there is no step-decoding test for them.
- The own-entropy ablation scores patches from a full-cache pass.
- Code results use 200 validation files; the TinyStories and WikiText code-mix models share one 15 percent code share and thresholds tuned on their own validation sets. No test split has been evaluated: TinyStories has no test split in the byte data.
- The requeue path is verified end to end under one job id (signal, checkpoint, requeue, resume, finish).
