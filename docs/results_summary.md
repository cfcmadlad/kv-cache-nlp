# Results summary (validation only, 21 Oct 2026)

All numbers are bits per byte (BPB) on matched validation documents: TinyStories first 500 documents (405,231 bytes) and WikiText-103 all 60 documents (1,129,194 bytes). The test splits have not been touched. Intervals are 95 percent bootstrap intervals over documents. Regenerate any table with the command in its section; the raw tables are in `results/part3/` and `results/phase2.md`.

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

Full-cache BPB for three seeds of the entropy-patched model, and the fixed-stride controls (one seed each):

| Dataset | Entropy patching, seeds 0, 1, 2 (mean) | Fixed stride 4 (4.0 bytes per patch) | Fixed stride 5 (5.0 bytes per patch) |
|---|---|---|---|
| TinyStories | 0.7980, 0.8026, 0.8544 (0.8183) | 0.7726 | 0.8071 |
| WikiText | 1.6801, 1.6994, 1.7080 (1.6958) | 1.7600 | 1.7768 |

Seed spread is 0.057 BPB on TinyStories and 0.028 on WikiText, so single-seed gaps below that are not established. On WikiText entropy patching beats both fixed strides by 0.065 to 0.081 BPB with fewer cache entries per byte than stride 4, which is more than the spread. On TinyStories entropy patching (4.5 bytes per patch) is not better than the fixed strides: stride 5 has fewer entries and 0.8071 sits inside the entropy-patched seed range. The earlier claim that entropy patching is 3.3 percent worse on TinyStories is therefore not a stable difference, and the fixed-stride seed spread is unmeasured.

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

## Why entropy does not work as a keep signal here

Prose-only models, seed 0 (`results/part3/attention_*.md`): attention received by patches at least 8 patches old correlates negatively with patch entropy in all six global layers on TinyStories (-0.04 to -0.38; the highest-entropy third gets about half the attention of the lowest third in layers 1 and 2) and in five of six on WikiText (-0.06 to -0.26). That suggests the model reads predictable patches more than surprising ones, with rare bytes resolved locally by the 256-byte windows.

The code-mix WikiText model does not repeat it (`results/part3/attention_wikitext_mix.md`): correlations are small and positive (+0.02 to +0.06) in five layers and -0.34 in layer 1. So the attention mechanism is not a stable explanation. What is stable is the outcome: entropy is never better than random or recency. Why remains open; one seed per analysis and no intervention.

## Caveats

- Three seeds only for the entropy-patched models; quantization, attention analysis and all ablations use seed 0, and the fixed-stride controls have one seed. Seed 2 is the worst seed on both TinyStories and the WikiText code mix.
- Seed 0 was trained on different hardware from the other seeds.
- Small models (13M parameters, 3,000 steps) and a 4,096-byte context; budgets and gains are specific to that context.
- Eviction and quantization are simulated by masks and rounding, checked equal to step-by-step decoding in tests; cache bytes are computed, not measured.
- The own-entropy ablation scores patches from a full-cache pass.
- The TinyStories models and the first WikiText models were trained without code; the code-mix WikiText models are the only ones with a code result, trained with a 15 percent code share and one threshold tuned on WikiText.
- The Block 2.0 requeue path (SIGUSR1) is untested; the smoke job and all training jobs ran to completion without preemption.
