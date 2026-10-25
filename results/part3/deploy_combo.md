Eviction and 4-bit quantization together (uniform 4-bit K and V, per-entry scale in fp16, all kept entries quantized). Cache bytes are computed from entry counts and bit widths. BPB is the mean over the three seeds of entropy patching; the interval is the seed-0 model's 95% document bootstrap (2000 resamples). Validation only. Cost is BPB minus the full fp16 cache of the same model.

## tinystories (3 seeds, mean full-cache BPB 0.8183, mean full cache 839 KB)

| Setting | Mean cache (KB) | Share of full | BPB (mean of seeds) | Cost over full | Seed 0 BPB [95% CI] |
|---|---|---|---|---|---|
| entropy 32 patches, 4-bit | 71 | 8.4% | 0.8604 | +0.0421 | 0.8104 to 0.8379 |
| random 32 patches, 4-bit | 71 | 8.4% | 0.8428 | +0.0245 | 0.7986 to 0.8259 |
| recency 32 patches, 4-bit | 71 | 8.4% | 0.8373 | +0.0189 | 0.7957 to 0.8226 |
| snap 32 patches, 4-bit * | 71 | 8.4% | 0.8321 | +0.0138 | 0.7913 to 0.8183 |
| entropy 64 patches, 4-bit | 125 | 14.9% | 0.8407 | +0.0224 | 0.7980 to 0.8260 |
| random 64 patches, 4-bit | 125 | 14.9% | 0.8288 | +0.0105 | 0.7905 to 0.8177 |
| recency 64 patches, 4-bit | 125 | 14.9% | 0.8256 | +0.0072 | 0.7893 to 0.8162 |
| snap 64 patches, 4-bit * | 125 | 14.9% | 0.8232 | +0.0049 | 0.7871 to 0.8140 |
| entropy 32 patches, fp16 | 266 | 31.7% | 0.8592 | +0.0409 | 0.8093 to 0.8367 |
| random 32 patches, fp16 | 266 | 31.7% | 0.8417 | +0.0234 | 0.7976 to 0.8250 |
| recency 32 patches, fp16 | 266 | 31.7% | 0.8361 | +0.0177 | 0.7944 to 0.8213 |
| snap 32 patches, fp16 | 266 | 31.7% | 0.8309 | +0.0126 | 0.7903 to 0.8173 |
| entropy 64 patches, fp16 | 472 | 56.2% | 0.8395 | +0.0212 | 0.7968 to 0.8248 |
| random 64 patches, fp16 | 472 | 56.2% | 0.8277 | +0.0093 | 0.7897 to 0.8169 |
| recency 64 patches, fp16 | 472 | 56.2% | 0.8244 | +0.0060 | 0.7884 to 0.8152 |
| snap 64 patches, fp16 * | 472 | 56.2% | 0.8221 | +0.0038 | 0.7862 to 0.8131 |
| full cache fp16 * | 839 | 100.0% | 0.8183 | +0.0000 | 0.7846 to 0.8114 |

A row is marked * when no other row has equal or fewer bytes and a strictly lower BPB.

## wikitext103 (3 seeds, mean full-cache BPB 1.6958, mean full cache 3,777 KB)

| Setting | Mean cache (KB) | Share of full | BPB (mean of seeds) | Cost over full | Seed 0 BPB [95% CI] |
|---|---|---|---|---|---|
| entropy 64 patches, 4-bit | 148 | 3.9% | 1.7154 | +0.0196 | 1.6523 to 1.7324 |
| random 64 patches, 4-bit | 148 | 3.9% | 1.7055 | +0.0097 | 1.6457 to 1.7264 |
| recency 64 patches, 4-bit | 148 | 3.9% | 1.7005 | +0.0046 | 1.6432 to 1.7240 |
| snap 64 patches, 4-bit * | 148 | 3.9% | 1.6985 | +0.0026 | 1.6419 to 1.7226 |
| entropy 128 patches, 4-bit | 281 | 7.4% | 1.7109 | +0.0151 | 1.6485 to 1.7285 |
| random 128 patches, 4-bit | 281 | 7.4% | 1.7014 | +0.0055 | 1.6430 to 1.7234 |
| recency 128 patches, 4-bit | 281 | 7.4% | 1.6977 | +0.0019 | 1.6411 to 1.7216 |
| snap 128 patches, 4-bit * | 281 | 7.4% | 1.6970 | +0.0012 | 1.6408 to 1.7213 |
| entropy 64 patches, fp16 | 558 | 14.8% | 1.7149 | +0.0191 | 1.6518 to 1.7321 |
| random 64 patches, fp16 | 558 | 14.8% | 1.7051 | +0.0093 | 1.6452 to 1.7260 |
| recency 64 patches, fp16 | 558 | 14.8% | 1.7001 | +0.0042 | 1.6427 to 1.7236 |
| snap 64 patches, fp16 | 558 | 14.8% | 1.6980 | +0.0022 | 1.6414 to 1.7221 |
| entropy 128 patches, fp16 | 1,059 | 28.0% | 1.7104 | +0.0146 | 1.6480 to 1.7281 |
| random 128 patches, fp16 | 1,059 | 28.0% | 1.7010 | +0.0051 | 1.6425 to 1.7232 |
| recency 128 patches, fp16 | 1,059 | 28.0% | 1.6973 | +0.0015 | 1.6407 to 1.7213 |
| snap 128 patches, fp16 * | 1,059 | 28.0% | 1.6966 | +0.0008 | 1.6404 to 1.7210 |
| full cache fp16 * | 3,777 | 100.0% | 1.6958 | +0.0000 | 1.6394 to 1.7199 |

A row is marked * when no other row has equal or fewer bytes and a strictly lower BPB.

## tinystoriesmix (3 seeds, mean full-cache BPB 0.9114, mean full cache 797 KB)

| Setting | Mean cache (KB) | Share of full | BPB (mean of seeds) | Cost over full | Seed 0 BPB [95% CI] |
|---|---|---|---|---|---|
| entropy 32 patches, 4-bit | 70 | 8.8% | 0.9307 | +0.0194 | 0.9030 to 0.9316 |
| random 32 patches, 4-bit | 70 | 8.8% | 0.9212 | +0.0099 | 0.8939 to 0.9222 |
| recency 32 patches, 4-bit | 70 | 8.8% | 0.9201 | +0.0087 | 0.8958 to 0.9238 |
| snap 32 patches, 4-bit * | 70 | 8.8% | 0.9161 | +0.0048 | 0.8918 to 0.9199 |
| entropy 64 patches, 4-bit | 124 | 15.5% | 0.9219 | +0.0105 | 0.8954 to 0.9239 |
| random 64 patches, 4-bit | 124 | 15.5% | 0.9155 | +0.0041 | 0.8903 to 0.9182 |
| recency 64 patches, 4-bit | 124 | 15.5% | 0.9148 | +0.0034 | 0.8910 to 0.9187 |
| snap 64 patches, 4-bit * | 124 | 15.5% | 0.9128 | +0.0014 | 0.8893 to 0.9174 |
| entropy 32 patches, fp16 | 264 | 33.2% | 0.9302 | +0.0188 | 0.9025 to 0.9312 |
| random 32 patches, fp16 | 264 | 33.2% | 0.9206 | +0.0093 | 0.8934 to 0.9218 |
| recency 32 patches, fp16 | 264 | 33.2% | 0.9195 | +0.0081 | 0.8953 to 0.9234 |
| snap 32 patches, fp16 | 264 | 33.2% | 0.9155 | +0.0042 | 0.8913 to 0.9194 |
| entropy 64 patches, fp16 | 466 | 58.4% | 0.9213 | +0.0099 | 0.8947 to 0.9234 |
| random 64 patches, fp16 | 466 | 58.4% | 0.9149 | +0.0035 | 0.8898 to 0.9177 |
| recency 64 patches, fp16 | 466 | 58.4% | 0.9142 | +0.0029 | 0.8906 to 0.9184 |
| snap 64 patches, fp16 * | 466 | 58.4% | 0.9122 | +0.0008 | 0.8888 to 0.9170 |
| full cache fp16 * | 797 | 100.0% | 0.9114 | +0.0000 | 0.8881 to 0.9160 |

A row is marked * when no other row has equal or fewer bytes and a strictly lower BPB.

## wikitext103mix (3 seeds, mean full-cache BPB 1.7482, mean full cache 3,778 KB)

| Setting | Mean cache (KB) | Share of full | BPB (mean of seeds) | Cost over full | Seed 0 BPB [95% CI] |
|---|---|---|---|---|---|
| entropy 64 patches, 4-bit | 148 | 3.9% | 1.7553 | +0.0071 | 1.6879 to 1.7650 |
| random 64 patches, 4-bit | 148 | 3.9% | 1.7530 | +0.0048 | 1.6857 to 1.7631 |
| recency 64 patches, 4-bit | 148 | 3.9% | 1.7507 | +0.0025 | 1.6828 to 1.7596 |
| snap 64 patches, 4-bit * | 148 | 3.9% | 1.7494 | +0.0012 | 1.6818 to 1.7583 |
| entropy 128 patches, 4-bit | 281 | 7.4% | 1.7533 | +0.0051 | 1.6858 to 1.7625 |
| random 128 patches, 4-bit | 281 | 7.4% | 1.7512 | +0.0030 | 1.6840 to 1.7609 |
| recency 128 patches, 4-bit | 281 | 7.4% | 1.7492 | +0.0010 | 1.6815 to 1.7580 |
| snap 128 patches, 4-bit * | 281 | 7.4% | 1.7488 | +0.0006 | 1.6812 to 1.7577 |
| entropy 64 patches, fp16 | 558 | 14.8% | 1.7550 | +0.0068 | 1.6876 to 1.7647 |
| random 64 patches, fp16 | 558 | 14.8% | 1.7527 | +0.0045 | 1.6855 to 1.7628 |
| recency 64 patches, fp16 | 558 | 14.8% | 1.7504 | +0.0022 | 1.6825 to 1.7593 |
| snap 64 patches, fp16 | 558 | 14.8% | 1.7491 | +0.0009 | 1.6815 to 1.7580 |
| entropy 128 patches, fp16 | 1,059 | 28.0% | 1.7530 | +0.0048 | 1.6856 to 1.7623 |
| random 128 patches, fp16 | 1,059 | 28.0% | 1.7509 | +0.0027 | 1.6837 to 1.7606 |
| recency 128 patches, fp16 | 1,059 | 28.0% | 1.7489 | +0.0007 | 1.6812 to 1.7578 |
| snap 128 patches, fp16 * | 1,059 | 28.0% | 1.7485 | +0.0003 | 1.6810 to 1.7574 |
| full cache fp16 * | 3,778 | 100.0% | 1.7482 | +0.0000 | 1.6805 to 1.7571 |

A row is marked * when no other row has equal or fewer bytes and a strictly lower BPB.
