Variants of entropy-based and attention-based cache policies on the entropy-patched models (three seeds), validation only. Paired BPB difference against the comparison policy, mean and 95% document bootstrap interval (2000 resamples); negative means the variant is better. The first patch and the 8 newest are always kept. The dynamic variant lets the entropy of the newest patch set the budget (1.5 times the base budget above the median entropy, 0.5 times below); its mean cache size is lower than the base budget and its peak is 1.5 times the budget, so it is compared with recency linearly interpolated, per document, to the same mean number of entries.

## tinystories (budget 32 patches)

| Variant | Comparison | s0 | s1 | s2 | Better in |
|---|---|---|---|---|---|
| entropy | recency | +0.0152 [+0.0140, +0.0164] | +0.0234 [+0.0219, +0.0249] | +0.0283 [+0.0262, +0.0307] | 0/3 |
| floor3 | recency | +0.0072 [+0.0060, +0.0084] | +0.0109 [+0.0096, +0.0123] | +0.0126 [+0.0107, +0.0146] | 0/3 |
| floor3 | entropy | -0.0080 [-0.0091, -0.0070] | -0.0126 [-0.0140, -0.0113] | -0.0157 [-0.0174, -0.0139] | 3/3 |
| floor4 | recency | +0.0051 [+0.0040, +0.0062] | +0.0087 [+0.0074, +0.0101] | +0.0090 [+0.0074, +0.0108] | 0/3 |
| floor4 | entropy | -0.0101 [-0.0112, -0.0091] | -0.0147 [-0.0161, -0.0133] | -0.0193 [-0.0213, -0.0174] | 3/3 |
| snap | recency | -0.0041 [-0.0046, -0.0035] | -0.0048 [-0.0054, -0.0043] | -0.0038 [-0.0046, -0.0031] | 3/3 |
| snap | entropy | -0.0193 [-0.0207, -0.0180] | -0.0282 [-0.0299, -0.0266] | -0.0322 [-0.0344, -0.0300] | 3/3 |
| hybrid0.25 | recency | -0.0015 [-0.0021, -0.0008] | -0.0016 [-0.0022, -0.0010] | +0.0001 [-0.0011, +0.0015] | 2/3 |
| hybrid0.25 | entropy | -0.0167 [-0.0179, -0.0156] | -0.0250 [-0.0265, -0.0235] | -0.0283 [-0.0301, -0.0265] | 3/3 |
| hybrid0.5 | recency | +0.0023 [+0.0015, +0.0031] | +0.0035 [+0.0026, +0.0043] | +0.0060 [+0.0046, +0.0076] | 0/3 |
| hybrid0.5 | entropy | -0.0129 [-0.0138, -0.0119] | -0.0200 [-0.0212, -0.0188] | -0.0223 [-0.0240, -0.0208] | 3/3 |
| dynamic | recency (matched size) | +0.0003 [+0.0001, +0.0005] | +0.0004 [+0.0001, +0.0006] | +0.0007 [+0.0003, +0.0010] | 0/3 |
| dynamic | entropy | -0.0031 [-0.0042, -0.0020] | -0.0090 [-0.0104, -0.0077] | -0.0021 [-0.0046, -0.0001] | 3/3 |

Mean BPB and mean resident cache entries, seed 0:

| Variant | BPB | Entries |
|---|---|---|
| full|0 | 0.7980 | 91.1 |
| recency|32 | 0.8078 | 28.9 |
| entropy|32 | 0.8231 | 28.9 |
| floor3|32 | 0.8150 | 28.9 |
| floor4|32 | 0.8129 | 28.9 |
| snap|32 | 0.8037 | 28.9 |
| hybrid0.25|32 | 0.8064 | 28.9 |
| hybrid0.5|32 | 0.8102 | 28.9 |
| dynamic|32 | 0.8199 | 16.3 |
| recency|12 | 0.8314 | 11.6 |
| recency|16 | 0.8214 | 15.2 |
| recency|20 | 0.8157 | 18.8 |
| recency|24 | 0.8121 | 22.2 |

## wikitext103 (budget 64 patches)

| Variant | Comparison | s0 | s1 | s2 | Better in |
|---|---|---|---|---|---|
| entropy | recency | +0.0087 [+0.0077, +0.0097] | +0.0192 [+0.0173, +0.0209] | +0.0169 [+0.0148, +0.0193] | 0/3 |
| floor3 | recency | +0.0049 [+0.0043, +0.0055] | +0.0096 [+0.0088, +0.0105] | +0.0089 [+0.0079, +0.0099] | 0/3 |
| floor3 | entropy | -0.0038 [-0.0048, -0.0028] | -0.0095 [-0.0111, -0.0079] | -0.0081 [-0.0103, -0.0061] | 3/3 |
| floor4 | recency | +0.0046 [+0.0038, +0.0055] | +0.0093 [+0.0084, +0.0101] | +0.0082 [+0.0073, +0.0092] | 0/3 |
| floor4 | entropy | -0.0041 [-0.0053, -0.0027] | -0.0099 [-0.0120, -0.0077] | -0.0087 [-0.0114, -0.0062] | 3/3 |
| snap | recency | -0.0014 [-0.0019, -0.0010] | -0.0017 [-0.0023, -0.0011] | -0.0030 [-0.0036, -0.0025] | 3/3 |
| snap | entropy | -0.0101 [-0.0114, -0.0090] | -0.0209 [-0.0229, -0.0188] | -0.0199 [-0.0227, -0.0173] | 3/3 |
| hybrid0.25 | recency | -0.0002 [-0.0008, +0.0003] | +0.0004 [-0.0002, +0.0010] | -0.0013 [-0.0018, -0.0008] | 1/3 |
| hybrid0.25 | entropy | -0.0089 [-0.0101, -0.0079] | -0.0187 [-0.0205, -0.0169] | -0.0182 [-0.0208, -0.0157] | 3/3 |
| hybrid0.5 | recency | +0.0013 [+0.0007, +0.0018] | +0.0050 [+0.0041, +0.0058] | +0.0029 [+0.0023, +0.0034] | 0/3 |
| hybrid0.5 | entropy | -0.0074 [-0.0084, -0.0065] | -0.0142 [-0.0158, -0.0127] | -0.0140 [-0.0163, -0.0120] | 3/3 |
| dynamic | recency (matched size) | -0.0000 [-0.0001, -0.0000] | +0.0001 [+0.0000, +0.0001] | +0.0000 [-0.0000, +0.0000] | 1/3 |
| dynamic | entropy | -0.0053 [-0.0062, -0.0044] | -0.0136 [-0.0152, -0.0120] | -0.0119 [-0.0140, -0.0100] | 3/3 |

Mean BPB and mean resident cache entries, seed 0:

| Variant | BPB | Entries |
|---|---|---|
| full|0 | 1.6801 | 409.9 |
| recency|64 | 1.6836 | 60.6 |
| entropy|64 | 1.6922 | 60.6 |
| floor3|64 | 1.6885 | 60.6 |
| floor4|64 | 1.6882 | 60.6 |
| snap|64 | 1.6822 | 60.6 |
| hybrid0.25|64 | 1.6834 | 60.6 |
| hybrid0.5|64 | 1.6849 | 60.6 |
| dynamic|64 | 1.6870 | 32.3 |
| recency|20 | 1.6906 | 19.7 |
| recency|32 | 1.6872 | 31.1 |
| recency|40 | 1.6859 | 38.6 |
| recency|48 | 1.6850 | 46.1 |

