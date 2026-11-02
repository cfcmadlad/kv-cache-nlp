Further scoring rules at one budget, entropy-patched models, validation only. Surprisal ranks patches by the realised mean negative log-probability of their bytes under the entropy model instead of its predictive entropy. KeyDiff keeps keys dissimilar from an anchor (the mean key of the last 16 patches). Expected attention scores each key by the softmax of its expected logit under a Gaussian fitted to the last 16 queries. Both use pre-rotary keys and queries, a simplification. SnapKV-style with value norm (snapv) multiplies the attention score by the L1 norm of the patch's value vector. Paired 95% document bootstrap intervals (2000 resamples); negative means the variant is better than the comparison.

## tinystories (budget 32 patches)

| Variant | Comparison | seed 0 | seed 1 | seed 2 | Better in |
|---|---|---|---|---|---|
| entropy | recency | +0.0152 [+0.0140, +0.0164] | +0.0234 [+0.0219, +0.0249] | +0.0283 [+0.0262, +0.0307] | 0/3 |
| surprisal | recency | +0.0117 [+0.0104, +0.0130] | +0.0184 [+0.0169, +0.0199] | +0.0205 [+0.0185, +0.0226] | 0/3 |
| keydiff | recency | +0.0085 [+0.0072, +0.0099] | +0.0164 [+0.0148, +0.0181] | +0.0139 [+0.0119, +0.0159] | 0/3 |
| keydiff | snap | +0.0126 [+0.0112, +0.0141] | +0.0212 [+0.0194, +0.0230] | +0.0177 [+0.0158, +0.0197] | 0/3 |
| expected | recency | -0.0030 [-0.0038, -0.0022] | -0.0020 [-0.0030, -0.0010] | -0.0019 [-0.0033, -0.0004] | 3/3 |
| expected | snap | +0.0010 [+0.0004, +0.0017] | +0.0028 [+0.0019, +0.0037] | +0.0020 [+0.0007, +0.0033] | 0/3 |
| snap | recency | -0.0041 [-0.0046, -0.0035] | -0.0048 [-0.0054, -0.0043] | -0.0038 [-0.0046, -0.0031] | 3/3 |
| snapv | recency | -0.0043 [-0.0049, -0.0037] | -0.0048 [-0.0054, -0.0043] | -0.0042 [-0.0049, -0.0034] | 3/3 |
| snapv | snap | -0.0002 [-0.0004, +0.0000] | -0.0000 [-0.0002, +0.0002] | -0.0003 [-0.0007, +0.0000] | 0/3 |

| Policy | BPB seed 0 | BPB seed 1 | BPB seed 2 |
|---|---|---|---|
| full|0 | 0.7980 | 0.8125 | 0.8453 |
| recency|32 | 0.8078 | 0.8247 | 0.8720 |
| entropy|32 | 0.8231 | 0.8481 | 0.9003 |
| surprisal|32 | 0.8195 | 0.8431 | 0.8925 |
| keydiff|32 | 0.8164 | 0.8411 | 0.8859 |
| expected|32 | 0.8048 | 0.8227 | 0.8702 |
| snap|32 | 0.8037 | 0.8199 | 0.8682 |
| snapv|32 | 0.8036 | 0.8199 | 0.8679 |

## wikitext103 (budget 64 patches)

| Variant | Comparison | seed 0 | seed 1 | seed 2 | Better in |
|---|---|---|---|---|---|
| entropy | recency | +0.0087 [+0.0077, +0.0097] | +0.0192 [+0.0173, +0.0209] | +0.0169 [+0.0148, +0.0193] | 0/3 |
| surprisal | recency | +0.0071 [+0.0060, +0.0082] | +0.0174 [+0.0156, +0.0191] | +0.0171 [+0.0148, +0.0194] | 0/3 |
| keydiff | recency | +0.0114 [+0.0095, +0.0136] | +0.0125 [+0.0110, +0.0138] | +0.0224 [+0.0197, +0.0255] | 0/3 |
| keydiff | snap | +0.0128 [+0.0109, +0.0151] | +0.0142 [+0.0127, +0.0156] | +0.0255 [+0.0223, +0.0290] | 0/3 |
| expected | recency | +0.0009 [+0.0005, +0.0013] | +0.0028 [+0.0020, +0.0036] | +0.0003 [-0.0004, +0.0012] | 0/3 |
| expected | snap | +0.0024 [+0.0019, +0.0028] | +0.0045 [+0.0040, +0.0049] | +0.0034 [+0.0029, +0.0039] | 0/3 |
| snap | recency | -0.0014 [-0.0019, -0.0010] | -0.0017 [-0.0023, -0.0011] | -0.0030 [-0.0036, -0.0025] | 3/3 |
| snapv | recency | -0.0013 [-0.0018, -0.0009] | -0.0017 [-0.0023, -0.0011] | -0.0032 [-0.0038, -0.0026] | 3/3 |
| snapv | snap | +0.0001 [+0.0000, +0.0002] | +0.0000 [-0.0001, +0.0001] | -0.0002 [-0.0003, -0.0001] | 1/3 |

| Policy | BPB seed 0 | BPB seed 1 | BPB seed 2 |
|---|---|---|---|
| full|0 | 1.6801 | 1.6994 | 1.7080 |
| recency|64 | 1.6836 | 1.7040 | 1.7126 |
| entropy|64 | 1.6922 | 1.7232 | 1.7294 |
| surprisal|64 | 1.6907 | 1.7214 | 1.7296 |
| keydiff|64 | 1.6950 | 1.7165 | 1.7349 |
| expected|64 | 1.6845 | 1.7068 | 1.7129 |
| snap|64 | 1.6822 | 1.7023 | 1.7096 |
| snapv|64 | 1.6823 | 1.7023 | 1.7094 |

