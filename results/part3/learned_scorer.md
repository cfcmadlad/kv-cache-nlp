Learned, attention-free eviction scorer. A small network predicts, from cheap features, the attention a cached patch will receive from the next 16 patches (the quantity the SnapKV-style rule measures). Features: age, patch length, entropy statistics (dropped in scorer_noent), key and value norms. Trained per model on training documents, evaluated on validation documents. Negative differences mean the first policy is better. Paired 95% document bootstrap intervals (2000 resamples), one column per seed.

## tinystories (budget 32 patches)

| Comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| scorer_all - recency | -0.0027 [-0.0031, -0.0024] | -0.0031 [-0.0036, -0.0027] | -0.0044 [-0.0053, -0.0036] |
| scorer_all - snap | +0.0014 [+0.0009, +0.0018] | +0.0017 [+0.0013, +0.0020] | +0.0021 [+0.0013, +0.0029] |
| scorer_all - scorer_noent | -0.0001 [-0.0004, +0.0001] | -0.0000 [-0.0002, +0.0001] | +0.0000 [-0.0002, +0.0003] |
| scorer_noent - recency | -0.0026 [-0.0030, -0.0022] | -0.0031 [-0.0035, -0.0027] | -0.0045 [-0.0053, -0.0036] |
| snap - recency | -0.0041 [-0.0046, -0.0035] | -0.0048 [-0.0054, -0.0043] | -0.0065 [-0.0075, -0.0055] |

| Policy | BPB seed 0 | BPB seed 1 | BPB seed 2 |
|---|---|---|---|
| full|0 | 0.7980 | 0.8026 | 0.8544 |
| recency|32 | 0.8078 | 0.8164 | 0.8840 |
| entropy|32 | 0.8231 | 0.8415 | 0.9131 |
| snap|32 | 0.8037 | 0.8116 | 0.8774 |
| scorer_all|32 | 0.8051 | 0.8133 | 0.8795 |
| scorer_noent|32 | 0.8052 | 0.8133 | 0.8795 |

## wikitext103 (budget 64 patches)

| Comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| scorer_all - recency | -0.0009 [-0.0013, -0.0005] | -0.0006 [-0.0009, -0.0003] | -0.0011 [-0.0014, -0.0009] |
| scorer_all - snap | +0.0005 [+0.0002, +0.0008] | +0.0011 [+0.0008, +0.0015] | +0.0019 [+0.0015, +0.0023] |
| scorer_all - scorer_noent | +0.0001 [+0.0000, +0.0002] | +0.0001 [-0.0000, +0.0002] | -0.0001 [-0.0002, +0.0000] |
| scorer_noent - recency | -0.0010 [-0.0014, -0.0007] | -0.0006 [-0.0009, -0.0004] | -0.0010 [-0.0013, -0.0008] |
| snap - recency | -0.0014 [-0.0019, -0.0010] | -0.0017 [-0.0023, -0.0011] | -0.0030 [-0.0036, -0.0025] |

| Policy | BPB seed 0 | BPB seed 1 | BPB seed 2 |
|---|---|---|---|
| full|0 | 1.6801 | 1.6994 | 1.7080 |
| recency|64 | 1.6836 | 1.7040 | 1.7126 |
| entropy|64 | 1.6922 | 1.7232 | 1.7294 |
| snap|64 | 1.6822 | 1.7023 | 1.7096 |
| scorer_all|64 | 1.6827 | 1.7034 | 1.7115 |
| scorer_noent|64 | 1.6826 | 1.7034 | 1.7115 |

