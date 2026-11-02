Merging adjacent cache entries instead of evicting them, at equal numbers of entries (training-free). When the cache is full, the two adjacent old entries are replaced by their byte-count-weighted average (keys and values, every layer): `weak` merges the pair whose shared boundary had the smallest entropy margin above the patching threshold (the weakest boundary is undone), `random` a random adjacent pair, `fixed` the oldest adjacent pair. `evict` drops the oldest entry (recency). The first patch and the 8 newest are never touched. Validation only; paired 95% document bootstrap intervals (2000 resamples); negative means the merge rule is better than the comparison.

## tinystories (500 documents, budget 32 entries)

| Rule | Comparison | seed 0 | seed 1 | seed 2 | Better in |
|---|---|---|---|---|---|
| weak | evict | +0.0038 [+0.0029, +0.0046] | +0.0082 [+0.0071, +0.0093] | +0.0089 [+0.0073, +0.0105] | 0/3 |
| weak | random | +0.0043 [+0.0035, +0.0050] | +0.0082 [+0.0071, +0.0092] | +0.0076 [+0.0062, +0.0090] | 0/3 |
| random | evict | -0.0005 [-0.0009, -0.0000] | +0.0000 [-0.0005, +0.0005] | +0.0014 [+0.0005, +0.0023] | 1/3 |
| fixed | evict | -0.0001 [-0.0002, +0.0000] | -0.0001 [-0.0002, +0.0001] | +0.0005 [+0.0003, +0.0008] | 0/3 |
| fixed | random | +0.0004 [-0.0001, +0.0008] | -0.0001 [-0.0006, +0.0004] | -0.0008 [-0.0017, +0.0000] | 0/3 |

| Rule | BPB seed 0 | BPB seed 1 | BPB seed 2 |
|---|---|---|---|
| full|0 | 0.7980 | 0.8026 | 0.8544 |
| evict|32 | 0.8078 | 0.8164 | 0.8840 |
| weak|32 | 0.8116 | 0.8246 | 0.8929 |
| random|32 | 0.8074 | 0.8164 | 0.8853 |
| fixed|32 | 0.8077 | 0.8163 | 0.8845 |

## wikitext103 (60 documents, budget 64 entries)

| Rule | Comparison | seed 0 | seed 1 | seed 2 | Better in |
|---|---|---|---|---|---|
| weak | evict | +0.0060 [+0.0050, +0.0072] | +0.0110 [+0.0100, +0.0123] | +0.0109 [+0.0096, +0.0124] | 0/3 |
| weak | random | +0.0051 [+0.0044, +0.0061] | +0.0095 [+0.0086, +0.0104] | +0.0088 [+0.0077, +0.0100] | 0/3 |
| random | evict | +0.0008 [+0.0005, +0.0012] | +0.0016 [+0.0012, +0.0020] | +0.0021 [+0.0018, +0.0025] | 0/3 |
| fixed | evict | +0.0000 [+0.0000, +0.0001] | +0.0001 [+0.0000, +0.0001] | +0.0001 [+0.0001, +0.0001] | 0/3 |
| fixed | random | -0.0008 [-0.0011, -0.0005] | -0.0015 [-0.0019, -0.0012] | -0.0020 [-0.0024, -0.0017] | 3/3 |

| Rule | BPB seed 0 | BPB seed 1 | BPB seed 2 |
|---|---|---|---|
| full|0 | 1.6801 | 1.6994 | 1.7080 |
| evict|64 | 1.6836 | 1.7040 | 1.7126 |
| weak|64 | 1.6896 | 1.7150 | 1.7234 |
| random|64 | 1.6844 | 1.7056 | 1.7147 |
| fixed|64 | 1.6836 | 1.7041 | 1.7127 |

