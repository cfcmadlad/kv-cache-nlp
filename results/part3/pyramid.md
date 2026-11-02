Per-layer cache budgets (PyramidKV-style). Every schedule has the same mean budget across the six global layers; the pyramid gives lower layers more patches, the inverse pyramid the opposite. Mean entries is the realized mean number of cached patch entries summed over layers per query. Validation only; paired differences use 2000 document bootstrap resamples, 95% intervals, per seed (negative favours the first entry).

## tinystories (mean budget 32 patches)

Schedules (layer 0 to 5): snap_uniform: [32, 32, 32, 32, 32, 32]; snap_pyramid: [48, 42, 35, 29, 22, 16]; snap_inverse: [16, 22, 29, 35, 42, 48]; recency_uniform: [32, 32, 32, 32, 32, 32]; recency_pyramid: [48, 42, 35, 29, 22, 16]; recency_inverse: [16, 22, 29, 35, 42, 48]

| Schedule | mean BPB | mean entries |
|---|---|---|
| snap_uniform | 0.8281 | 28.9 |
| snap_pyramid | 0.8273 | 28.5 |
| snap_inverse | 0.8336 | 28.5 |
| recency_uniform | 0.8361 | 28.9 |
| recency_pyramid | 0.8332 | 28.5 |
| recency_inverse | 0.8421 | 28.5 |
| full | 0.8183 | |

| Comparison (a - b) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| snap_pyramid - snap_uniform | -0.0009 [-0.0012, -0.0006] | -0.0008 [-0.0011, -0.0005] | -0.0007 [-0.0012, -0.0002] |
| snap_inverse - snap_uniform | +0.0045 [+0.0040, +0.0049] | +0.0052 [+0.0047, +0.0057] | +0.0067 [+0.0060, +0.0073] |
| snap_pyramid - snap_inverse | -0.0054 [-0.0059, -0.0049] | -0.0060 [-0.0065, -0.0055] | -0.0073 [-0.0081, -0.0065] |
| recency_pyramid - recency_uniform | -0.0024 [-0.0027, -0.0021] | -0.0020 [-0.0024, -0.0016] | -0.0044 [-0.0049, -0.0038] |
| recency_inverse - recency_uniform | +0.0062 [+0.0057, +0.0067] | +0.0054 [+0.0049, +0.0060] | +0.0064 [+0.0058, +0.0071] |
| snap_pyramid - recency_uniform | -0.0062 [-0.0068, -0.0056] | -0.0081 [-0.0088, -0.0075] | -0.0119 [-0.0131, -0.0108] |
| snap_uniform - recency_uniform | -0.0053 [-0.0059, -0.0047] | -0.0073 [-0.0079, -0.0067] | -0.0112 [-0.0124, -0.0100] |

## wikitext103 (mean budget 64 patches)

Schedules (layer 0 to 5): snap_uniform: [64, 64, 64, 64, 64, 64]; snap_pyramid: [96, 83, 70, 58, 45, 32]; snap_inverse: [32, 45, 58, 70, 83, 96]; recency_uniform: [64, 64, 64, 64, 64, 64]; recency_pyramid: [96, 83, 70, 58, 45, 32]; recency_inverse: [32, 45, 58, 70, 83, 96]

| Schedule | mean BPB | mean entries |
|---|---|---|
| snap_uniform | 1.6977 | 60.6 |
| snap_pyramid | 1.6972 | 60.3 |
| snap_inverse | 1.6993 | 60.3 |
| recency_uniform | 1.7001 | 60.6 |
| recency_pyramid | 1.6993 | 60.3 |
| recency_inverse | 1.7021 | 60.3 |
| full | 1.6958 | |

| Comparison (a - b) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| snap_pyramid - snap_uniform | -0.0004 [-0.0005, -0.0003] | -0.0007 [-0.0008, -0.0006] | -0.0005 [-0.0006, -0.0004] |
| snap_inverse - snap_uniform | +0.0011 [+0.0009, +0.0012] | +0.0020 [+0.0019, +0.0022] | +0.0016 [+0.0014, +0.0018] |
| snap_pyramid - snap_inverse | -0.0015 [-0.0017, -0.0012] | -0.0027 [-0.0029, -0.0025] | -0.0021 [-0.0023, -0.0019] |
| recency_pyramid - recency_uniform | -0.0003 [-0.0004, -0.0002] | -0.0008 [-0.0010, -0.0006] | -0.0013 [-0.0014, -0.0011] |
| recency_inverse - recency_uniform | +0.0011 [+0.0008, +0.0013] | +0.0024 [+0.0022, +0.0027] | +0.0025 [+0.0023, +0.0028] |
| snap_pyramid - recency_uniform | -0.0020 [-0.0025, -0.0017] | -0.0027 [-0.0032, -0.0021] | -0.0039 [-0.0044, -0.0034] |
| snap_uniform - recency_uniform | -0.0016 [-0.0021, -0.0012] | -0.0020 [-0.0025, -0.0014] | -0.0034 [-0.0039, -0.0029] |

