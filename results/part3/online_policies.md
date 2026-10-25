Online H2O and SnapKV-style eviction against the offline approximation used in `attention_policies.md`. Online scores are accumulated during step-by-step patch decoding: the entry to drop is chosen before the new patch is attended to, from attention rows already seen (h2o: all rows; snap: the last 16). The offline scores come from a full-cache pass refined once under the resulting mask. `recency (online)` runs the same code path as the attention policies and must equal the masked recency run. Validation only, 95% document bootstrap intervals (2000 resamples). Differences are the first minus the second (negative means the first is better).

## tinystories

| Model | Budget | recency | recency (online) | h2o offline | h2o online | snap offline | snap online | h2o on - recency | snap on - recency | snap on - snap off |
|---|---|---|---|---|---|---|---|---|---|---|
| blt_tinystories_entropy | 32 | 0.7986 | 0.7986 | 0.8014 | 0.8023 | 0.7942 | 0.7938 | +0.0037 [+0.0024, +0.0051] | -0.0048 [-0.0056, -0.0041] | -0.0004 [-0.0008, -0.0001] |
| blt_tinystories_entropy | 64 | 0.7923 | 0.7923 | 0.7946 | 0.7949 | 0.7900 | 0.7900 | +0.0025 [+0.0014, +0.0037] | -0.0024 [-0.0030, -0.0018] | -0.0001 [-0.0002, +0.0001] |
| blt_tinystories_entropy_s1 | 32 | 0.8079 | 0.8079 | 0.8205 | 0.8213 | 0.8031 | 0.8027 | +0.0134 [+0.0114, +0.0154] | -0.0053 [-0.0061, -0.0044] | -0.0005 [-0.0009, -0.0001] |
| blt_tinystories_entropy_s1 | 64 | 0.7985 | 0.7985 | 0.8057 | 0.8066 | 0.7961 | 0.7961 | +0.0080 [+0.0059, +0.0101] | -0.0024 [-0.0030, -0.0018] | +0.0000 [-0.0002, +0.0002] |
| blt_tinystories_entropy_s2 | 32 | 0.8750 | 0.8750 | 0.8837 | 0.8842 | 0.8681 | 0.8673 | +0.0092 [+0.0066, +0.0119] | -0.0077 [-0.0091, -0.0064] | -0.0009 [-0.0015, -0.0003] |
| blt_tinystories_entropy_s2 | 64 | 0.8549 | 0.8549 | 0.8626 | 0.8631 | 0.8524 | 0.8522 | +0.0082 [+0.0057, +0.0107] | -0.0027 [-0.0038, -0.0017] | -0.0002 [-0.0005, +0.0001] |

## wikitext103

| Model | Budget | recency | recency (online) | h2o offline | h2o online | snap offline | snap online | h2o on - recency | snap on - recency | snap on - snap off |
|---|---|---|---|---|---|---|---|---|---|---|
| blt_wikitext103_entropy | 64 | 1.6836 | 1.6836 | 1.6881 | 1.6882 | 1.6822 | 1.6821 | +0.0046 [+0.0040, +0.0053] | -0.0015 [-0.0020, -0.0010] | -0.0000 [-0.0002, +0.0001] |
| blt_wikitext103_entropy | 128 | 1.6814 | 1.6814 | 1.6849 | 1.6850 | 1.6811 | 1.6810 | +0.0037 [+0.0031, +0.0043] | -0.0004 [-0.0007, -0.0002] | -0.0001 [-0.0002, +0.0000] |
| blt_wikitext103_entropy_s1 | 64 | 1.7040 | 1.7040 | 1.7136 | 1.7139 | 1.7023 | 1.7022 | +0.0099 [+0.0089, +0.0109] | -0.0018 [-0.0024, -0.0013] | -0.0001 [-0.0002, -0.0000] |
| blt_wikitext103_entropy_s1 | 128 | 1.7010 | 1.7010 | 1.7080 | 1.7084 | 1.7004 | 1.7004 | +0.0074 [+0.0066, +0.0082] | -0.0006 [-0.0008, -0.0003] | +0.0000 [-0.0001, +0.0001] |
| blt_wikitext103_entropy_s2 | 64 | 1.7126 | 1.7126 | 1.7199 | 1.7202 | 1.7096 | 1.7095 | +0.0077 [+0.0069, +0.0085] | -0.0031 [-0.0037, -0.0026] | -0.0001 [-0.0002, -0.0000] |
| blt_wikitext103_entropy_s2 | 128 | 1.7095 | 1.7095 | 1.7153 | 1.7157 | 1.7084 | 1.7083 | +0.0062 [+0.0055, +0.0069] | -0.0012 [-0.0015, -0.0009] | -0.0000 [-0.0001, +0.0000] |

## tinystoriesmix

| Model | Budget | recency | recency (online) | h2o offline | h2o online | snap offline | snap online | h2o on - recency | snap on - recency | snap on - snap off |
|---|---|---|---|---|---|---|---|---|---|---|
| blt_tinystoriesmix_entropy | 32 | 0.8996 | 0.8996 | 0.8976 | 0.8977 | 0.8958 | 0.8957 | -0.0018 [-0.0029, -0.0009] | -0.0039 [-0.0046, -0.0032] | -0.0002 [-0.0005, +0.0002] |
| blt_tinystoriesmix_entropy | 64 | 0.8951 | 0.8951 | 0.8948 | 0.8948 | 0.8935 | 0.8934 | -0.0003 [-0.0009, +0.0004] | -0.0017 [-0.0021, -0.0013] | -0.0001 [-0.0002, +0.0000] |
| blt_tinystoriesmix_entropy_s1 | 32 | 0.8918 | 0.8918 | 0.8954 | 0.8960 | 0.8885 | 0.8882 | +0.0041 [+0.0026, +0.0057] | -0.0036 [-0.0042, -0.0031] | -0.0003 [-0.0005, +0.0000] |
| blt_tinystoriesmix_entropy_s1 | 64 | 0.8867 | 0.8867 | 0.8895 | 0.8899 | 0.8846 | 0.8845 | +0.0032 [+0.0019, +0.0045] | -0.0022 [-0.0026, -0.0018] | -0.0001 [-0.0002, +0.0000] |
| blt_tinystoriesmix_entropy_s2 | 32 | 0.9394 | 0.9394 | 0.9411 | 0.9420 | 0.9345 | 0.9337 | +0.0026 [+0.0013, +0.0039] | -0.0057 [-0.0065, -0.0050] | -0.0008 [-0.0012, -0.0004] |
| blt_tinystoriesmix_entropy_s2 | 64 | 0.9330 | 0.9330 | 0.9344 | 0.9348 | 0.9306 | 0.9305 | +0.0018 [+0.0008, +0.0028] | -0.0025 [-0.0030, -0.0020] | -0.0001 [-0.0002, +0.0000] |

## wikitext103mix

| Model | Budget | recency | recency (online) | h2o offline | h2o online | snap offline | snap online | h2o on - recency | snap on - recency | snap on - snap off |
|---|---|---|---|---|---|---|---|---|---|---|
| blt_wikitext103mix_entropy | 64 | 1.7211 | 1.7211 | 1.7255 | 1.7259 | 1.7199 | 1.7198 | +0.0048 [+0.0043, +0.0054] | -0.0013 [-0.0018, -0.0009] | -0.0001 [-0.0002, -0.0000] |
| blt_wikitext103mix_entropy | 128 | 1.7197 | 1.7197 | 1.7232 | 1.7235 | 1.7194 | 1.7193 | +0.0039 [+0.0034, +0.0044] | -0.0004 [-0.0006, -0.0001] | -0.0000 [-0.0001, +0.0000] |
| blt_wikitext103mix_entropy_s1 | 64 | 1.7139 | 1.7139 | 1.7170 | 1.7171 | 1.7133 | 1.7132 | +0.0032 [+0.0027, +0.0037] | -0.0007 [-0.0010, -0.0004] | -0.0001 [-0.0002, +0.0000] |
| blt_wikitext103mix_entropy_s1 | 128 | 1.7125 | 1.7125 | 1.7153 | 1.7153 | 1.7123 | 1.7122 | +0.0029 [+0.0025, +0.0032] | -0.0002 [-0.0005, -0.0000] | -0.0001 [-0.0002, +0.0000] |
| blt_wikitext103mix_entropy_s2 | 64 | 1.8162 | 1.8162 | 1.8180 | 1.8181 | 1.8142 | 1.8142 | +0.0019 [+0.0013, +0.0026] | -0.0020 [-0.0024, -0.0017] | -0.0000 [-0.0001, +0.0001] |
| blt_wikitext103mix_entropy_s2 | 128 | 1.8145 | 1.8145 | 1.8164 | 1.8166 | 1.8139 | 1.8140 | +0.0021 [+0.0015, +0.0027] | -0.0005 [-0.0007, -0.0003] | +0.0000 [-0.0000, +0.0001] |

## Tally over all models and budgets

| Comparison | first better (interval below zero) | tie | first worse |
|---|---|---|---|
| h2o online vs recency | 1 | 1 | 22 |
| snap online vs recency | 24 | 0 | 0 |
| snap online vs snap offline | 7 | 17 | 0 |
| recency online vs recency | 0 | 22 | 2 |