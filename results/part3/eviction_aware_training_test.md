Eviction-aware training: models trained with a policy applied to the cache (regimes) and evaluated under every policy, test split, evaluated once, mean over the available seeds. 'none' is the standard model trained with a full cache. Paired differences use 2000 document bootstrap resamples, 95% intervals, per seed.

## wikitext103 (budget 64 patches)

| Training regime | Seeds | full | recency | random | entropy | snap | entropy - recency | entropy - random |
|---|---|---|---|---|---|---|---|---|
| none | 3 | 1.7210 | 1.7263 | 1.7316 | 1.7421 | 1.7233 | +0.0158 (worse 3, better 0 of 3) | +0.0104 (worse 3, better 0 of 3) |
| recency | 3 | 1.7047 | 1.7059 | 1.7109 | 1.7145 | 1.7048 | +0.0086 (worse 3, better 0 of 3) | +0.0036 (worse 3, better 0 of 3) |
| random | 3 | 1.7265 | 1.7282 | 1.7288 | 1.7315 | 1.7266 | +0.0033 (worse 3, better 0 of 3) | +0.0027 (worse 3, better 0 of 3) |
| entropy | 3 | 1.7093 | 1.7112 | 1.7119 | 1.7118 | 1.7099 | +0.0006 (worse 2, better 0 of 3) | -0.0001 (worse 0, better 1 of 3) |
| mixed | 3 | 1.7114 | 1.7127 | 1.7145 | 1.7160 | 1.7115 | +0.0033 (worse 3, better 0 of 3) | +0.0016 (worse 3, better 0 of 3) |

