Eviction-aware training: models trained with a policy applied to the cache (regimes) and evaluated under every policy, validation only, mean over the available seeds. 'none' is the standard model trained with a full cache. Paired differences use 2000 document bootstrap resamples, 95% intervals, per seed.

## tinystories (budget 32 patches)

| Training regime | Seeds | full | recency | random | entropy | snap | entropy - recency | entropy - random |
|---|---|---|---|---|---|---|---|---|
| none | 3 | 0.8183 | 0.8361 | 0.8417 | 0.8592 | n/a | +0.0232 (worse 3, better 0 of 3) | +0.0175 (worse 3, better 0 of 3) |
| recency | 3 | 0.8163 | 0.8208 | 0.8286 | 0.8334 | 0.8201 | +0.0126 (worse 3, better 0 of 3) | +0.0048 (worse 3, better 0 of 3) |
| random | 3 | 0.8184 | 0.8243 | 0.8242 | 0.8277 | 0.8223 | +0.0034 (worse 3, better 0 of 3) | +0.0035 (worse 3, better 0 of 3) |
| entropy | 3 | 0.8210 | 0.8271 | 0.8271 | 0.8257 | 0.8253 | -0.0013 (worse 0, better 1 of 3) | -0.0014 (worse 0, better 3 of 3) |
| mixed | 3 | 0.8184 | 0.8220 | 0.8240 | 0.8246 | 0.8211 | +0.0025 (worse 3, better 0 of 3) | +0.0006 (worse 2, better 0 of 3) |

## wikitext103 (budget 64 patches)

| Training regime | Seeds | full | recency | random | entropy | snap | entropy - recency | entropy - random |
|---|---|---|---|---|---|---|---|---|
| none | 3 | 1.6958 | 1.7001 | 1.7051 | 1.7149 | n/a | +0.0149 (worse 3, better 0 of 3) | +0.0099 (worse 3, better 0 of 3) |
| recency | 3 | 1.6839 | 1.6847 | 1.6895 | 1.6931 | 1.6839 | +0.0084 (worse 3, better 0 of 3) | +0.0036 (worse 3, better 0 of 3) |
| random | 3 | 1.7001 | 1.7012 | 1.7019 | 1.7044 | 1.7001 | +0.0032 (worse 3, better 0 of 3) | +0.0025 (worse 3, better 0 of 3) |
| entropy | 3 | 1.6863 | 1.6876 | 1.6884 | 1.6882 | 1.6865 | +0.0006 (worse 2, better 0 of 3) | -0.0001 (worse 0, better 1 of 3) |
| mixed | 3 | 1.6863 | 1.6872 | 1.6891 | 1.6903 | 1.6864 | +0.0030 (worse 3, better 0 of 3) | +0.0012 (worse 3, better 0 of 3) |

