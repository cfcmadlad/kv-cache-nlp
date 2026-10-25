Recency-protected entropy eviction. The budget is split: the newest budget/2 patches are protected (a recency window), and the other half of the slots go to the patches with the highest mean entropy (`hybrid entropy`) or to random patches (`hybrid random`). Plain recency and plain high-entropy eviction (8 protected patches) are shown for reference. Validation only, 95% document bootstrap intervals (2000 resamples). Differences are the first minus the second (negative means the first is better).

## tinystories

| Model | Budget | recency | hybrid random | hybrid entropy | entropy | hybrid entropy - recency | hybrid entropy - hybrid random | hybrid random - recency |
|---|---|---|---|---|---|---|---|---|
| blt_tinystories_entropy | 32 | 0.8078 | 0.8066 | 0.8126 | 0.8231 | +0.0047 [+0.0040, +0.0054] | +0.0059 [+0.0054, +0.0065] | -0.0012 [-0.0018, -0.0007] |
| blt_tinystories_entropy | 64 | 0.8016 | 0.8003 | 0.8027 | 0.8109 | +0.0011 [+0.0007, +0.0016] | +0.0024 [+0.0020, +0.0027] | -0.0013 [-0.0017, -0.0009] |
| blt_tinystories_entropy_s1 | 32 | 0.8164 | 0.8156 | 0.8240 | 0.8415 | +0.0076 [+0.0068, +0.0085] | +0.0085 [+0.0077, +0.0092] | -0.0008 [-0.0015, -0.0002] |
| blt_tinystories_entropy_s1 | 64 | 0.8074 | 0.8061 | 0.8101 | 0.8237 | +0.0026 [+0.0021, +0.0032] | +0.0039 [+0.0035, +0.0044] | -0.0013 [-0.0017, -0.0009] |
| blt_tinystories_entropy_s2 | 32 | 0.8840 | 0.8827 | 0.8956 | 0.9131 | +0.0117 [+0.0102, +0.0131] | +0.0129 [+0.0117, +0.0142] | -0.0012 [-0.0023, -0.0002] |
| blt_tinystories_entropy_s2 | 64 | 0.8640 | 0.8634 | 0.8693 | 0.8839 | +0.0053 [+0.0043, +0.0065] | +0.0060 [+0.0052, +0.0068] | -0.0007 [-0.0013, +0.0001] |

Models (worse / tie / better, interval excluding zero) per comparison:

| Budget | hybrid entropy vs recency | hybrid entropy vs hybrid random | hybrid random vs recency |
|---|---|---|---|
| 32 | 3 / 0 / 0 | 3 / 0 / 0 | 0 / 0 / 3 |
| 64 | 3 / 0 / 0 | 3 / 0 / 0 | 0 / 1 / 2 |

## wikitext103

| Model | Budget | recency | hybrid random | hybrid entropy | entropy | hybrid entropy - recency | hybrid entropy - hybrid random | hybrid random - recency |
|---|---|---|---|---|---|---|---|---|
| blt_wikitext103_entropy | 64 | 1.6836 | 1.6835 | 1.6855 | 1.6922 | +0.0020 [+0.0016, +0.0023] | +0.0021 [+0.0018, +0.0024] | -0.0001 [-0.0003, +0.0001] |
| blt_wikitext103_entropy | 128 | 1.6814 | 1.6814 | 1.6826 | 1.6885 | +0.0012 [+0.0010, +0.0015] | +0.0012 [+0.0010, +0.0015] | -0.0000 [-0.0001, +0.0001] |
| blt_wikitext103_entropy_s1 | 64 | 1.7040 | 1.7044 | 1.7073 | 1.7232 | +0.0033 [+0.0026, +0.0040] | +0.0029 [+0.0024, +0.0035] | +0.0004 [+0.0001, +0.0007] |
| blt_wikitext103_entropy_s1 | 128 | 1.7010 | 1.7011 | 1.7030 | 1.7177 | +0.0020 [+0.0015, +0.0024] | +0.0019 [+0.0016, +0.0023] | +0.0001 [-0.0001, +0.0003] |
| blt_wikitext103_entropy_s2 | 64 | 1.7126 | 1.7131 | 1.7152 | 1.7294 | +0.0026 [+0.0020, +0.0033] | +0.0022 [+0.0016, +0.0027] | +0.0005 [+0.0002, +0.0007] |
| blt_wikitext103_entropy_s2 | 128 | 1.7095 | 1.7098 | 1.7110 | 1.7251 | +0.0015 [+0.0010, +0.0019] | +0.0012 [+0.0008, +0.0015] | +0.0003 [+0.0001, +0.0005] |

Models (worse / tie / better, interval excluding zero) per comparison:

| Budget | hybrid entropy vs recency | hybrid entropy vs hybrid random | hybrid random vs recency |
|---|---|---|---|
| 64 | 3 / 0 / 0 | 3 / 0 / 0 | 2 / 1 / 0 |
| 128 | 3 / 0 / 0 | 3 / 0 / 0 | 1 / 2 / 0 |

## tinystoriesmix

| Model | Budget | recency | hybrid random | hybrid entropy | entropy | hybrid entropy - recency | hybrid entropy - hybrid random | hybrid random - recency |
|---|---|---|---|---|---|---|---|---|
| blt_tinystoriesmix_entropy | 32 | 0.9091 | 0.9060 | 0.9110 | 0.9167 | +0.0019 [+0.0013, +0.0025] | +0.0050 [+0.0046, +0.0055] | -0.0032 [-0.0037, -0.0027] |
| blt_tinystoriesmix_entropy | 64 | 0.9044 | 0.9029 | 0.9050 | 0.9090 | +0.0007 [+0.0004, +0.0010] | +0.0021 [+0.0018, +0.0024] | -0.0015 [-0.0018, -0.0012] |
| blt_tinystoriesmix_entropy_s1 | 32 | 0.9005 | 0.8989 | 0.9036 | 0.9124 | +0.0031 [+0.0024, +0.0038] | +0.0046 [+0.0041, +0.0052] | -0.0015 [-0.0021, -0.0010] |
| blt_tinystoriesmix_entropy_s1 | 64 | 0.8954 | 0.8940 | 0.8965 | 0.9032 | +0.0011 [+0.0008, +0.0014] | +0.0025 [+0.0022, +0.0028] | -0.0014 [-0.0017, -0.0011] |
| blt_tinystoriesmix_entropy_s2 | 32 | 0.9489 | 0.9473 | 0.9519 | 0.9614 | +0.0031 [+0.0024, +0.0038] | +0.0047 [+0.0042, +0.0052] | -0.0016 [-0.0021, -0.0011] |
| blt_tinystoriesmix_entropy_s2 | 64 | 0.9430 | 0.9417 | 0.9441 | 0.9516 | +0.0011 [+0.0007, +0.0015] | +0.0023 [+0.0020, +0.0027] | -0.0012 [-0.0016, -0.0009] |

Models (worse / tie / better, interval excluding zero) per comparison:

| Budget | hybrid entropy vs recency | hybrid entropy vs hybrid random | hybrid random vs recency |
|---|---|---|---|
| 32 | 3 / 0 / 0 | 3 / 0 / 0 | 0 / 0 / 3 |
| 64 | 3 / 0 / 0 | 3 / 0 / 0 | 0 / 0 / 3 |

## wikitext103mix

| Model | Budget | recency | hybrid random | hybrid entropy | entropy | hybrid entropy - recency | hybrid entropy - hybrid random | hybrid random - recency |
|---|---|---|---|---|---|---|---|---|
| blt_wikitext103mix_entropy | 64 | 1.7211 | 1.7211 | 1.7218 | 1.7264 | +0.0007 [+0.0004, +0.0010] | +0.0006 [+0.0004, +0.0009] | +0.0001 [-0.0001, +0.0002] |
| blt_wikitext103mix_entropy | 128 | 1.7197 | 1.7198 | 1.7201 | 1.7240 | +0.0004 [+0.0003, +0.0006] | +0.0003 [+0.0001, +0.0005] | +0.0001 [+0.0000, +0.0002] |
| blt_wikitext103mix_entropy_s1 | 64 | 1.7139 | 1.7138 | 1.7143 | 1.7181 | +0.0005 [+0.0001, +0.0008] | +0.0006 [+0.0003, +0.0008] | -0.0001 [-0.0003, +0.0001] |
| blt_wikitext103mix_entropy_s1 | 128 | 1.7125 | 1.7126 | 1.7129 | 1.7163 | +0.0004 [+0.0002, +0.0006] | +0.0003 [+0.0001, +0.0005] | +0.0001 [-0.0000, +0.0002] |
| blt_wikitext103mix_entropy_s2 | 64 | 1.8162 | 1.8161 | 1.8165 | 1.8206 | +0.0002 [-0.0001, +0.0006] | +0.0004 [+0.0002, +0.0007] | -0.0002 [-0.0004, +0.0000] |
| blt_wikitext103mix_entropy_s2 | 128 | 1.8145 | 1.8148 | 1.8151 | 1.8188 | +0.0007 [+0.0004, +0.0009] | +0.0003 [+0.0001, +0.0005] | +0.0003 [+0.0002, +0.0005] |

Models (worse / tie / better, interval excluding zero) per comparison:

| Budget | hybrid entropy vs recency | hybrid entropy vs hybrid random | hybrid random vs recency |
|---|---|---|---|
| 64 | 2 / 1 / 0 | 3 / 0 / 0 | 0 / 3 / 0 |
| 128 | 3 / 0 / 0 | 3 / 0 / 0 | 2 / 1 / 0 |
