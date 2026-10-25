Bounded-memory streaming over long WikiText-103 documents (validation documents longer than 8,192 bytes, code-mix models trained on 4,096-byte chunks). `chunked 4k` resets the context every 4,096 bytes and keeps the full cache inside a chunk (up to about 900 patches). Streaming runs 16,384-byte chunks with a fixed patch budget, so patch positions reach about 3,600, four times beyond training; attention policies are scored online. 95% document bootstrap intervals (2000 resamples). Differences are the first minus the second (negative means the first is better).

## blt_wikitext103mix_entropy (46 documents)

Chunked 4k, full cache: 1.7210; chunked 16k stream, full cache: 1.7201.

| Budget | Policy | Streaming BPB | Cost over chunked 4k full | Policy - recency |
|---|---|---|---|---|
| 64 | recency | 1.7180 | -0.0030 |  |
| 64 | random | 1.7231 | +0.0021 | +0.0051 [+0.0046, +0.0058] |
| 64 | entropy | 1.7268 | +0.0057 | +0.0087 [+0.0077, +0.0101] |
| 64 | h2o | 1.7252 | +0.0042 | +0.0072 [+0.0065, +0.0080] |
| 64 | snap | 1.7166 | -0.0044 | -0.0014 [-0.0020, -0.0009] |
| 128 | recency | 1.7165 | -0.0045 |  |
| 128 | random | 1.7216 | +0.0006 | +0.0051 [+0.0046, +0.0057] |
| 128 | entropy | 1.7251 | +0.0040 | +0.0086 [+0.0078, +0.0097] |
| 128 | h2o | 1.7238 | +0.0028 | +0.0074 [+0.0067, +0.0082] |
| 128 | snap | 1.7160 | -0.0050 | -0.0005 [-0.0008, -0.0002] |
| 256 | recency | 1.7159 | -0.0051 |  |
| 256 | random | 1.7204 | -0.0006 | +0.0045 [+0.0041, +0.0050] |
| 256 | entropy | 1.7232 | +0.0022 | +0.0074 [+0.0067, +0.0080] |
| 256 | h2o | 1.7226 | +0.0015 | +0.0067 [+0.0061, +0.0073] |
| 256 | snap | 1.7157 | -0.0053 | -0.0002 [-0.0004, -0.0000] |

## blt_wikitext103mix_entropy_s1 (46 documents)

Chunked 4k, full cache: 1.7141; chunked 16k stream, full cache: 1.7178.

| Budget | Policy | Streaming BPB | Cost over chunked 4k full | Policy - recency |
|---|---|---|---|---|
| 64 | recency | 1.7109 | -0.0032 |  |
| 64 | random | 1.7153 | +0.0012 | +0.0044 [+0.0039, +0.0049] |
| 64 | entropy | 1.7183 | +0.0042 | +0.0074 [+0.0065, +0.0085] |
| 64 | h2o | 1.7177 | +0.0036 | +0.0068 [+0.0062, +0.0076] |
| 64 | snap | 1.7101 | -0.0040 | -0.0008 [-0.0011, -0.0004] |
| 128 | recency | 1.7092 | -0.0049 |  |
| 128 | random | 1.7141 | +0.0000 | +0.0049 [+0.0044, +0.0054] |
| 128 | entropy | 1.7173 | +0.0032 | +0.0081 [+0.0074, +0.0089] |
| 128 | h2o | 1.7169 | +0.0028 | +0.0077 [+0.0071, +0.0084] |
| 128 | snap | 1.7089 | -0.0052 | -0.0003 [-0.0006, -0.0000] |
| 256 | recency | 1.7083 | -0.0058 |  |
| 256 | random | 1.7136 | -0.0005 | +0.0053 [+0.0047, +0.0058] |
| 256 | entropy | 1.7165 | +0.0024 | +0.0082 [+0.0075, +0.0089] |
| 256 | h2o | 1.7170 | +0.0029 | +0.0087 [+0.0079, +0.0094] |
| 256 | snap | 1.7084 | -0.0057 | +0.0000 [-0.0002, +0.0002] |

## blt_wikitext103mix_entropy_s2 (46 documents)

Chunked 4k, full cache: 1.8162; chunked 16k stream, full cache: 1.8149.

| Budget | Policy | Streaming BPB | Cost over chunked 4k full | Policy - recency |
|---|---|---|---|---|
| 64 | recency | 1.8154 | -0.0008 |  |
| 64 | random | 1.8188 | +0.0026 | +0.0034 [+0.0027, +0.0043] |
| 64 | entropy | 1.8213 | +0.0050 | +0.0058 [+0.0048, +0.0070] |
| 64 | h2o | 1.8195 | +0.0033 | +0.0041 [+0.0032, +0.0052] |
| 64 | snap | 1.8130 | -0.0032 | -0.0024 [-0.0028, -0.0021] |
| 128 | recency | 1.8135 | -0.0027 |  |
| 128 | random | 1.8171 | +0.0009 | +0.0036 [+0.0030, +0.0043] |
| 128 | entropy | 1.8201 | +0.0039 | +0.0067 [+0.0057, +0.0077] |
| 128 | h2o | 1.8182 | +0.0020 | +0.0048 [+0.0039, +0.0058] |
| 128 | snap | 1.8127 | -0.0035 | -0.0008 [-0.0010, -0.0005] |
| 256 | recency | 1.8127 | -0.0035 |  |
| 256 | random | 1.8157 | -0.0005 | +0.0030 [+0.0026, +0.0034] |
| 256 | entropy | 1.8191 | +0.0029 | +0.0064 [+0.0056, +0.0072] |
| 256 | h2o | 1.8171 | +0.0009 | +0.0044 [+0.0037, +0.0053] |
| 256 | snap | 1.8126 | -0.0036 | -0.0001 [-0.0002, +0.0000] |
