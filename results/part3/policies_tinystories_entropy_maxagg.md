blt_tinystories_entropy on tinystories val (500 docs, 405,231 bytes), 2000 document bootstrap resamples, 95% intervals.

Full cache: BPB 0.7980 [0.7846, 0.8114], 1,669.3 KB mean global cache per chunk

| Policy | Budget (patches) | Mean cache KB | BPB [95% CI] | Cost over full |
|---|---|---|---|---|
| entropy | 12 | 106.7 | 0.8349 [0.8211, 0.8486] | +0.0369 |
| entropy_inv | 12 | 106.7 | 0.8273 [0.8132, 0.8413] | +0.0293 |
| entropy | 16 | 140.4 | 0.8276 [0.8138, 0.8414] | +0.0296 |
| entropy_inv | 16 | 140.4 | 0.8214 [0.8074, 0.8352] | +0.0234 |
| entropy | 32 | 265.9 | 0.8134 [0.7996, 0.8272] | +0.0154 |
| entropy_inv | 32 | 265.9 | 0.8127 [0.7988, 0.8266] | +0.0146 |
| entropy | 64 | 472.8 | 0.8039 [0.7900, 0.8179] | +0.0059 |
| entropy_inv | 64 | 472.8 | 0.8055 [0.7917, 0.8194] | +0.0075 |
| entropy | 128 | 802.3 | 0.7996 [0.7861, 0.8133] | +0.0016 |
| entropy_inv | 128 | 802.3 | 0.8006 [0.7868, 0.8143] | +0.0025 |

Paired BPB difference (first minus second, negative means the first is better), mean [95% CI]:

| Budget | entropy - entropy_inv |
|---|---|
| 12 | +0.0076 [+0.0064, +0.0087] * |
| 16 | +0.0062 [+0.0049, +0.0074] * |
| 32 | +0.0007 [-0.0004, +0.0018] |
| 64 | -0.0016 [-0.0025, -0.0008] * |
| 128 | -0.0009 [-0.0015, -0.0004] * |

* the interval excludes zero.