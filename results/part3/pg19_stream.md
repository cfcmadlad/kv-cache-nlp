Long books (PG-19 test split, `emozilla/pg19` mirror; the first 100,000 bytes of each of the first 20 books) decoded one byte at a time with a real fixed-size cache; patch positions grow to about 22,000, two orders of magnitude beyond the 4,096-byte training chunks. `reset4096` resets the context every 4,096 bytes with a full cache inside each chunk; `full` keeps every patch for the whole book. Models are WikiText code-mix models, so PG-19 (pre-1919 English) is out of domain. 95% book bootstrap intervals (2000 resamples). Differences are the first minus the second (negative means the first is better).

## blt_wikitext103mix_entropy_x10 (20 books, 1,865,760 bytes)

reset4096 full cache within chunks: 2.2097; whole-book full cache: 2.3467 [2.2641, 2.4280].

| Budget | Policy | BPB | Policy - reset | Policy - recency |
|---|---|---|---|---|
| 64 | recency | 2.2265 | +0.0168 [+0.0149, +0.0184] |  |
| 128 | recency | 2.2107 | +0.0009 [+0.0000, +0.0017] |  |
| 256 | recency | 2.2012 | -0.0086 [-0.0093, -0.0079] |  |
| 64 | random | 2.3069 | +0.0972 [+0.0936, +0.1012] | +0.0805 [+0.0771, +0.0840] |
| 128 | random | 2.2870 | +0.0773 [+0.0737, +0.0807] | +0.0763 [+0.0731, +0.0797] |
| 256 | random | 2.2692 | +0.0595 [+0.0556, +0.0630] | +0.0681 [+0.0644, +0.0715] |
| 64 | entropy | 2.3193 | +0.1095 [+0.1030, +0.1153] | +0.0927 [+0.0873, +0.0979] |
| 128 | entropy | 2.3160 | +0.1062 [+0.0995, +0.1123] | +0.1052 [+0.0991, +0.1112] |
| 256 | entropy | 2.3227 | +0.1129 [+0.1047, +0.1210] | +0.1215 [+0.1135, +0.1292] |

## blt_wikitext103mix_entropy (20 books, 1,865,760 bytes)

reset4096 full cache within chunks: 2.5566; whole-book full cache: 2.5763 [2.4751, 2.6774].

| Budget | Policy | BPB | Policy - reset | Policy - recency |
|---|---|---|---|---|
| 64 | recency | 2.5538 | -0.0028 [-0.0038, -0.0017] |  |
| 128 | recency | 2.5522 | -0.0044 [-0.0051, -0.0037] |  |
| 256 | recency | 2.5512 | -0.0054 [-0.0062, -0.0047] |  |
| 64 | random | 2.5578 | +0.0011 [-0.0011, +0.0034] | +0.0039 [+0.0020, +0.0058] |
| 128 | random | 2.5566 | -0.0000 [-0.0020, +0.0021] | +0.0044 [+0.0025, +0.0063] |
| 256 | random | 2.5567 | +0.0001 [-0.0016, +0.0018] | +0.0055 [+0.0038, +0.0072] |
| 64 | entropy | 2.5789 | +0.0223 [+0.0175, +0.0273] | +0.0251 [+0.0209, +0.0293] |
| 128 | entropy | 2.5788 | +0.0222 [+0.0175, +0.0272] | +0.0266 [+0.0223, +0.0311] |
| 256 | entropy | 2.5795 | +0.0229 [+0.0179, +0.0281] | +0.0283 [+0.0238, +0.0335] |
