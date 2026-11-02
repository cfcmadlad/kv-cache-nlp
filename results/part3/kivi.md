KIVI-style asymmetric quantization (per-channel keys in groups of 16 patches, per-token values) against symmetric per-token quantization (sym), per-token keys and values (tok), and variants keeping the 8 most recent patches in fp16 (_r8). Cost is the mean cache size in bytes per document, including scales. Eviction rows use the SnapKV-style rule at the stated budget. Validation only, mean over three seeds; paired differences use 2000 document bootstrap resamples, 95% intervals, per seed (negative favours the first entry).

## tinystories (budget 32 patches)

| Variant | bits | evict | mean BPB | mean bytes | vs full BPB |
|---|---|---|---|---|---|
| full | 16 | none | 0.8183 | 839,246 | +0.0000 |
| sym | 4 | full | 0.8195 | 222,925 | +0.0012 |
| sym | 3 | full | 0.8248 | 170,472 | +0.0064 |
| sym | 2 | full | 0.9151 | 118,019 | +0.0968 |
| sym_r8 | 4 | full | 0.8186 | 275,865 | +0.0002 |
| sym_r8 | 3 | full | 0.8195 | 227,918 | +0.0012 |
| sym_r8 | 2 | full | 0.8349 | 179,971 | +0.0165 |
| tok | 4 | full | 0.8191 | 236,038 | +0.0008 |
| tok | 3 | full | 0.8222 | 183,585 | +0.0038 |
| tok | 2 | full | 0.8409 | 131,132 | +0.0225 |
| kivi | 4 | full | 0.8184 | 321,285 | +0.0001 |
| kivi | 3 | full | 0.8188 | 273,102 | +0.0005 |
| kivi | 2 | full | 0.8212 | 224,920 | +0.0028 |
| kivi_r8 | 4 | full | 0.8184 | 334,340 | +0.0000 |
| kivi_r8 | 3 | full | 0.8186 | 287,372 | +0.0002 |
| kivi_r8 | 2 | full | 0.8199 | 240,404 | +0.0015 |
| sym | 4 | snap | 0.8321 | 70,625 | +0.0138 |
| sym | 3 | snap | 0.8373 | 54,007 | +0.0190 |
| sym | 2 | snap | 0.9200 | 37,390 | +0.1017 |
| kivi_r8 | 4 | snap | 0.8310 | 145,911 | +0.0126 |
| kivi_r8 | 3 | snap | 0.8312 | 134,751 | +0.0128 |
| kivi_r8 | 2 | snap | 0.8325 | 123,591 | +0.0142 |

| Comparison (a - b) | bytes a / bytes b | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| kivi|2|full|0 - sym|3|full|0 | 1.319 | -0.0027 [-0.0032, -0.0022] | -0.0039 [-0.0046, -0.0032] | -0.0042 [-0.0049, -0.0035] |
| kivi|2|full|0 - sym|2|full|0 | 1.906 | -0.0697 [-0.0720, -0.0675] | -0.1044 [-0.1069, -0.1020] | -0.1076 [-0.1105, -0.1050] |
| kivi_r8|2|full|0 - sym_r8|2|full|0 | 1.336 | -0.0104 [-0.0117, -0.0093] | -0.0150 [-0.0163, -0.0137] | -0.0195 [-0.0216, -0.0177] |
| kivi_r8|3|full|0 - sym|3|full|0 | 1.686 | -0.0046 [-0.0051, -0.0041] | -0.0065 [-0.0071, -0.0059] | -0.0074 [-0.0080, -0.0067] |
| kivi|2|full|0 - tok|2|full|0 | 1.715 | -0.0150 [-0.0159, -0.0142] | -0.0213 [-0.0224, -0.0202] | -0.0228 [-0.0238, -0.0218] |
| tok|3|full|0 - sym|3|full|0 | 1.077 | -0.0017 [-0.0023, -0.0011] | -0.0028 [-0.0035, -0.0021] | -0.0033 [-0.0041, -0.0026] |
| kivi_r8|3|snap|32 - sym|3|snap|32 | 2.495 | -0.0047 [-0.0051, -0.0042] | -0.0068 [-0.0073, -0.0062] | -0.0070 [-0.0076, -0.0065] |
| kivi_r8|2|snap|32 - sym|3|snap|32 | 2.288 | -0.0038 [-0.0043, -0.0032] | -0.0054 [-0.0060, -0.0048] | -0.0053 [-0.0059, -0.0047] |

Byte-matched: each variant against the piecewise-linear interpolation of the sym (or sym_r8) 2, 3 and 4-bit curve at the same mean bytes, means over seeds. Negative favours the variant.

| Variant | bits | evict | mean bytes | variant BPB | baseline BPB at same bytes | difference |
|---|---|---|---|---|---|---|
| tok | 2 | full | 131,132 | 0.8409 | 0.8925 (sym) | -0.0516 |
| tok | 3 | full | 183,585 | 0.8222 | 0.8234 (sym) | -0.0013 |
| tok | 4 | full | 236,038 | 0.8191 | outside sym range | |
| kivi | 2 | full | 224,920 | 0.8212 | outside sym range | |
| kivi | 3 | full | 273,102 | 0.8188 | outside sym range | |
| kivi | 4 | full | 321,285 | 0.8184 | outside sym range | |
| kivi_r8 | 2 | full | 240,404 | 0.8199 | 0.8193 (sym_r8) | +0.0006 |
| kivi_r8 | 3 | full | 287,372 | 0.8186 | outside sym_r8 range | |
| kivi_r8 | 4 | full | 334,340 | 0.8184 | outside sym_r8 range | |
| kivi_r8 | 2 | snap | 123,591 | 0.8325 | outside sym range | |
| kivi_r8 | 3 | snap | 134,751 | 0.8312 | outside sym range | |
| kivi_r8 | 4 | snap | 145,911 | 0.8310 | outside sym range | |

## wikitext103 (budget 64 patches)

| Variant | bits | evict | mean BPB | mean bytes | vs full BPB |
|---|---|---|---|---|---|
| full | 16 | none | 1.6958 | 3,777,329 | +0.0000 |
| sym | 4 | full | 1.6962 | 1,003,353 | +0.0004 |
| sym | 3 | full | 1.6979 | 767,270 | +0.0021 |
| sym | 2 | full | 1.7259 | 531,187 | +0.0300 |
| sym_r8 | 4 | full | 1.6959 | 1,057,151 | +0.0001 |
| sym_r8 | 3 | full | 1.6962 | 825,646 | +0.0004 |
| sym_r8 | 2 | full | 1.7031 | 594,142 | +0.0073 |
| tok | 4 | full | 1.6961 | 1,062,374 | +0.0003 |
| tok | 3 | full | 1.6970 | 826,291 | +0.0012 |
| tok | 2 | full | 1.7033 | 590,208 | +0.0075 |
| kivi | 4 | full | 1.6959 | 1,285,750 | +0.0001 |
| kivi | 3 | full | 1.6960 | 1,053,976 | +0.0001 |
| kivi | 2 | full | 1.6968 | 822,201 | +0.0010 |
| kivi_r8 | 4 | full | 1.6959 | 1,299,418 | +0.0000 |
| kivi_r8 | 3 | full | 1.6959 | 1,068,914 | +0.0001 |
| kivi_r8 | 2 | full | 1.6964 | 838,411 | +0.0006 |
| sym | 4 | snap | 1.6985 | 148,349 | +0.0026 |
| sym | 3 | snap | 1.7002 | 113,444 | +0.0044 |
| sym | 2 | snap | 1.7244 | 78,538 | +0.0285 |
| kivi_r8 | 4 | snap | 1.6980 | 243,236 | +0.0022 |
| kivi_r8 | 3 | snap | 1.6981 | 213,910 | +0.0023 |
| kivi_r8 | 2 | snap | 1.6986 | 184,583 | +0.0028 |

| Comparison (a - b) | bytes a / bytes b | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| kivi|2|full|0 - sym|3|full|0 | 1.072 | -0.0009 [-0.0011, -0.0007] | -0.0015 [-0.0017, -0.0013] | -0.0009 [-0.0011, -0.0008] |
| kivi|2|full|0 - sym|2|full|0 | 1.548 | -0.0235 [-0.0245, -0.0224] | -0.0385 [-0.0401, -0.0369] | -0.0254 [-0.0268, -0.0240] |
| kivi_r8|2|full|0 - sym_r8|2|full|0 | 1.411 | -0.0053 [-0.0058, -0.0048] | -0.0082 [-0.0089, -0.0076] | -0.0068 [-0.0076, -0.0061] |
| kivi_r8|3|full|0 - sym|3|full|0 | 1.393 | -0.0018 [-0.0020, -0.0016] | -0.0023 [-0.0026, -0.0021] | -0.0019 [-0.0021, -0.0017] |
| kivi|2|full|0 - tok|2|full|0 | 1.393 | -0.0062 [-0.0066, -0.0059] | -0.0080 [-0.0083, -0.0076] | -0.0053 [-0.0057, -0.0050] |
| tok|3|full|0 - sym|3|full|0 | 1.077 | -0.0008 [-0.0011, -0.0005] | -0.0009 [-0.0012, -0.0007] | -0.0009 [-0.0011, -0.0007] |
| kivi_r8|3|snap|64 - sym|3|snap|64 | 1.886 | -0.0018 [-0.0020, -0.0016] | -0.0026 [-0.0028, -0.0023] | -0.0019 [-0.0021, -0.0018] |
| kivi_r8|2|snap|64 - sym|3|snap|64 | 1.627 | -0.0013 [-0.0015, -0.0011] | -0.0021 [-0.0023, -0.0018] | -0.0015 [-0.0016, -0.0013] |

Byte-matched: each variant against the piecewise-linear interpolation of the sym (or sym_r8) 2, 3 and 4-bit curve at the same mean bytes, means over seeds. Negative favours the variant.

| Variant | bits | evict | mean bytes | variant BPB | baseline BPB at same bytes | difference |
|---|---|---|---|---|---|---|
| tok | 2 | full | 590,208 | 1.7033 | 1.7189 (sym) | -0.0156 |
| tok | 3 | full | 826,291 | 1.6970 | 1.6975 (sym) | -0.0004 |
| tok | 4 | full | 1,062,374 | 1.6961 | outside sym range | |
| kivi | 2 | full | 822,201 | 1.6968 | 1.6975 (sym) | -0.0007 |
| kivi | 3 | full | 1,053,976 | 1.6960 | outside sym range | |
| kivi | 4 | full | 1,285,750 | 1.6959 | outside sym range | |
| kivi_r8 | 2 | full | 838,411 | 1.6964 | 1.6962 (sym_r8) | +0.0002 |
| kivi_r8 | 3 | full | 1,068,914 | 1.6959 | outside sym_r8 range | |
| kivi_r8 | 4 | full | 1,299,418 | 1.6959 | outside sym_r8 range | |
| kivi_r8 | 2 | snap | 184,583 | 1.6986 | outside sym range | |
| kivi_r8 | 3 | snap | 213,910 | 1.6981 | outside sym range | |
| kivi_r8 | 4 | snap | 243,236 | 1.6980 | outside sym range | |

