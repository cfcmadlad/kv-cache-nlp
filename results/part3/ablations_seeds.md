Ablations repeated on seeds 1 and 2 (the main ablation table uses seed 0). Each cell is keeping high entropy minus random (positive means worse), mean and 95% document bootstrap interval (2000 resamples), at the larger budget of each dataset. Validation only.

| Dataset | Setting | seed 0 | seed 1 | seed 2 | Seeds with the interval above zero |
|---|---|---|---|---|---|
| tinystories (budget 64) | entropy model, mean (main setting) | +0.0076 [+0.0067, +0.0086] | +0.0117 [+0.0104, +0.0130] | +0.0162 [+0.0141, +0.0185] | 3/3 |
| tinystories (budget 64) | model's own entropy, mean | +0.0067 [+0.0058, +0.0076] | +0.0095 [+0.0083, +0.0106] | +0.0143 [+0.0123, +0.0164] | 3/3 |
| tinystories (budget 64) | entropy model, max | +0.0006 [+0.0001, +0.0012] | -0.0012 [-0.0021, -0.0003] | +0.0024 [+0.0011, +0.0038] | 2/3 |
| tinystories (budget 64) | entropy model, first byte | +0.0050 [+0.0041, +0.0060] | +0.0080 [+0.0066, +0.0095] | +0.0127 [+0.0105, +0.0150] | 3/3 |
| tinystories (budget 64) | entropy model, sum | -0.0002 [-0.0007, +0.0003] | -0.0024 [-0.0031, -0.0017] | -0.0001 [-0.0011, +0.0009] | 0/3 |
| tinystories (budget 64) | sink 0 | +0.0075 [+0.0066, +0.0085] | +0.0117 [+0.0105, +0.0130] | +0.0160 [+0.0140, +0.0183] | 3/3 |
| tinystories (budget 64) | sink 4 | +0.0068 [+0.0061, +0.0075] | +0.0093 [+0.0084, +0.0103] | +0.0140 [+0.0124, +0.0157] | 3/3 |
| wikitext103 (budget 128) | entropy model, mean (main setting) | +0.0052 [+0.0045, +0.0061] | +0.0120 [+0.0104, +0.0135] | +0.0113 [+0.0092, +0.0134] | 3/3 |
| wikitext103 (budget 128) | model's own entropy, mean | +0.0050 [+0.0043, +0.0058] | +0.0118 [+0.0103, +0.0132] | +0.0110 [+0.0091, +0.0130] | 3/3 |
| wikitext103 (budget 128) | entropy model, max | +0.0030 [+0.0019, +0.0042] | +0.0041 [+0.0030, +0.0053] | +0.0045 [+0.0035, +0.0054] | 3/3 |
| wikitext103 (budget 128) | entropy model, first byte | +0.0047 [+0.0039, +0.0056] | +0.0072 [+0.0063, +0.0081] | +0.0063 [+0.0052, +0.0075] | 3/3 |
| wikitext103 (budget 128) | entropy model, sum | +0.0035 [+0.0026, +0.0045] | +0.0033 [+0.0024, +0.0041] | +0.0051 [+0.0039, +0.0063] | 3/3 |
| wikitext103 (budget 128) | sink 0 | +0.0052 [+0.0045, +0.0060] | +0.0119 [+0.0104, +0.0134] | +0.0112 [+0.0092, +0.0133] | 3/3 |
| wikitext103 (budget 128) | sink 4 | +0.0052 [+0.0044, +0.0060] | +0.0118 [+0.0102, +0.0132] | +0.0111 [+0.0091, +0.0132] | 3/3 |