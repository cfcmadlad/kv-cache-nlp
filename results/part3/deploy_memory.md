Cache memory per sequence at long contexts, computed from entry counts (6 layers, width 384, 6 heads in every model, so only the number of positions differs). The BPE Transformer is the Part 2 model at 4.0 bytes per token (measured on validation text: 4.1 for TinyStories, 3.9 for WikiText-103); a byte-level Transformer keeps one entry per byte; the byte-level BLT here makes one global entry per 4.5 bytes. Fixed costs not shown: the byte window caches of the BLT (786 KB at 256 bytes) and the entropy model's own window cache (about 2,100 KB), plus weights (13.3M and 3.3M parameters). Sizes in MB.

| Context (bytes) | Byte-level Transformer, full | BPE Transformer, full | BLT patches, full fp16 | BLT, 256 patches fp16 | BLT, 256 patches 4-bit | BLT, 64 patches 4-bit |
|---|---|---|---|---|---|---|
| 4,096 | 37.7 | 9.4 | 8.4 | 2.4 | 0.6 | 0.2 |
| 16,384 | 151.0 | 37.7 | 33.6 | 2.4 | 0.6 | 0.2 |
| 65,536 | 604.0 | 151.0 | 134.2 | 2.4 | 0.6 | 0.2 |
| 262,144 | 2,415.9 | 604.0 | 536.9 | 2.4 | 0.6 | 0.2 |
| 1,048,576 | 9,663.7 | 2,415.9 | 2,147.5 | 2.4 | 0.6 | 0.2 |