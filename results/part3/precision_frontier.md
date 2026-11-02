Tokens against precision at a fixed number of cache bytes: eviction (recency or the SnapKV-style rule) combined with uniform K and V quantization, entry counts 16 to 256 and 16, 8, 4, 3 and 2 bits, one entropy-patched model (seed 0), validation only. Cost is bits per byte (BPB); bytes are the mean resident global cache, as a share of the full fp16 cache. For each byte share, the best configuration at or below it, and the best configuration at each width.

## tinystories (full fp16 cache 839 KB, 0.7980 BPB)

| Byte share | Best configuration | BPB | Excess over full | Best at 16 bit | Best at 8 bit | Best at 4 bit | Best at 3 bit | Best at 2 bit |
|---|---|---|---|---|---|---|---|---|
| 3% | snap, 16 entries, 2-bit | 0.8814 | +0.0834 | n/a | n/a | n/a | n/a | 0.8814 (snap, 16) |
| 5% | snap, 24 entries, 3-bit | 0.8129 | +0.0149 | n/a | n/a | 0.8160 (snap, 16) | 0.8129 (snap, 24) | 0.8721 (snap, 32) |
| 8% | snap, 32 entries, 3-bit | 0.8086 | +0.0106 | n/a | n/a | 0.8091 (snap, 24) | 0.8086 (snap, 32) | 0.8692 (snap, 64) |
| 12% | snap, 48 entries, 4-bit | 0.8016 | +0.0036 | n/a | 0.8147 (snap, 16) | 0.8016 (snap, 48) | 0.8045 (snap, 64) | 0.8685 (snap, 128) |
| 20% | snap, 96 entries, 4-bit | 0.7996 | +0.0016 | 0.8147 (snap, 16) | 0.8037 (snap, 32) | 0.7996 (snap, 96) | 0.8027 (recency, 256) | 0.8685 (snap, 128) |
| 30% | snap, 256 entries, 4-bit | 0.7990 | +0.0009 | 0.8081 (snap, 24) | 0.7996 (snap, 64) | 0.7990 (snap, 256) | 0.8027 (recency, 256) | 0.8685 (snap, 128) |

## wikitext103 (full fp16 cache 3777 KB, 1.6801 BPB)

| Byte share | Best configuration | BPB | Excess over full | Best at 16 bit | Best at 8 bit | Best at 4 bit | Best at 3 bit | Best at 2 bit |
|---|---|---|---|---|---|---|---|---|
| 3% | snap, 48 entries, 4-bit | 1.6834 | +0.0033 | n/a | 1.6868 (snap, 24) | 1.6834 (snap, 48) | 1.6850 (snap, 48) | 1.7037 (snap, 64) |
| 5% | snap, 64 entries, 4-bit | 1.6826 | +0.0025 | 1.6905 (snap, 16) | 1.6848 (snap, 32) | 1.6826 (snap, 64) | 1.6834 (snap, 96) | 1.7031 (snap, 128) |
| 8% | snap, 128 entries, 4-bit | 1.6814 | +0.0014 | 1.6848 (snap, 32) | 1.6822 (snap, 64) | 1.6814 (snap, 128) | 1.6829 (snap, 128) | 1.7030 (snap, 192) |
| 12% | snap, 192 entries, 4-bit | 1.6810 | +0.0009 | 1.6830 (snap, 48) | 1.6815 (snap, 96) | 1.6810 (snap, 192) | 1.6822 (snap, 256) | 1.7030 (snap, 192) |
| 20% | snap, 256 entries, 4-bit | 1.6807 | +0.0006 | 1.6822 (snap, 64) | 1.6811 (snap, 128) | 1.6807 (snap, 256) | 1.6822 (snap, 256) | 1.7030 (snap, 192) |
| 30% | snap, 256 entries, 8-bit | 1.6803 | +0.0002 | 1.6811 (snap, 128) | 1.6803 (snap, 256) | 1.6807 (snap, 256) | 1.6822 (snap, 256) | 1.7030 (snap, 192) |

Cells give BPB with the policy and entry count in brackets.
