Measured step-by-step decoding of one stream on NVIDIA RTX PRO 6000 Blackwell Server Edition (batch 1, fp32 weights, Python loop, one byte per step). Cache bytes are counted from the tensors held after the given number of bytes, expressed at 2 bytes per element (fp16). `Global` is the patch cache, `local` is the fixed 3-layer byte window cache. Times are cumulative wall time. Eviction here uses real removal from the cache tensors (recency, first patch and 8 newest protected).

| Policy | Bytes decoded | Patches | Global cache (KB) | Local cache (KB) | Seconds | ms per byte since previous row |
|---|---|---|---|---|---|---|
| full | 2,048 | 476 | 4,387 | 786 | 1.8 | 0.86 |
| full | 8,192 | 1,872 | 17,252 | 786 | 7.1 | 0.88 |
| full | 32,768 | 7,511 | 69,221 | 786 | 37.9 | 1.25 |
| full | 65,536 | 14,816 | 136,544 | 786 | 102.2 | 1.96 |
| recency|64 | 2,048 | 476 | 590 | 786 | 2.8 | 1.37 |
| recency|64 | 8,192 | 1,872 | 590 | 786 | 8.3 | 0.90 |
| recency|64 | 32,768 | 7,511 | 590 | 786 | 30.8 | 0.91 |
| recency|64 | 65,536 | 14,816 | 590 | 786 | 61.0 | 0.92 |
| recency|128 | 2,048 | 476 | 1,180 | 786 | 1.8 | 0.89 |
| recency|128 | 8,192 | 1,872 | 1,180 | 786 | 7.3 | 0.90 |
| recency|128 | 32,768 | 7,511 | 1,180 | 786 | 29.8 | 0.91 |
| recency|128 | 65,536 | 14,816 | 1,180 | 786 | 60.0 | 0.92 |
| recency|256 | 2,048 | 476 | 2,359 | 786 | 1.8 | 0.88 |
| recency|256 | 8,192 | 1,872 | 2,359 | 786 | 7.3 | 0.90 |
| recency|256 | 32,768 | 7,511 | 2,359 | 786 | 29.7 | 0.91 |
| recency|256 | 65,536 | 14,816 | 2,359 | 786 | 59.9 | 0.92 |