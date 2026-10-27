# BLT-style model with incremental decoding

## Architecture (`models/blt.py`)

The model is given the byte sequence `[BOS, b0, b1, ...]` and, for each position, a flag that announces whether the next byte opens a new patch. The flags are furnished by the entropy model (or by a fixed stride) and depend upon earlier bytes alone, so that the future is never consulted.

1. Local encoder: a byte embedding and one transformer layer (width 256, 4 heads, rotary positions, a causal window of 256 bytes).
2. Pooling: the encoder states of the bytes within a patch are averaged and a linear layer carries the mean to width 384. There are no hashed n-gram embeddings and no cross-attention pooling; this is a simplification of the original BLT, and the report says so.
3. Global model: 6 layers, width 384, 6 heads, rotary positions over the patch index, causal over patches. Its keys and values are the cache that the later phases compress.
4. Local decoder: the encoder state of the byte, joined to a linear image of the global output of the patch that precedes the one holding the byte to be predicted, passes through 2 transformer layers (width 256, window 256) and a 256-way head.

A byte in patch p sees the global outputs of patches before p, and none other. The BOS token is a patch by itself, so that even the first byte is predicted from a legitimate global state.

Parameters: 13,333,632, which agrees with the hand count in `expected_params` (each block costs 12 d^2 + 9 d; embeddings, projections, norms and head are added separately). The aim had been about 14M.

## Incremental decoding

`BLT.step(state, token, starts_next)` admits one byte at a time. It extends the local encoder cache (the last 256 positions) and adds the byte to the running patch sum; when the byte closes a patch, it sends the mean through the global model and appends one key and one value per global layer. The decoder then proceeds with a window cache of its own. In the global model a patch costs, at fp16, 2 (K and V) x 6 layers x 384 x 2 = 9,216 bytes; at 4.5 bytes per patch that comes to about 2.05 MB per thousand input bytes before any compression. The local caches are fixed windows and are reckoned as overhead.

## Tests (`tests/test_blt.py`)

- Step-by-step decoding equals the full forward pass within 1e-5, and the number of cached patches equals the number of start flags.
- Leakage: altering the bytes from position j onward never moves a logit before j, for three values of j.
- A byte inside a patch leaves the logits of earlier positions untouched, and a patch summary reaches later bytes only by way of the global model (tested with a window of 2, so that local attention cannot smuggle it across).
- Two documents in one sequence are blind to each other.
- The parameter count equals the hand calculation, for a small and for the full configuration.
