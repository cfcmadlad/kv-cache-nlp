# Block 2.3: BLT-style model with incremental decoding

Rubric: Novelty (3), Demo and viva (20).

## Architecture (`models/blt.py`)

Input is the byte sequence `[BOS, b0, b1, ...]` and a flag per position saying whether the next byte starts a new patch. Flags come from the entropy model (or a fixed stride) and depend only on earlier bytes.

1. Local encoder: byte embedding, one transformer layer (width 256, 4 heads, RoPE, causal window of 256 bytes).
2. Pooling: mean of the encoder states of the bytes in a patch, then a linear layer to width 384. No hash n-gram embeddings and no cross-attention pooling (a simplification of BLT, to state in the report).
3. Global model: 6 layers, width 384, 6 heads, RoPE over patch index, causal over patches. Its key and value tensors are the cache we compress in later phases.
4. Local decoder: the byte's encoder state plus a linear map of the global output of the patch before the one that holds the byte being predicted, then 2 transformer layers (width 256, window 256), then a 256-way head.

Bytes in patch p see global outputs of patches before p only. The BOS token is a patch of its own, so the first byte is predicted from a valid global state.

Parameters: 13,333,632, equal to the hand count in `expected_params` (blocks cost 12 d^2 + 9 d each; embeddings, projections, norms and head are added separately). The target was about 14M.

## Incremental decoding

`BLT.step(state, token, starts_next)` feeds one byte. It extends the local encoder cache (last 256 positions), adds the byte to the running patch sum, and when the byte ends a patch it pushes the mean through the global model and appends one key/value pair per global layer. The decoder then runs with its own window cache. Cache bytes in the global model per patch at fp16: 2 (K and V) x 6 layers x 384 x 2 = 9,216. At 4.5 bytes per patch that is about 2.05 MB per 1000 input bytes before any compression. The local caches are fixed windows and counted as overhead.

## Tests (`tests/test_blt.py`)

- Step-by-step decoding equals the full forward pass within 1e-5, and the number of cached patches equals the number of start flags.
- Leakage: changing bytes from position j on never moves logits before j, for three values of j.
- A byte inside a patch does not change logits of earlier positions, and a patch summary reaches later bytes only through the global model (tested with a window of 2 so local attention cannot carry it).
- Two documents in one sequence do not see each other.
- The parameter count equals the hand calculation, for a small and the full configuration.

## Viva notes

Why mean pooling? It is the simplest pooling that needs no extra parameters beyond one linear layer; BLT's cross-attention pooling is a possible later upgrade.
Why is the BOS patch separate? Without it, the first byte would have no global state to read.
What is the compression target? The per-layer key and value tensors of the global model, one entry per patch.
