import math

import numpy as np

LN2 = math.log(2)


def bpb(nats: np.ndarray, nbytes: np.ndarray) -> float:
    return float(nats.sum() / LN2 / nbytes.sum())


def bootstrap_indices(num_docs: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, num_docs, size=(resamples, num_docs))


def bootstrap_ci(nats, nbytes, idx: np.ndarray, level: float = 0.95) -> tuple:
    vals = nats[idx].sum(1) / LN2 / nbytes[idx].sum(1)
    return tuple(np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2]).tolist())


def paired_diff_ci(nats_a, nats_b, nbytes, idx: np.ndarray, level: float = 0.95) -> tuple:
    vals = (nats_a[idx].sum(1) - nats_b[idx].sum(1)) / LN2 / nbytes[idx].sum(1)
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(vals.mean()), float(lo), float(hi)
