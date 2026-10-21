import math
from dataclasses import dataclass

import numpy as np

MODES = ("full", "recency", "random", "entropy", "entropy_inv")


@dataclass(frozen=True)
class Policy:
    mode: str = "full"
    budget: int = 0
    sink: int = 1
    recent: int = 8
    agg: str = "mean"
    seed: int = 0


def patch_entropy_scores(patch_in: np.ndarray, h: np.ndarray, num_patches: int, agg: str) -> np.ndarray:
    out = np.zeros(num_patches)
    if agg == "max":
        np.maximum.at(out, patch_in, h)
    else:
        np.add.at(out, patch_in, h)
        if agg == "mean":
            out /= np.maximum(np.bincount(patch_in, minlength=num_patches), 1)
    return out


def patch_scores(policy: Policy, num_patches: int, entropy_scores: np.ndarray = None, rng=None) -> np.ndarray:
    if policy.mode == "recency":
        return np.arange(num_patches, dtype=float)
    if policy.mode == "random":
        return rng.random(num_patches)
    if policy.mode == "entropy":
        return entropy_scores
    if policy.mode == "entropy_inv":
        return -entropy_scores
    return np.zeros(num_patches)


def evict_one(ids: np.ndarray, scores: np.ndarray, policy: Policy) -> int:
    protected = (ids < policy.sink) | (ids > ids.max() - policy.recent)
    candidates = np.flatnonzero(~protected)
    if len(candidates) == 0:
        raise ValueError(f"budget {policy.budget} too small for sink {policy.sink} and recent {policy.recent}")
    return int(candidates[np.lexsort((ids[candidates], scores[ids[candidates]]))[0]])


def visible_mask(num_patches: int, scores: np.ndarray, policy: Policy) -> np.ndarray:
    mask = np.zeros((num_patches, num_patches), dtype=bool)
    ids = np.empty(0, dtype=np.int64)
    for i in range(num_patches):
        ids = np.append(ids, i)
        if policy.mode != "full" and len(ids) > policy.budget:
            ids = np.delete(ids, evict_one(ids, scores, policy))
        mask[i, ids] = True
    return mask


def mean_cache_entries(mask: np.ndarray) -> float:
    return float(mask.sum(1).mean())


@dataclass(frozen=True)
class QuantPolicy:
    mode: str = "uniform"
    frac: float = 0.25
    hi_bits: int = 8
    lo_bits: int = 2
    window: int = 8
    sink: int = 1
    agg: str = "mean"
    seed: int = 0


QUANT_MODES = ("uniform", "recency", "random", "entropy", "entropy_inv")


def insertion_hi(qp: QuantPolicy, pid: int, score: float, seen: np.ndarray, draw: float) -> bool:
    if qp.mode == "uniform":
        return False
    if pid < qp.sink:
        return True
    if qp.mode == "random":
        return draw < qp.frac
    if qp.mode in ("entropy", "entropy_inv"):
        s = score if qp.mode == "entropy" else -score
        seen = seen if qp.mode == "entropy" else -seen
        return 1 + int((seen >= s).sum()) <= math.ceil(qp.frac * (len(seen) + 1))
    return False


def tier_flags(qp: QuantPolicy, num_patches: int, scores: np.ndarray, rng) -> np.ndarray:
    draws = rng.random(num_patches) if qp.mode == "random" else np.zeros(num_patches)
    return np.array([insertion_hi(qp, j, scores[j], scores[:j], draws[j]) for j in range(num_patches)])


def tier_mask(qp: QuantPolicy, num_patches: int, scores: np.ndarray, rng) -> np.ndarray:
    i, j = np.indices((num_patches, num_patches))
    if qp.mode == "recency":
        return (i - j < qp.window) | (j < qp.sink)
    return np.broadcast_to(tier_flags(qp, num_patches, scores, rng)[None], (num_patches, num_patches)).copy()


def tier_row(qp: QuantPolicy, ids: np.ndarray, flags: np.ndarray, pid: int) -> np.ndarray:
    if qp.mode == "recency":
        return (pid - ids < qp.window) | (ids < qp.sink)
    return flags


def mean_entry_bytes(qp: QuantPolicy, mask: np.ndarray, per_bits) -> float:
    lower = np.tril(np.ones_like(mask, dtype=bool))
    hi = (mask & lower).sum(1)
    n = lower.sum(1)
    return float(((hi * per_bits(qp.hi_bits) + (n - hi) * per_bits(qp.lo_bits))).mean())
