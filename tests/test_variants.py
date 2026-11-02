import numpy as np

from eval_variants import dynamic_mask, hybrid_scores, pct_rank
from kvc.cache import Policy, visible_mask


def test_pct_rank_orders_values():
    r = pct_rank(np.array([3.0, 1.0, 2.0]))
    assert r.tolist() == [1.0, 0.0, 0.5]


def test_hybrid_scores_are_causal_and_weighted():
    snap = np.random.default_rng(0).random((6, 6))
    ent = np.random.default_rng(1).random(6)
    out = hybrid_scores(snap, ent, 0.5)
    assert np.all(out[np.triu_indices(6, 1)] == 0)
    assert out[5, 0] == pct_rank(snap[5])[0] + 0.5 * pct_rank(ent)[0]


def test_dynamic_mask_is_causal_and_keeps_protected_patches():
    P, pol = 40, Policy("recency", 16, 1, 4)
    ent = np.random.default_rng(2).random(P)
    mask = dynamic_mask(P, ent, 16, pol)
    assert not np.triu(mask, 1).any()
    assert mask[:, 0].all()
    assert all(mask[i, max(0, i - 3):i + 1].all() for i in range(P))


def test_dynamic_mask_matches_recency_when_entropy_is_constant_and_high():
    P, pol = 40, Policy("recency", 24, 1, 4)
    ent = np.ones(P)
    expect = visible_mask(P, np.arange(P, dtype=float), Policy("recency", 36, 1, 4))
    assert (dynamic_mask(P, ent, 24, pol) == expect).all()
