import numpy as np
import pytest
import torch

from kvc.cache import Policy, evict_one, mean_cache_entries, patch_entropy_scores, patch_scores, visible_mask
from models.blt import BLT
from models.entropy import BOS
from test_blt import SMALL


def make(n=80, seed=0):
    g = np.random.default_rng(seed)
    idx = torch.from_numpy(g.integers(0, 256, (1, n)))
    idx[0, 0] = BOS
    nxt = torch.from_numpy(g.random(n) < 0.35)[None]
    nxt[0, 0] = True
    return idx, nxt, g.random(n) * 3


def masked_forward(m, idx, nxt, h, policy):
    cs = nxt.long().cumsum(1)
    patch_in = (cs - nxt.long())[0].numpy()
    P = int(cs[0, -1]) + 1
    rng = np.random.default_rng(policy.seed)
    scores = patch_scores(policy, P, patch_entropy_scores(patch_in, h, P, policy.agg), rng)
    keep = torch.from_numpy(visible_mask(P, scores, policy))[None]
    with torch.no_grad():
        return m((idx, nxt, keep))[0], keep


@pytest.mark.parametrize("mode", ["recency", "random", "entropy", "entropy_inv"])
@pytest.mark.parametrize("agg", ["mean", "max"])
def test_masked_forward_equals_real_eviction(mode, agg):
    torch.manual_seed(0)
    m = BLT(**SMALL).eval()
    idx, nxt, h = make()
    policy = Policy(mode, budget=6, sink=1, recent=3, agg=agg, seed=5)
    full, keep = masked_forward(m, idx, nxt, h, policy)
    state, rows = m.init_state(), []
    for i in range(idx.size(1)):
        rows.append(m.step(state, idx[:, i], bool(nxt[0, i]), float(h[i]), policy))
    assert torch.allclose(full, torch.stack(rows, 1), atol=1e-5)
    assert state["glob"][0]["k"].size(2) == min(policy.budget, int(nxt.sum()))
    assert keep[0].sum(1).max() <= policy.budget


def test_eviction_actually_changes_outputs():
    torch.manual_seed(0)
    m = BLT(**SMALL).eval()
    idx, nxt, h = make()
    with torch.no_grad():
        base = m((idx, nxt))[0]
    out, _ = masked_forward(m, idx, nxt, h, Policy("recency", budget=4, sink=1, recent=2))
    assert not torch.allclose(base, out, atol=1e-5)


def test_full_policy_is_identity():
    torch.manual_seed(0)
    m = BLT(**SMALL).eval()
    idx, nxt, h = make()
    with torch.no_grad():
        base = m((idx, nxt))[0]
    out, _ = masked_forward(m, idx, nxt, h, Policy("full"))
    assert torch.allclose(base, out, atol=1e-6)


def test_visible_mask_is_monotone_and_respects_budget():
    scores = np.random.default_rng(1).random(50)
    policy = Policy("random", budget=10, sink=2, recent=4)
    mask = visible_mask(50, scores, policy)
    assert mask.sum(1).max() == 10 and np.all(np.diag(mask))
    for i in range(1, 50):
        assert not (mask[i] & ~mask[i - 1])[:i].any()
    assert mean_cache_entries(mask) <= 10


def test_protected_entries_are_never_evicted():
    ids = np.arange(12)
    scores = np.full(12, -1.0)
    pol = Policy("entropy", budget=5, sink=2, recent=3)
    drop = evict_one(ids, scores, pol)
    assert 2 <= ids[drop] < 9


def test_entropy_policy_keeps_high_entropy_patches():
    scores = np.zeros(30)
    scores[[7, 11]] = 5.0
    mask = visible_mask(30, scores, Policy("entropy", budget=8, sink=1, recent=3))
    assert mask[-1, 7] and mask[-1, 11]
    inv = visible_mask(30, -scores, Policy("entropy_inv", budget=8, sink=1, recent=3))
    assert not inv[-1, 7] and not inv[-1, 11]
