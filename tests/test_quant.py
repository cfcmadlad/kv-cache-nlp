import numpy as np
import pytest
import torch

from kvc.cache import QuantPolicy, mean_entry_bytes, patch_entropy_scores, tier_mask
from kvc.quant import entry_bytes, fake_quant
from models.blt import BLT
from test_cache import make
from test_blt import SMALL


def forward_with_tiers(m, idx, nxt, h, qp):
    cs = nxt.long().cumsum(1)
    patch_in = (cs - nxt.long())[0].numpy()
    P = int(cs[0, -1]) + 1
    scores = patch_entropy_scores(patch_in, h, P, qp.agg)
    mask = tier_mask(qp, P, scores, np.random.default_rng(qp.seed))
    with torch.no_grad():
        return m((idx, nxt, None, (torch.from_numpy(mask)[None], qp.hi_bits, qp.lo_bits)))[0], mask


def test_fake_quant_levels_and_error():
    x = torch.randn(4, 64)
    for bits in (2, 4, 8):
        q = fake_quant(x, bits)
        assert all(torch.unique(row).numel() <= 2 * (2 ** (bits - 1) - 1) + 1 for row in q)
        assert (q - x).abs().max() <= x.abs().amax(-1).max() / (2 ** (bits - 1) - 1) / 2 + 1e-6
    assert torch.equal(fake_quant(x, 16), x)
    assert (fake_quant(x, 8) - x).abs().mean() < (fake_quant(x, 4) - x).abs().mean()


@pytest.mark.parametrize("mode", ["recency", "random", "entropy", "entropy_inv", "uniform"])
def test_tier_mask_forward_equals_real_cache(mode):
    torch.manual_seed(0)
    m = BLT(**SMALL).eval()
    idx, nxt, h = make()
    qp = QuantPolicy(mode, frac=0.4, hi_bits=8, lo_bits=3, window=3, sink=1, seed=2)
    full, _ = forward_with_tiers(m, idx, nxt, h, qp)
    state, rows = m.init_state(), []
    for i in range(idx.size(1)):
        rows.append(m.step(state, idx[:, i], bool(nxt[0, i]), float(h[i]), None, qp))
    assert torch.allclose(full, torch.stack(rows, 1), atol=1e-5)


def test_high_precision_everywhere_matches_unquantized():
    torch.manual_seed(0)
    m = BLT(**SMALL).eval()
    idx, nxt, h = make()
    with torch.no_grad():
        base = m((idx, nxt))[0]
    out, _ = forward_with_tiers(m, idx, nxt, h, QuantPolicy("uniform", lo_bits=16))
    assert torch.allclose(base, out, atol=1e-5)
    low, _ = forward_with_tiers(m, idx, nxt, h, QuantPolicy("uniform", lo_bits=2))
    assert not torch.allclose(base, low, atol=1e-3)


def test_byte_accounting():
    assert entry_bytes(6, 384, 6, 16) == 9216
    assert entry_bytes(6, 384, 6, 8) == 6 * (768 + 24)
    qp = QuantPolicy("recency", hi_bits=8, lo_bits=2, window=2, sink=0)
    mask = tier_mask(qp, 6, np.zeros(6), np.random.default_rng(0))
    per = lambda b: entry_bytes(6, 384, 6, b)
    expected = np.mean([min(i + 1, 2) * per(8) + max(i + 1 - 2, 0) * per(2) for i in range(6)])
    assert mean_entry_bytes(qp, mask, per) == pytest.approx(expected)


def test_entropy_tier_prefers_high_entropy_patches():
    scores = np.zeros(40)
    scores[[10, 25]] = 9.0
    qp = QuantPolicy("entropy", frac=0.1, sink=1)
    mask = tier_mask(qp, 40, scores, np.random.default_rng(0))
    assert mask[-1, 10] and mask[-1, 25] and mask[-1].sum() <= 8
