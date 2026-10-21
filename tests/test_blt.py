import pytest
import torch

from models.blt import BLT, expected_params
from models.entropy import BOS

SMALL = dict(local_dim=32, local_heads=2, local_window=8, enc_layers=1, dec_layers=2,
             global_dim=48, global_heads=2, global_layers=2)


def model():
    torch.manual_seed(0)
    return BLT(**SMALL).eval()


def sample(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 256, (1, n), generator=g)
    idx[0, 0] = BOS
    nxt = torch.rand(1, n, generator=g) < 0.25
    nxt[0, 0] = True
    return idx, nxt


def test_incremental_decode_matches_full_forward():
    m = model()
    idx, nxt = sample()
    with torch.no_grad():
        full = m((idx, nxt))[0]
    state, rows = m.init_state(), []
    for i in range(idx.size(1)):
        rows.append(m.step(state, idx[:, i], bool(nxt[0, i])))
    assert torch.allclose(full, torch.stack(rows, 1), atol=1e-5)
    assert state["patches"] == int(nxt.sum())
    assert state["glob"][0]["k"].size(2) == int(nxt.sum())


def test_no_leakage_from_future_bytes():
    m = model()
    idx, nxt = sample(seed=1)
    for j in (7, 20, 41):
        other = idx.clone()
        other[0, j:] = torch.randint(0, 256, (idx.size(1) - j,))
        with torch.no_grad():
            a, b = m((idx, nxt))[0], m((other, nxt))[0]
        assert torch.allclose(a[:, :j], b[:, :j], atol=1e-5)
        assert not torch.allclose(a[:, j:], b[:, j:], atol=1e-5)


def test_bytes_see_only_earlier_patch_summaries():
    m = model()
    idx = torch.randint(0, 256, (1, 24))
    idx[0, 0] = BOS
    nxt = torch.zeros(1, 24, dtype=torch.bool)
    nxt[0, [0, 6, 12, 18]] = True
    other = idx.clone()
    other[0, 13] = (other[0, 13] + 1) % 256
    with torch.no_grad():
        a, b = m((idx, nxt))[0], m((other, nxt))[0]
    assert torch.allclose(a[:, :13], b[:, :13], atol=1e-5)
    assert not torch.allclose(a[:, 13:], b[:, 13:], atol=1e-5)


def test_patch_summary_reaches_later_bytes_through_global_model_only():
    torch.manual_seed(0)
    m = BLT(**{**SMALL, "local_window": 2}).eval()
    idx = torch.randint(0, 256, (1, 24))
    idx[0, 0] = BOS
    nxt = torch.zeros(1, 24, dtype=torch.bool)
    nxt[0, [0, 6, 12, 18]] = True
    other = idx.clone()
    other[0, 1] = (other[0, 1] + 1) % 256
    with torch.no_grad():
        a, b = m((idx, nxt))[0], m((other, nxt))[0]
    assert not torch.allclose(a[:, 8:], b[:, 8:], atol=1e-6)


def test_document_boundaries_isolate_patches():
    m = model()
    idx, nxt = sample(n=40, seed=3)
    second = sample(n=40, seed=4)
    joined_idx, joined_nxt = torch.cat([second[0], idx], 1), torch.cat([second[1], nxt], 1)
    with torch.no_grad():
        alone = m((idx, nxt))[0]
        joined = m((joined_idx, joined_nxt))[0][:, 40:]
    assert torch.allclose(alone, joined, atol=1e-5)


def test_parameter_count_matches_hand_calculation():
    assert BLT(**SMALL).num_params() == expected_params(32, 1, 2, 48, 2)
    assert BLT().num_params() == expected_params() == 13_333_632
