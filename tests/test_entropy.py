import numpy as np
import torch

from kvc.patching import entropy_from_logits, starts_from_entropy, tune_threshold, word_start_stats
from models.entropy import BOS, EntropyLM


def small():
    torch.manual_seed(0)
    return EntropyLM(n_layer=2, n_head=2, dim=32, window=16).eval()


def test_model_is_causal():
    m = small()
    a = torch.randint(0, 256, (1, 80))
    b = a.clone()
    b[0, 40:] = torch.randint(0, 256, (40,))
    with torch.no_grad():
        la, lb = m(a)[0], m(b)[0]
    assert torch.allclose(la[:, :40], lb[:, :40], atol=1e-5)
    assert not torch.allclose(la[:, 41:], lb[:, 41:], atol=1e-5)


def test_window_limits_receptive_field():
    m = small()
    a = torch.randint(0, 256, (1, 100))
    b = a.clone()
    b[0, 0] = (b[0, 0] + 1) % 256
    with torch.no_grad():
        la, lb = m(a)[0], m(b)[0]
    reach = 2 * 15 + 1
    assert torch.allclose(la[:, reach + 1:], lb[:, reach + 1:], atol=1e-5)
    assert not torch.allclose(la[:, :16], lb[:, :16], atol=1e-5)


def test_blockwise_equals_dense_masking():
    m = small()
    x = torch.randint(0, 256, (2, 70))
    full = m(x)[0]
    single = torch.cat([m(x[:, :t + 1])[0][:, -1:] for t in range(70)], dim=1)
    assert torch.allclose(full, single, atol=1e-4)


def test_documents_do_not_see_each_other():
    m = small()
    doc = torch.randint(0, 256, (1, 30))
    prefix = torch.cat([torch.tensor([[BOS]]), torch.randint(0, 256, (1, 20))], dim=1)
    alone = torch.cat([torch.tensor([[BOS]]), doc], dim=1)
    joined = torch.cat([prefix, alone], dim=1)
    with torch.no_grad():
        assert torch.allclose(m(alone)[0], m(joined)[0][:, prefix.size(1):], atol=1e-5)


def test_boundaries_ignore_the_future():
    rng = np.random.default_rng(0)
    h = rng.random(500) * 3
    for cut in (50, 137, 400):
        full = starts_from_entropy(h.copy(), 1.5, 8)
        prefix = starts_from_entropy(h[:cut].copy(), 1.5, 8)
        assert np.array_equal(full[full < cut], prefix)


def test_boundaries_depend_only_on_past_bytes_end_to_end():
    m = small()
    a = torch.randint(0, 256, (1, 120))
    b = a.clone()
    b[0, 60:] = torch.randint(0, 256, (60,))
    with torch.no_grad():
        ha = entropy_from_logits(m(torch.cat([torch.tensor([[BOS]]), a[:, :-1]], 1))[0])[0].numpy()
        hb = entropy_from_logits(m(torch.cat([torch.tensor([[BOS]]), b[:, :-1]], 1))[0])[0].numpy()
    sa, sb = starts_from_entropy(ha, 5.0, 12), starts_from_entropy(hb, 5.0, 12)
    assert np.array_equal(sa[sa <= 60], sb[sb <= 60])


def test_max_len_and_first_start():
    h = np.zeros(100)
    s = starts_from_entropy(h, 1.0, 16)
    assert s[0] == 0 and np.diff(np.append(s, 100)).max() == 16


def test_tuned_threshold_hits_target():
    rng = np.random.default_rng(1)
    es = [rng.random(2000) * 4 for _ in range(5)]
    t = tune_threshold(es, 4.5, 0)
    got = sum(len(e) for e in es) / sum(len(starts_from_entropy(e, t)) for e in es)
    assert abs(got - 4.5) < 0.05


def test_word_start_stats():
    doc = b"the cat sat"
    stats = word_start_stats([doc], [np.array([0, 4, 8])])
    assert stats["word_start_rate"] == 1.0 and abs(stats["chance_rate"] - 3 / 11) < 1e-9
