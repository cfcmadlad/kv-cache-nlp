import numpy as np
import torch
import torch.nn.functional as F

from eval_stream import reset_nats, stream_nats
from kvc.cache import Policy
from models.blt import BLT
from models.entropy import BOS

SMALL = dict(local_dim=32, local_heads=2, local_window=16, enc_layers=1, dec_layers=1, global_dim=48, global_heads=2, global_layers=2)


def setup(n=200):
    torch.manual_seed(0)
    model = BLT(**SMALL).eval()
    rng = np.random.default_rng(0)
    doc = rng.integers(97, 123, n).astype(np.int64)
    h = rng.random(n).astype(np.float32)
    flags = np.zeros(n, dtype=bool)
    flags[::4] = True
    tokens = torch.from_numpy(np.concatenate([[BOS], doc[:-1]]).astype(np.int64))
    return model, doc, h, flags, tokens, torch.from_numpy(doc)


def test_stream_decoding_matches_full_forward_and_large_budget():
    model, doc, h, flags, tokens, targets = setup()
    with torch.no_grad():
        logits = model((tokens[None], torch.from_numpy(flags)[None]))[0][0]
    ref = F.cross_entropy(logits, targets, reduction="sum").item()
    assert abs(stream_nats(model, tokens, targets, flags, h, None, "cpu") - ref) < 1e-2
    assert abs(stream_nats(model, tokens, targets, flags, h, Policy("recency", 1000, 1, 2), "cpu") - ref) < 1e-2
    assert stream_nats(model, tokens, targets, flags, h, Policy("recency", 6, 1, 2), "cpu") != ref


def test_reset_blocks_equal_full_when_one_block():
    model, doc, h, flags, tokens, targets = setup(120)
    ps = {"mode": "fixed", "theta": None, "max_len": 8, "stride": 4}
    one = reset_nats(model, doc, h, ps, 1000, "cpu")
    two = reset_nats(model, doc, h, ps, 60, "cpu")
    assert one != two
    with torch.no_grad():
        f = torch.zeros(1, 120, dtype=torch.bool)
        f[0, ::4] = True
        ref = F.cross_entropy(model((tokens[None], f))[0][0], targets, reduction="sum").item()
    assert abs(one - ref) < 1e-2
