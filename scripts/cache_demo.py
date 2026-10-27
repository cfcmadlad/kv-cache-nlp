import math

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import GLOBAL_CACHE_BYTES_PER_PATCH, load_blt
from kvc.cache import Policy, mean_cache_entries, patch_entropy_scores, patch_scores, visible_mask
from kvc.patching import doc_entropy_and_nats, patch_flags
from kvc.quant import entry_bytes
from models.entropy import BOS, EntropyLM
from train_blt import patch_settings

MAX_BYTES = 4096
POLICIES = ["recency", "random", "entropy", "entropy_inv"]


def load_pair(dataset: str, device: str):
    model, cfg = load_blt(f"blt_{dataset}_entropy", device)
    ckpt = torch.load(REPO_ROOT / "runs" / cfg["patching"]["entropy_run"] / "model.pt", map_location=device, weights_only=True)
    m = ckpt["config"]["model"]
    entropy = EntropyLM(m["n_layer"], m["n_head"], m["dim"], m["window"]).to(device).eval()
    entropy.load_state_dict(ckpt["state_dict"])
    return model, entropy, cfg


def mark_patches(doc: bytes, flags: np.ndarray) -> str:
    parts = np.split(np.frombuffer(doc, dtype=np.uint8), np.flatnonzero(flags)[1:])
    return "|".join(bytes(p).decode("utf-8", errors="replace").replace("\n", " ") for p in parts)


@torch.no_grad()
def analyse(text: str, model, entropy, cfg: dict, budget: int, device: str, random_seeds: int = 3, sink: int = 1, recent: int = 8) -> dict:
    doc = text.encode("utf-8")[:MAX_BYTES]
    ps = patch_settings(cfg)
    h = doc_entropy_and_nats(entropy, doc, device)[0]
    n = len(doc)
    flags = patch_flags(h, np.arange(n), np.zeros(n, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
    cs = np.cumsum(flags)
    P = int(cs[-1]) + 1
    ent = patch_entropy_scores(cs - flags, h.astype(np.float64), P, "mean")
    x = np.array([BOS, *doc[:-1]], dtype=np.int64)
    inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
    target = torch.from_numpy(np.frombuffer(doc, dtype=np.uint8).astype(np.int64)).to(device)
    m = cfg["model"]
    per_patch = GLOBAL_CACHE_BYTES_PER_PATCH(m)
    per_bits = lambda b: entry_bytes(m["global_layers"], m["global_dim"], m["global_heads"], b)

    def bpb(extra=()):
        logits = model((*inputs, *extra))[0][0]
        return F.cross_entropy(logits.float(), target, reduction="sum").item() / math.log(2) / n

    rows = [("full cache (fp16)", bpb(), float(np.mean(np.arange(1, P + 1)) * per_patch))]
    for mode in POLICIES:
        pol = Policy(mode, budget, sink, recent)
        reps = random_seeds if mode == "random" else 1
        total, kept = 0.0, 0.0
        for r in range(reps):
            mask = visible_mask(P, patch_scores(pol, P, ent, np.random.default_rng([0, r])), pol)
            total += bpb((torch.from_numpy(mask)[None].to(device),))
            kept += mean_cache_entries(mask)
        rows.append((f"keep {budget} patches: {mode}", total / reps, kept / reps * per_patch))
    for bits in (4, 3):
        rows.append((f"uniform {bits}-bit", bpb((None, (torch.from_numpy(np.zeros((P, P), dtype=bool))[None].to(device), 16, bits))),
                     float(np.mean(np.arange(1, P + 1)) * per_bits(bits))))
    return {"bytes": n, "patches": P, "mean_patch_len": n / P, "marked": mark_patches(doc, flags), "rows": rows}
