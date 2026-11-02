import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import REPO_ROOT
from eval_blt import load_blt
from eval_policies import attention_matrix
from kvc.bytedata import open_split
from kvc.cache import Policy, attention_scores, mean_cache_entries, patch_entropy_scores, patch_scores, visible_mask
from kvc.patching import doc_entropy_and_nats, patch_flags
from models.entropy import BOS, EntropyLM
from train_blt import patch_settings

NAMES = ("recency", "entropy", "snap", "snapv", "surprisal", "keydiff", "expected")
WINDOW = 16


@torch.no_grad()
def layer_qkv(model, inputs):
    store = []
    hooks = [blk.attn.qkv.register_forward_hook(lambda m, i, o: store.append(o.detach()[0])) for blk in model.glob]
    model(inputs)
    for hk in hooks:
        hk.remove()
    H = model.glob[0].attn.n_head
    out = []
    for o in store:
        P, C3 = o.shape
        q, k, v = o.view(P, 3, H, C3 // (3 * H)).unbind(1)
        out.append((q, k, v))
    return out


def window_mean(x: torch.Tensor, w: int) -> torch.Tensor:
    c = torch.cumsum(x, 0)
    lag = torch.cat([torch.zeros_like(c[:w]), c[:-w]]) if w < len(c) else torch.zeros_like(c)
    n = torch.clamp(torch.arange(1, len(c) + 1, device=x.device), max=w).view(-1, *([1] * (x.dim() - 1)))
    return (c - lag) / n


@torch.no_grad()
def keydiff_scores(qkv) -> np.ndarray:
    total = 0.0
    for _, k, _ in qkv:
        kn = F.normalize(k, dim=-1)
        anchor = F.normalize(window_mean(kn, WINDOW), dim=-1)
        total = total + torch.einsum("ihd,jhd->ij", anchor, kn) / kn.size(1)
    return (-total / len(qkv)).cpu().numpy()


@torch.no_grad()
def expected_scores(qkv) -> np.ndarray:
    P = qkv[0][0].size(0)
    causal = torch.tril(torch.ones(P, P, dtype=torch.bool, device=qkv[0][0].device))
    total = 0.0
    for q, k, _ in qkv:
        mu = window_mean(q, WINDOW)
        var = (window_mean(q * q, WINDOW) - mu * mu).clamp(min=0)
        d = q.size(-1)
        logit = (torch.einsum("ihd,jhd->hij", mu, k) + 0.5 * torch.einsum("ihd,jhd->hij", var, k * k)) / d ** 0.5
        logit = logit.masked_fill(~causal[None], float("-inf"))
        total = total + torch.softmax(logit, -1).mean(0)
    return (total / len(qkv)).cpu().numpy()


def value_norm(qkv) -> np.ndarray:
    return torch.stack([v.abs().sum((-1, -2)) for _, _, v in qkv]).mean(0).cpu().numpy().astype(np.float64)


def load_entropy_model(cfg, device):
    ckpt = torch.load(REPO_ROOT / "runs" / cfg["patching"]["entropy_run"] / "model.pt", map_location=device, weights_only=True)
    m = ckpt["config"]["model"]
    model = EntropyLM(m["n_layer"], m["n_head"], m["dim"], m["window"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model


@torch.no_grad()
def build_mask(name, model, inputs, P, ent, surprisal, qkv, pol, device, snap_window):
    if name in ("snap", "snapv"):
        keep, vn = None, value_norm(qkv)
        for _ in range(2):
            sc = attention_scores(attention_matrix(model, inputs, keep), "snap", snap_window)
            if name == "snapv":
                sc = sc * vn[None]
            mask = visible_mask(P, sc, pol)
            keep = torch.from_numpy(mask)[None].to(device)
        return mask
    if name == "keydiff":
        return visible_mask(P, keydiff_scores(qkv), pol)
    if name == "expected":
        return visible_mask(P, expected_scores(qkv), pol)
    if name == "surprisal":
        return visible_mask(P, surprisal, pol)
    return visible_mask(P, patch_scores(Policy(name, pol.budget, pol.sink, pol.recent), P, ent), pol)


@torch.no_grad()
def run(model, cfg, args, device):
    ps = patch_settings(cfg)
    base = REPO_ROOT / "data" / "bytes" / cfg["dataset"]
    data, offsets = open_split(base, args.split)
    entropy = np.load(base / f"entropy_{cfg['patching']['entropy_run']}" / f"{args.split}.npy", mmap_mode="r")
    ent_model = load_entropy_model(cfg, device)
    count = min(args.max_docs, len(offsets) - 1)
    variants = [("full", 0)] + [(n, args.budget) for n in NAMES]
    nats = {v: np.zeros(count) for v in variants}
    entries = {v: [] for v in variants}
    sizes = np.zeros(count, dtype=np.int64)
    block = cfg["block"]
    for di in range(count):
        s, e = offsets[di], offsets[di + 1]
        doc, h = np.asarray(data[s:e]), entropy[s:e].astype(np.float32)
        _, byte_nats = doc_entropy_and_nats(ent_model, bytes(doc.astype(np.uint8)), device)
        sizes[di] = len(doc)
        for a in range(0, len(doc), block):
            b = min(a + block, len(doc))
            x = np.empty(b - a, dtype=np.int64)
            x[0] = BOS if a == 0 else doc[a - 1]
            x[1:] = doc[a:b - 1]
            flags = patch_flags(h[a:b] if ps["mode"] == "entropy" else None, np.arange(a, b), np.zeros(b - a, dtype=bool), ps["mode"], ps["theta"], ps["max_len"], ps["stride"])
            cs = np.cumsum(flags)
            patch_in = cs - flags
            P = int(cs[-1]) + 1
            inputs = (torch.from_numpy(x)[None].to(device), torch.from_numpy(flags)[None].to(device))
            tgt = torch.from_numpy(doc[a:b].astype(np.int64)).to(device)
            full = F.cross_entropy(model(inputs)[0][0].float(), tgt, reduction="sum").item()
            ent = patch_entropy_scores(patch_in, h[a:b].astype(np.float64), P, "mean")
            surprisal = patch_entropy_scores(patch_in, byte_nats[a:b].astype(np.float64), P, "mean")
            qkv = layer_qkv(model, inputs) if P > args.budget else None
            for v in variants:
                name, budget = v
                if name == "full" or budget >= P:
                    nats[v][di] += full
                    entries[v].append((P + 1) / 2)
                    continue
                pol = Policy("snap", budget, 1, 8)
                mask = build_mask(name, model, inputs, P, ent, surprisal, qkv, pol, device, args.snap_window)
                logits = model((*inputs, torch.from_numpy(mask)[None].to(device)))[0][0]
                nats[v][di] += F.cross_entropy(logits.float(), tgt, reduction="sum").item()
                entries[v].append(mean_cache_entries(mask))
    return {"run": args.run, "dataset": cfg["dataset"], "split": args.split, "docs": count, "bytes_per_doc": sizes.tolist(),
            "variants": {f"{n}|{b}": {"nats": nats[(n, b)].tolist(), "mean_entries": float(np.mean(entries[(n, b)]))} for n, b in variants}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--snap-window", type=int, default=16)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_blt(args.run, device)
    result = run(model, cfg, args, device)
    out = REPO_ROOT / "runs" / args.run / f"extras_{args.split}_{result['docs']}.json"
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
