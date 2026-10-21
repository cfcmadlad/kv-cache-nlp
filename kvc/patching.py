import numpy as np
import torch
import torch.nn.functional as F

from models.entropy import BOS


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def starts_from_entropy(entropy: np.ndarray, threshold: float, max_len: int = 0) -> np.ndarray:
    mask = entropy > threshold
    mask[0] = True
    starts = np.flatnonzero(mask)
    if max_len:
        ends = np.append(starts[1:], len(entropy))
        extra = [np.arange(s + max_len, e, max_len) for s, e in zip(starts, ends) if e - s > max_len]
        starts = np.sort(np.concatenate([starts, *extra]))
    return starts


@torch.no_grad()
def doc_entropy_and_nats(model, doc: bytes, device: str):
    ids = torch.tensor([BOS, *doc[:-1]], device=device)[None]
    targets = torch.tensor(list(doc), device=device)
    logits = model(ids)[0][0]
    nats = F.cross_entropy(logits.float(), targets, reduction="none")
    return entropy_from_logits(logits).cpu().numpy(), nats.cpu().numpy()


def mean_patch_length(entropies: list, threshold: float, max_len: int) -> float:
    total = sum(len(e) for e in entropies)
    return total / sum(len(starts_from_entropy(e, threshold, max_len)) for e in entropies)


def tune_threshold(entropies: list, target: float, max_len: int, iters: int = 40) -> float:
    lo, hi = 0.0, float(max(e.max() for e in entropies))
    for _ in range(iters):
        mid = (lo + hi) / 2
        if mean_patch_length(entropies, mid, max_len) > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def is_word_byte(b: np.ndarray) -> np.ndarray:
    return ((b >= 48) & (b <= 57)) | ((b >= 65) & (b <= 90)) | ((b >= 97) & (b <= 122)) | (b >= 128)


def word_start_stats(docs: list, starts_per_doc: list) -> dict:
    hits = starts = positions = word_positions = space_starts = 0
    for doc, st in zip(docs, starts_per_doc):
        b = np.frombuffer(doc, dtype=np.uint8)
        w = is_word_byte(b)
        is_ws = np.zeros(len(b), dtype=bool)
        is_ws[1:] = w[1:] & ~w[:-1]
        is_ws[0] = w[0]
        hits += int(is_ws[st].sum())
        space_starts += int((b[st] == 32).sum())
        starts += len(st)
        word_positions += int(is_ws.sum())
        positions += len(b)
    return {"starts": starts, "word_start_rate": hits / starts, "chance_rate": word_positions / positions,
            "space_start_rate": space_starts / starts}


def patch_flags(h, rel: np.ndarray, forced: np.ndarray, mode: str, theta: float, max_len: int, stride: int) -> np.ndarray:
    flags = (rel % stride == 0) if mode == "fixed" else (h > theta)
    flags = flags | forced
    flags[0] = True
    if mode == "entropy" and max_len:
        starts = starts_from_entropy(np.where(flags, np.inf, -np.inf), 0.0, max_len)
        flags = np.zeros(len(flags), dtype=bool)
        flags[starts] = True
    return flags


@torch.no_grad()
def batched_entropies(model, docs: list, device: str, budget: int = 65536) -> list:
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]))
    out, i = [None] * len(docs), 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * len(docs[order[j]]) <= budget:
            j += 1
        j = max(j, i + 1)
        group = order[i:j]
        longest = max(len(docs[k]) for k in group)
        ids = torch.zeros(len(group), longest, dtype=torch.long)
        for r, k in enumerate(group):
            ids[r, 0] = BOS
            ids[r, 1:len(docs[k])] = torch.tensor(list(docs[k][:-1]), dtype=torch.long)
        logits = model(ids.to(device))[0]
        h = entropy_from_logits(logits).cpu().numpy()
        for r, k in enumerate(group):
            out[k] = h[r, :len(docs[k])].astype(np.float32)
        i = j
    return out
