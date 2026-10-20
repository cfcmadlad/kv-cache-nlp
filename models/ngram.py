from __future__ import annotations

import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

VOCAB_SIZE = 256
START = -1


class KneserNeyByteLM:
    def __init__(self, order: int = 4):
        if order < 2:
            raise ValueError("order must be >= 2")
        self.order = order
        self.counts: list = [defaultdict(Counter) for _ in range(order)]
        self.cont: list = [defaultdict(Counter) for _ in range(order - 1)]
        self.discounts: list = [0.75] * order

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_stats", None)
        return state

    def _table(self, k: int) -> dict:
        return self.counts[k] if k == self.order - 1 else self.cont[k]

    def _context_stats(self) -> list:
        stats = self.__dict__.get("_stats")
        if stats is None:
            stats = [{ctx: (c, sum(c.values()), len(c)) for ctx, c in self._table(k).items() if c}
                     for k in range(self.order)]
            self._stats = stats
        return stats

    def fit(self, byte_docs: list) -> None:
        self.counts = [defaultdict(Counter) for _ in range(self.order)]
        self.cont = [defaultdict(Counter) for _ in range(self.order - 1)]
        self.__dict__.pop("_stats", None)
        for doc in tqdm(byte_docs, desc="counting n-grams", unit="doc"):
            padded = (START,) * (self.order - 1) + tuple(doc)
            for i in range(self.order - 1, len(padded)):
                byte = padded[i]
                for k in range(self.order):
                    self.counts[k][padded[i - k:i]][byte] += 1
        for k in range(self.order - 1):
            for context, next_counter in self.counts[k + 1].items():
                suffix = context[1:]
                for byte in next_counter:
                    self.cont[k][suffix][byte] += 1
        self._fit_discounts()

    def _fit_discounts(self) -> None:
        for k in range(self.order):
            n1 = n2 = 0
            for next_counter in self._table(k).values():
                for c in next_counter.values():
                    n1 += c == 1
                    n2 += c == 2
            d = n1 / (n1 + 2 * n2) if n1 + n2 else 0.75
            self.discounts[k] = min(max(d, 0.1), 0.9)

    def _prob(self, context: tuple, byte: int) -> float:
        stats = self._context_stats()
        p = 1.0 / VOCAB_SIZE
        for k in range(self.order):
            entry = stats[k].get(context[len(context) - k:] if k else ())
            if entry is None:
                continue
            counter, total, num_types = entry
            d = self.discounts[k]
            p = max(counter.get(byte, 0) - d, 0) / total + d * num_types / total * p
        return p

    def log2_prob_doc(self, doc: bytes) -> float:
        padded = (START,) * (self.order - 1) + tuple(doc)
        n = self.order - 1
        return sum(math.log2(max(self._prob(padded[i - n:i], padded[i]), 1e-12))
                   for i in range(n, len(padded)))

    def generate(self, prompt: bytes, max_new_bytes: int) -> bytes:
        n = self.order - 1
        padded = list((START,) * n + tuple(prompt))
        for _ in range(max_new_bytes):
            context = tuple(padded[-n:])
            probs = [self._prob(context, b) for b in range(VOCAB_SIZE)]
            padded.append(random.choices(range(VOCAB_SIZE), weights=probs)[0] if sum(probs) > 0 else 0)
        return bytes(padded[n:])

    def bits_per_byte(self, docs: list) -> float:
        docs = [d for d in docs if d]
        total_bytes = sum(map(len, docs))
        return -sum(map(self.log2_prob_doc, docs)) / total_bytes if total_bytes else float("nan")

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "KneserNeyByteLM":
        with open(path, "rb") as f:
            return _RestrictedUnpickler(f).load()


class _RestrictedUnpickler(pickle.Unpickler):
    ALLOWED = {("models.ngram", "KneserNeyByteLM"), ("collections", "Counter"),
               ("collections", "defaultdict")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from an n-gram checkpoint")
        return super().find_class(module, name)
