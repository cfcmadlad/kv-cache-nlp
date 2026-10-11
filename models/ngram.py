"""Classical baseline: byte-level n-gram LM with interpolated Kneser-Ney smoothing.

Trained by scripts/train_ngram.py and saved as a pickle of this class; loaded again by
scripts/data_utils.load_ngram for the demo, error analysis and check_setup.py.
Vocabulary is the 256 raw byte values, so there is no tokenizer.
Reference: Kneser & Ney (1995); interpolated formulation from Chen & Goodman (1999).
"""

from __future__ import annotations

import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

VOCAB_SIZE = 256
START = -1  # sentinel context symbol, outside 0-255, marks document start


class KneserNeyByteLM:
    """Interpolated Kneser-Ney n-gram model over raw bytes.

    `order` is the n-gram order (order=4 uses up to 3 bytes of left context).
    Context is reset at document boundaries so counts never leak across docs.

    Every order k (0 = unigram .. order-1 = highest) is scored by the same
    interpolation formula:
        P(byte | context) = max(c - D, 0) / total + D * num_types / total * lower
    `lower` is the order-(k-1) estimate; at k == 0 it bottoms out at a uniform
    distribution over the vocabulary. The only thing that differs per order is
    which count table backs `c`/`total`/`num_types`: raw counts at the top
    order, continuation counts (from `_table`) below that.
    """

    def __init__(self, order: int = 4):
        if order < 2:
            raise ValueError("order must be >= 2 (Kneser-Ney needs bigram continuation counts)")
        self.order = order
        # counts[k]: context (tuple, len k) -> Counter(next_byte -> raw count), k = 0..order-1
        self.counts: list = [defaultdict(Counter) for _ in range(order)]
        # cont[k]: context (tuple, len k) -> Counter(next_byte -> continuation count), k = 0..order-2
        self.cont: list = [defaultdict(Counter) for _ in range(order - 1)]
        self.discounts: list = [0.75] * order

    def _table(self, k: int) -> dict:
        """Count table backing order k: raw counts at the top order, else continuation counts."""
        return self.counts[k] if k == self.order - 1 else self.cont[k]

    def fit(self, byte_docs: list) -> None:
        """Count n-grams of every order, derive continuation counts, then fit the discounts."""
        # Reset state so fit() is safe to call more than once on the same instance.
        self.counts = [defaultdict(Counter) for _ in range(self.order)]
        self.cont = [defaultdict(Counter) for _ in range(self.order - 1)]

        for doc in tqdm(byte_docs, desc="counting n-grams", unit="doc"):
            padded = (START,) * (self.order - 1) + tuple(doc)
            for i in range(self.order - 1, len(padded)):
                byte = padded[i]
                for k in range(self.order):
                    context = padded[i - k:i]
                    self.counts[k][context][byte] += 1

        # Continuation counts: for order k, count distinct (order k+1) contexts
        # that support each (shorter context, byte) pair, not raw frequency. This is what
        # makes it Kneser-Ney: a byte that follows many different contexts gets more
        # lower-order mass than one that is frequent only after a single context.
        for k in range(self.order - 1):
            for context, next_counter in self.counts[k + 1].items():
                suffix = context[1:]
                for byte in next_counter:
                    self.cont[k][suffix][byte] += 1

        self._fit_discounts()

    def _fit_discounts(self) -> None:
        # Absolute discount per order from count-of-counts: D = n1 / (n1 + 2 * n2), where
        # n1/n2 are how many (context, byte) pairs have count exactly 1/2 in that order's
        # table (Ney, Essen & Kneser 1994). Clamped to [0.1, 0.9] for stability.
        for k in range(self.order):
            n1 = n2 = 0
            for next_counter in self._table(k).values():
                for c in next_counter.values():
                    if c == 1:
                        n1 += 1
                    elif c == 2:
                        n2 += 1
            d = n1 / (n1 + 2 * n2) if (n1 + n2) > 0 else 0.75
            self.discounts[k] = min(max(d, 0.1), 0.9)

    def _prob(self, context: tuple, byte: int, k: int) -> float:
        """P(byte | context) at order k (len(context) == k), recursing to k-1."""
        lower = 1.0 / VOCAB_SIZE if k == 0 else self._prob(context[1:], byte, k - 1)

        next_counter = self._table(k).get(context)
        if not next_counter:
            return lower

        total = sum(next_counter.values())
        c = next_counter.get(byte, 0)
        num_types = len(next_counter)
        d = self.discounts[k]
        return max(c - d, 0) / total + d * num_types / total * lower

    def log2_prob_doc(self, doc: bytes) -> float:
        """log2 P(doc): sum over every byte, each predicted from up to order-1 previous bytes."""
        padded = (START,) * (self.order - 1) + tuple(doc)
        total_log2 = 0.0
        for i in range(self.order - 1, len(padded)):
            context = padded[i - (self.order - 1):i]
            byte = padded[i]
            p = max(self._prob(context, byte, self.order - 1), 1e-12)
            total_log2 += math.log2(p)
        return total_log2

    def generate(self, prompt: bytes, max_new_bytes: int) -> bytes:
        """Sample max_new_bytes of continuation from the model's own distribution."""
        padded = list((START,) * (self.order - 1) + tuple(prompt))
        for _ in range(max_new_bytes):
            context = tuple(padded[-(self.order - 1):])
            probs = [self._prob(context, b, self.order - 1) for b in range(VOCAB_SIZE)]
            total = sum(probs)
            byte = random.choices(range(VOCAB_SIZE), weights=probs)[0] if total > 0 else 0
            padded.append(byte)
        return bytes(padded[self.order - 1:])

    def bits_per_byte(self, docs: list) -> float:
        """Total bits over all docs divided by total bytes (not a mean of per-doc BPB)."""
        total_log2 = 0.0
        total_bytes = 0
        for doc in docs:
            if not doc:
                continue
            total_log2 += self.log2_prob_doc(doc)
            total_bytes += len(doc)
        return -total_log2 / total_bytes if total_bytes else float("nan")

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "KneserNeyByteLM":
        # The pickle refers to the class as `models.ngram.KneserNeyByteLM`, so the repo root
        # must be on sys.path before loading (importing scripts/data_utils.py does that).
        # For the same reason, renaming this module or class breaks existing checkpoints.
        with open(path, "rb") as f:
            return _RestrictedUnpickler(f).load()


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only rebuilds the classes a saved KneserNeyByteLM contains.

    A plain pickle.load can run arbitrary code hidden in a tampered file; this one refuses
    anything outside the three classes our checkpoints actually use."""

    ALLOWED = {("models.ngram", "KneserNeyByteLM"), ("collections", "Counter"),
               ("collections", "defaultdict")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name} from an n-gram checkpoint")
        return super().find_class(module, name)
