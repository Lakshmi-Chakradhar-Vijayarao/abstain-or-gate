"""Contamination check between training prompts and test questions, reported whatever it shows.

n-gram: for each test question, the share of its word 8-grams found anywhere in the training
questions (8 words: long enough that common phrasing rarely matches).
embedding (optional): each test question's highest cosine similarity to any training question.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def _grams(t: str, n: int) -> set[tuple[str, ...]]:
    w = _words(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def ngram_overlap(train: list[str], test: list[str], n: int = 8) -> list[float]:
    pool: set[tuple[str, ...]] = set().union(*(_grams(t, n) for t in train)) if train else set()
    out = []
    for q in test:
        g = _grams(q, n)
        out.append(len(g & pool) / len(g) if g else 0.0)
    return out


def report(train: list[str], test: list[str], embed: Callable[[list[str]], np.ndarray] | None = None,
           ngram_flag: float = 0.5, cos_flag: float = 0.9) -> dict:
    ov = ngram_overlap(train, test)
    out = {"n_train": len(train), "n_test": len(test), "ngram_n": 8,
           "ngram_max": max(ov, default=0.0), "ngram_mean": float(np.mean(ov)) if ov else 0.0,
           "ngram_flagged": sum(o >= ngram_flag for o in ov), "exact_duplicates": len(set(train) & set(test))}
    if embed is not None and train and test:
        a, b = embed(train), embed(test)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        best = (b @ a.T).max(axis=1)
        out.update({"cos_max": float(best.max()), "cos_mean": float(best.mean()),
                    "cos_flagged": int((best >= cos_flag).sum())})
    return out
