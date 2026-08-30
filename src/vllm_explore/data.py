"""Datasets and loaders. Start with something synthetic so the loop runs today."""

from __future__ import annotations

import numpy as np


def make_regression(n: int = 4096, dim: int = 32, noise: float = 0.1, seed: int = 0):
    """A trivial linear-ish dataset: y = Xw + b + noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype("float32")
    w = rng.standard_normal(dim).astype("float32")
    y = (X @ w + 0.3 + noise * rng.standard_normal(n)).astype("float32")
    return X, y
