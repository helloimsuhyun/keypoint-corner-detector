"""Small geometric helpers used by 3-D plane/edge visualization."""

from __future__ import annotations

import numpy as np


def normalize(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return value / norm


def make_tangent_basis(normal):
    n = normalize(normal)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(n @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(reference, n))
    v = normalize(np.cross(n, u))
    return u, v, n


__all__ = ["make_tangent_basis", "normalize"]
