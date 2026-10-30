"""Continuous boundary-support weights shared by the 2-D edge pipeline."""

from __future__ import annotations

import numpy as np


def boundary_weights(
    distances_mm: np.ndarray,
    valid: np.ndarray,
    sigma_mm: float,
) -> np.ndarray:
    """Return one Gaussian boundary-proximity weight per observed point."""
    distances = np.asarray(distances_mm, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if distances.ndim != 1 or distances.shape != valid.shape:
        raise ValueError("distances_mm and valid must be 1-D arrays of equal length")
    if not np.isfinite(sigma_mm) or sigma_mm <= 0:
        raise ValueError("sigma_mm must be finite and positive")
    weight = np.zeros(len(distances), dtype=float)
    good = valid & np.isfinite(distances) & (distances >= 0)
    weight[good] = np.exp(-0.5 * (distances[good] / sigma_mm) ** 2)
    return weight


__all__ = ["boundary_weights"]
