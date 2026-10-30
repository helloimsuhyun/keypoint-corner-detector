"""Bounded-memory quantile soft votes. No smoothing or coordinate averaging."""

from time import perf_counter

import numpy as np
from scipy.special import expit


def linear_quantile_99(projections):
    """Select two ranks, retaining NumPy's default linear quantile semantics.

    Partition a copy: changing the original columns would destroy point IDs.
    Inputs are finite C-contiguous direction x point projection blocks.
    """
    rank = 0.99 * (projections.shape[1] - 1)
    lower = int(np.floor(rank))
    upper = min(lower + 1, projections.shape[1] - 1)
    fraction = rank - lower
    selected = np.partition(projections, (lower, upper), axis=1)
    a, b = selected[:, lower], selected[:, upper]
    difference = (b - a).astype(np.float64)
    # Match np.quantile's stable interpolation, including float32 differences.
    threshold = (b.astype(np.float64) - difference * (1 - fraction) if fraction >= 0.5
                 else a.astype(np.float64) + difference * fraction)
    return threshold.astype(projections.dtype)


def point_direction_diagnostics(delta, directions, cfg, point_index):
    """Read-only directional mass; an index array batches several selected points."""
    points = np.ascontiguousarray(delta, dtype=cfg.voting_dtype)
    vectors = np.ascontiguousarray(directions, dtype=cfg.voting_dtype)
    indices = np.asarray(point_index)
    scalar = indices.ndim == 0
    if indices.ndim > 1 or indices.dtype.kind not in "iu":
        raise IndexError("point_index must address the component search support")
    indices = indices.reshape(-1)
    if np.any(indices < 0) or np.any(indices >= len(points)) or len(np.unique(indices)) != len(indices):
        raise IndexError("point indices must be unique and inside the component search support")
    projection = np.empty((len(vectors), len(indices)), dtype=cfg.voting_dtype)
    # Threshold is shared by the selected columns, not independently estimated.
    threshold = np.empty(len(vectors), dtype=cfg.voting_dtype)
    contribution = np.zeros_like(projection)
    # Keep hard argmax's original multiplication shape and exact tie behavior.
    hard_projection = points @ vectors.T if cfg.voting_mode == "hard" else None
    for start in range(0, len(vectors), 64):
        stop = min(start + 64, len(vectors))
        values = np.ascontiguousarray(hard_projection[:, start:stop].T if hard_projection is not None
                                      else vectors[start:stop] @ points.T)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite diagnostic projection; use float64")
        cutoff = (np.quantile(values, 0.99, axis=1).astype(cfg.voting_dtype)
                  if cfg.voting_mode == "soft_dense" else linear_quantile_99(values))
        projection[start:stop] = values[:, indices]
        threshold[start:stop] = cutoff
        if cfg.voting_mode == "hard":
            contribution[start:stop] = np.argmax(values, axis=1)[:, None] == indices
        elif cfg.voting_mode == "soft_sparse":
            rows, columns = np.nonzero(values >= cutoff[:, None] - cfg.soft_alpha * cfg.soft_tau_mm)
            weights = (values[rows, columns] - cutoff[rows]) / cfg.soft_tau_mm
            expit(weights, out=weights)
            mass = np.bincount(rows, weights=weights, minlength=len(cutoff)).astype(cfg.voting_dtype)
            weights /= mass[rows]
            for column, point in enumerate(indices):
                selected = columns == point
                contribution[start + rows[selected], column] = weights[selected]
        else:
            values -= cutoff[:, None]
            values /= cfg.soft_tau_mm
            expit(values, out=values)
            values /= values.sum(axis=1, keepdims=True)
            contribution[start:stop] = values[:, indices]
    return (projection[:, 0], threshold, contribution[:, 0]) if scalar else (projection, threshold, contribution)


def soft_direction_votes(delta, directions, cfg):
    """Normalize each direction on ALL search points before the caller gates r.

    Blocks of 64 directions bound temporary memory. Sparse mode also compares
    every projection, but evaluates sigmoid only on retained support; its gather
    and indexing costs are deliberately included in the sigmoid timer.
    """
    dtype = np.dtype(cfg.voting_dtype)
    points = np.ascontiguousarray(delta, dtype=dtype)
    vectors = np.ascontiguousarray(directions, dtype=dtype)
    votes = np.zeros(len(points), dtype=dtype)
    timings = dict.fromkeys(("projection", "quantile", "sigmoid", "normalization", "accumulation"), 0.0)
    evaluated = 0
    for start in range(0, len(vectors), 64):
        stage = perf_counter()
        projections = np.ascontiguousarray(vectors[start:start + 64] @ points.T)
        if not np.isfinite(projections).all():
            raise ValueError("Nonfinite projection; check coordinates or use float64")
        timings["projection"] += perf_counter() - stage
        stage = perf_counter()
        threshold = (np.quantile(projections, 0.99, axis=1).astype(dtype)
                     if cfg.voting_mode == "soft_dense" else linear_quantile_99(projections))
        timings["quantile"] += perf_counter() - stage
        stage = perf_counter()
        if cfg.voting_mode == "soft_sparse":
            rows, columns = np.nonzero(projections >= threshold[:, None] - cfg.soft_alpha * cfg.soft_tau_mm)
            weights = (projections[rows, columns] - threshold[rows]) / cfg.soft_tau_mm
            expit(weights, out=weights)
            evaluated += len(weights)
            timings["sigmoid"] += perf_counter() - stage
            stage = perf_counter()
            mass = np.bincount(rows, weights=weights, minlength=len(threshold)).astype(dtype)
            weights /= mass[rows]
            timings["normalization"] += perf_counter() - stage
            stage = perf_counter()
            votes += np.bincount(columns, weights=weights, minlength=len(points)).astype(dtype)
        else:
            projections -= threshold[:, None]
            projections /= cfg.soft_tau_mm
            expit(projections, out=projections)
            evaluated += projections.size
            timings["sigmoid"] += perf_counter() - stage
            stage = perf_counter()
            projections /= projections.sum(axis=1, keepdims=True)
            timings["normalization"] += perf_counter() - stage
            stage = perf_counter()
            votes += projections.sum(axis=0)
        timings["accumulation"] += perf_counter() - stage
    return votes, timings, evaluated, len(vectors) * len(points)
