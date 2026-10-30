"""Deterministic farthest-point query sampling."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .common import PointCloudFrame
from .query import QuerySelection, _source_points, queries_from_indices


def farthest_point_sampling(
    points_mm: np.ndarray,
    count: int,
    *,
    start_index: int | None = None,
    min_spacing_mm: float | None = None,
    candidate_indices: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Return deterministic FPS indices into ``points_mm``.

    The default seed is the point nearest the candidate centroid. Selection
    stops early when the farthest remaining candidate is closer than
    ``min_spacing_mm`` to the selected set.
    """

    points = _source_points(np.asarray(points_mm))
    if count < 1:
        raise ValueError("count must be >= 1")
    if min_spacing_mm is not None and (
        not np.isfinite(min_spacing_mm) or min_spacing_mm < 0
    ):
        raise ValueError("min_spacing_mm must be finite and >= 0")
    if candidate_indices is None:
        candidates = np.arange(len(points), dtype=np.int64)
    else:
        candidates = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
        if len(np.unique(candidates)) != len(candidates):
            raise ValueError("candidate_indices must be unique")
        if np.any(candidates < 0) or np.any(candidates >= len(points)):
            raise IndexError("candidate_indices contains an out-of-range index")
    if not len(candidates):
        return np.empty(0, dtype=np.int64)

    candidate_points = points[candidates].astype(np.float64, copy=False)
    if start_index is None:
        centroid = np.mean(candidate_points, axis=0)
        local = int(np.argmin(np.sum((candidate_points - centroid) ** 2, axis=1)))
    else:
        matches = np.flatnonzero(candidates == int(start_index))
        if not len(matches):
            raise ValueError("start_index must belong to candidate_indices")
        local = int(matches[0])

    target_count = min(int(count), len(candidates))
    selected_local: list[int] = []
    minimum_distance2 = np.full(len(candidates), np.inf, dtype=np.float64)
    spacing2 = None if min_spacing_mm is None else float(min_spacing_mm) ** 2

    while len(selected_local) < target_count:
        if selected_local and spacing2 is not None and minimum_distance2[local] < spacing2:
            break
        selected_local.append(local)
        delta = candidate_points - candidate_points[local]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        minimum_distance2 = np.minimum(minimum_distance2, distance2)
        minimum_distance2[np.asarray(selected_local, dtype=np.int64)] = -1.0
        if len(selected_local) >= target_count:
            break
        local = int(np.argmax(minimum_distance2))
        if minimum_distance2[local] <= 0.0:
            break
    return candidates[np.asarray(selected_local, dtype=np.int64)]


def select_fps_queries(
    source: PointCloudFrame | np.ndarray,
    count: int,
    *,
    start_index: int | None = None,
    min_spacing_mm: float | None = None,
    candidate_indices: Sequence[int] | np.ndarray | None = None,
) -> QuerySelection:
    """Select FPS queries and return indices, coordinates, and source pixels."""

    points = _source_points(source)
    indices = farthest_point_sampling(
        points,
        count,
        start_index=start_index,
        min_spacing_mm=min_spacing_mm,
        candidate_indices=candidate_indices,
    )
    selection = queries_from_indices(
        source,
        indices,
        method="farthest-point-sampling",
        metadata={
            "requested_count": int(count),
            "actual_count": int(len(indices)),
            "min_spacing_mm": None if min_spacing_mm is None else float(min_spacing_mm),
            "start_index": None if not len(indices) else int(indices[0]),
        },
    )
    return selection
