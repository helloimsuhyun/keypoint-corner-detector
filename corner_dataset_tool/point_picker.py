"""Pick *existing* points in screen pixels using current Open3D camera matrices.

Unlike reading a rendered depth pixel, this selection is independent of the ROI
wireframe, GT markers and rendering-depth alignment.  No Open3D import needed;
therefore the geometry can be unit-tested without a graphics session.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PickResult:
    """Index refers to the supplied point array (not the full raw scene)."""
    index: int
    distance_px: float
    depth_mm: float


def pick_projected_point(
    points_xyz: np.ndarray,
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    tolerance_px: float = 12.0,
    chunk_size: int = 100_000,
) -> PickResult | None:
    """Return the nearest selectable visible-screen point, or None.

    Camera convention: coordinates in the scene are projected using
    projection @ view; screen origin is TOP LEFT. Open3D's view camera looks
    toward -Z, so -view_z is used to break screen-distance ties in favour of
    the frontmost point. Only finite, in-frustum points are considered.

    In dense areas, candidates within a 2 px distance band from the closest
    projected point compete by depth. This avoids selecting a farther point
    that projects to approximately the same screen position.

    Note that this is *point-sprite* selection, not full mesh occlusion testing.
    """
    pts = np.asarray(points_xyz)
    V = np.asarray(view_matrix, dtype=np.float64)
    P = np.asarray(projection_matrix, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must be Nx3")
    if V.shape != (4, 4) or P.shape != (4, 4):
        raise ValueError("view and projection must be 4x4")
    if width <= 0 or height <= 0 or tolerance_px <= 0 or chunk_size < 1:
        raise ValueError("invalid viewport, tolerance or chunk size")

    VP = P @ V
    candidates: list[tuple[float, float, int]] = []
    r2 = float(tolerance_px) ** 2
    for start in range(0, len(pts), chunk_size):
        block = np.asarray(pts[start:start + chunk_size], dtype=np.float64)
        n = len(block)
        hom = np.empty((n, 4), dtype=np.float64)
        hom[:, :3] = block
        hom[:, 3] = 1.0

        clip = hom @ VP.T
        view = hom @ V.T
        w = clip[:, 3]
        front_depth = -view[:, 2]
        # clip.w>0 works for both perspective and orthographic projections.
        valid = np.isfinite(clip).all(axis=1) & np.isfinite(front_depth)
        valid &= (w > 1.0e-9) & (front_depth > 0)
        if not np.any(valid):
            continue
        valid_ix = np.flatnonzero(valid)
        ndc = clip[valid_ix, :3] / w[valid_ix, None]
        visible = (np.abs(ndc[:, 0]) <= 1.0) & (np.abs(ndc[:, 1]) <= 1.0)
        # OpenGL -1..1, Vulkan 0..1: accepting -1..1 covers both.
        visible &= (ndc[:, 2] >= -1.001) & (ndc[:, 2] <= 1.001)
        if not np.any(visible):
            continue
        local_ix = valid_ix[visible]
        ndc = ndc[visible]
        sx = (ndc[:, 0] + 1.0) * width * 0.5
        sy = (1.0 - ndc[:, 1]) * height * 0.5
        d2 = (sx - x) ** 2 + (sy - y) ** 2
        close = np.flatnonzero(d2 <= r2)
        if not len(close):
            continue
        # Keep all points within tolerance rather than only closest of each
        # chunk, to make the result independent of chunk size.
        candidates.extend(
            (float(d2[j]), float(front_depth[local_ix[j]]), int(start + local_ix[j]))
            for j in close
        )

    if not candidates:
        return None
    d_min = min(z[0] for z in candidates)
    # First favor pixel proximity; among almost-identical projections choose
    # nearer camera depth. A 2px band handles near-overlapping points.
    allowance = (np.sqrt(d_min) + 2.0) ** 2
    nearby = [c for c in candidates if c[0] <= allowance]
    d2, depth, index = min(nearby, key=lambda c: (c[1], c[0], c[2]))
    return PickResult(index=index, distance_px=float(np.sqrt(d2)), depth_mm=depth)
