"""Origin-constrained unit-sphere RANSAC and per-plane cleaned angular-gap evidence.

Numerical reference: corner_lab_open3d_cone.py; no GUI dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SphericalPlane:
    normal: np.ndarray                # unit normal, plane offset d == 0 by construction
    inlier_mask: np.ndarray           # N bool; may overlap with OTHER planes
    projected_directions: np.ndarray  # M x 3 on great circle; one per inlier
    angles_deg: np.ndarray           # M in [0, 360), within this plane
    residual_rms_deg: float
    newly_explained_count: int



@dataclass(frozen=True)
class SphericalPlaneAnalysis:
    directions: np.ndarray       # N x 3, derived from same downsampled cloud
    planes: tuple[SphericalPlane, ...]
    membership_count: np.ndarray  # N, may be 0, 1, 2 or 3
    requested_k: int
    tolerance_deg: float



def _sphere_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal basis of a zero-offset plane."""
    n = np.asarray(normal, dtype=np.float64)
    ref = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = ref - np.dot(ref, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2



def _spherical_plane_residual_deg(u: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Angle to GREAT CIRCLE, rather than a Euclidean offset in millimetres."""
    return np.rad2deg(np.arcsin(np.clip(np.abs(u @ normal), 0.0, 1.0)))



def _refit_origin_plane(u: np.ndarray, seed: np.ndarray,
                        tolerance_deg: float, max_iters: int = 8) -> np.ndarray:
    """Robust Huber IRLS total least squares, with NO mean subtraction."""
    n = np.asarray(seed, dtype=np.float64).copy()
    huber_deg = max(0.5, 0.50 * float(tolerance_deg))
    for _ in range(max_iters):
        r = _spherical_plane_residual_deg(u, n)
        support = r <= tolerance_deg
        if support.sum() < 3:
            break
        r_s = r[support]
        weights = np.minimum(1.0, huber_deg / np.maximum(r_s, 1e-9))
        v = u[support]
        # Homogeneous constraint: smallest eigenvector of sum w*u*u.T.
        scatter = (v * weights[:, None]).T @ v
        _, eigenvectors = np.linalg.eigh(scatter)
        n_new = eigenvectors[:, 0]
        if np.dot(n_new, n) < 0:
            n_new = -n_new
        change_deg = np.rad2deg(np.arccos(np.clip(abs(n_new @ n), 0, 1)))
        n = n_new
        if change_deg < 1e-5:
            break
    return n



def segment_origin_spherical_planes(directions: np.ndarray, k: int = 3,
                                    tolerance_deg: float = 7.0,
                                    iterations: int = 300,
                                    min_support: int = 12,
                                    min_fraction: float = 0.06,
                                    seed: int = 20261110) -> SphericalPlaneAnalysis:
    """Sequential robust Great-Circle RANSAC with multiple memberships.

    Stage A: each RANSAC round prioritizes currently UNEXPLAINED points.
    Stage B: refit each accepted plane with ALL matching directions.
    Stage C: classify ALL directions independently against EVERY plane.
    Points near plane intersections belong to multiple planes, not just one.
    """
    u = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(u).all():
        raise ValueError('Unit directions contain non-finite values')
    norms = np.linalg.norm(u, axis=1)
    if len(u) < 3 or np.any(norms < 1e-10):
        raise ValueError('Need at least three valid nonzero directions')
    u = u / norms[:, None]
    if k not in (1, 2, 3):
        raise ValueError('k must be 1, 2, or 3')
    if not (0 < tolerance_deg < 45) or iterations < 10:
        raise ValueError('Invalid angular tolerance or RANSAC iterations')

    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    explained = np.zeros(len(u), dtype=bool)
    minimum = max(int(min_support), int(np.ceil(min_fraction * len(u))))
    max_abs_cos = float(np.sin(np.deg2rad(tolerance_deg)))
    # Reject almost duplicate planes, but do NOT reject point-level overlap.
    duplicate_cos = float(np.cos(np.deg2rad(max(12.0, 1.5 * tolerance_deg))))

    for plane_index in range(k):
        available = np.flatnonzero(~explained)
        if len(available) < minimum:
            break
        # Sample representative residual directions to bound per-frame work.
        if len(available) > 4000:
            sample_ids = rng.choice(available, size=4000, replace=False)
        else:
            sample_ids = available
        sample_u = u[sample_ids]
        best_n, best_count, best_cost = None, -1, np.inf

        for _ in range(iterations):
            i, j = rng.choice(len(sample_u), size=2, replace=False)
            n = np.cross(sample_u[i], sample_u[j])
            mag = np.linalg.norm(n)
            if mag < 0.16:  # Avoid near-collinear minimal samples.
                continue
            n /= mag
            if any(abs(n @ old) > duplicate_cos for old in accepted):
                continue
            distances = np.abs(sample_u @ n)
            inliers = distances <= max_abs_cos
            cnt = int(inliers.sum())
            # Simple robust tie-break for similarly supported models.
            cost = float(np.sum(np.minimum(distances, max_abs_cos)**2))
            if cnt > best_count or (cnt == best_count and cost < best_cost):
                best_n, best_count, best_cost = n.copy(), cnt, cost

        if best_n is None:
            break
        # Refitting must use the complete neighborhood, not just RANSAC's sample.
        refined = _refit_origin_plane(u, best_n, tolerance_deg)
        if any(abs(refined @ old) > duplicate_cos for old in accepted):
            break
        residual = _spherical_plane_residual_deg(u, refined)
        inliers = residual <= tolerance_deg
        newly_explained = inliers & ~explained
        if newly_explained.sum() < minimum:
            break
        accepted.append(refined)
        explained |= inliers

    # Final full-set membership: independent threshold for each plane.
    # Even an intersection point already claimed by one plane is retested
    # against all other planes and can be included in every applicable plane.
    planes: list[SphericalPlane] = []
    membership = np.zeros(len(u), dtype=np.uint8)
    covered_final = np.zeros(len(u), dtype=bool)
    for n in accepted:
        # Refit with all current inliers; still constrained to d=0.
        n = _refit_origin_plane(u, n, tolerance_deg)
        # Fix normal sign for repeatable local azimuth histograms.
        if n[int(np.argmax(np.abs(n)))] < 0:
            n = -n
        r = _spherical_plane_residual_deg(u, n)
        mask = r <= tolerance_deg
        membership += mask.astype(np.uint8)
        newly_explained_count = int(np.count_nonzero(mask & ~covered_final))
        covered_final |= mask
        selected = u[mask]
        projected = selected - (selected @ n)[:, None] * n[None, :]
        magnitudes = np.linalg.norm(projected, axis=1)
        usable = magnitudes > 1e-10  # Always true for r < 45 degrees.
        projected = projected[usable] / magnitudes[usable, None]
        e1, e2 = _sphere_plane_basis(n)
        angle = np.mod(np.rad2deg(np.arctan2(projected @ e2, projected @ e1)), 360.)
        angle[angle >= 360.] = 0.  # Guard floating-point rounding at the seam.
        rms = float(np.sqrt(np.mean(r[mask]**2))) if mask.any() else float('nan')
        planes.append(SphericalPlane(n, mask, projected, angle, rms, newly_explained_count))
    return SphericalPlaneAnalysis(u, tuple(planes), membership, k, float(tolerance_deg))



@dataclass(frozen=True)
class PlaneGapTest:
    plane_index: int
    raw_count: int
    kept_count: int
    kept_mask: np.ndarray  # original plane.angles_deg order; False = rejected angular noise
    discarded_count: int
    component_count: int
    max_gap_deg: float
    observed_span_deg: float  # 360 - max_gap after removing angular outliers
    arc_start_deg: float | None
    arc_end_deg: float | None
    pcl_boundary_like: bool  # classical >90 degree angular gap, NOT corner classification
    edge_like: bool          # experimental edge-vs-corner decision via span
    reliable: bool          # a nonempty cleaned arc exists; no count/fraction gate
    raw_span_deg: float = 0.0
    tail_removed_count: int = 0
    tail_trimmed_bins: int = 0
    density_threshold_count: int = 0
    density_reference_count: float = 0.0

@dataclass(frozen=True)
class PlaneGapValidation:
    status: str              # CORNER / REJECT / UNKNOWN
    tests: tuple[PlaneGapTest, ...]
    max_link_gap_deg: float
    edge_span_deg: float

def _dominant_angular_component(angles_deg: np.ndarray,
                                max_link_gap_deg: float = 20.0) -> tuple[np.ndarray, int]:
    """Keep largest connected component on a circular angular coordinate.

    Sorted neighbors are connected if their *circular angular separation* is
    <= max_link_gap_deg. Disconnected small clusters (e.g. a few points opposite
    the actual surface arc) cannot change the measured main-arc span.
    This is angular connectivity, NOT Euclidean point-cloud connectivity.
    """
    a = np.mod(np.asarray(angles_deg, dtype=np.float64).reshape(-1), 360.)
    if not 0. < max_link_gap_deg < 180.:
        raise ValueError('Angular connection gap must be between 0 and 180 degrees')
    n = len(a)
    if n == 0:
        return np.zeros(0, dtype=bool), 0
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite angular direction')
    order = np.argsort(a, kind='stable')
    theta = a[order]
    # cyclic gap after each sorted angle, including the 360/0 seam
    gaps = np.diff(np.r_[theta, theta[0] + 360.])
    cuts = np.flatnonzero(gaps > max_link_gap_deg + 1e-10)
    if len(cuts) == 0:
        return np.ones(n, dtype=bool), 1
    # Linearize just after one cut, then split at all other cuts.
    linear = (np.arange(n, dtype=int) + int(cuts[0]) + 1) % n
    breaks = np.flatnonzero(gaps[linear[:-1]] > max_link_gap_deg) + 1
    groups = np.split(linear, breaks)
    # Prefer larger group; equal size chooses compact span, then smallest angle.
    def rank(group):
        spread = float(np.mod(theta[group[-1]] - theta[group[0]], 360.))
        return (-len(group), spread, float(theta[group[0]]))
    winner = min(groups, key=rank)
    keep = np.zeros(n, dtype=bool)
    keep[order[winner]] = True
    return keep, len(groups)

def trim_sparse_arc_ends(angles_deg: np.ndarray, dominant_mask: np.ndarray,
                         bin_width_deg: float = 10.0,
                         density_fraction: float = 0.50,
                         minimum_bin_count: int = 2,
                         enabled: bool = True,
                         ) -> tuple[np.ndarray, float, int, int, int, float]:
    """Lightly trim low-count angular bins ONLY from both ends of dominant arc.

    Returns (kept_mask, raw_span_deg, trimmed_bins, removed_points,
             count_threshold, reference_median_count). The circular 0/360
    seam is handled by unwrapping relative to the largest angular gap.
    Internal sparse bins are deliberately retained; only tails are trimmed.
    """
    angles = np.mod(np.asarray(angles_deg, dtype=np.float64).reshape(-1), 360.)
    dominant = np.asarray(dominant_mask, dtype=bool).reshape(-1)
    if angles.size != dominant.size:
        raise ValueError('Angular points and component mask must match')
    if not (0 < bin_width_deg <= 90 and 0 <= density_fraction <= 1
            and minimum_bin_count >= 1):
        raise ValueError('Invalid angular tail filter parameters')
    if not dominant.any():
        return dominant.copy(), 0.0, 0, 0, minimum_bin_count, 0.0
    ids = np.flatnonzero(dominant)
    ordered = np.sort(angles[ids])
    circular_gaps = np.diff(np.r_[ordered, ordered[0] + 360.])
    gap_idx = int(np.argmax(circular_gaps))
    start = float(ordered[(gap_idx + 1) % len(ordered)])
    raw_span = float(np.clip(360. - circular_gaps[gap_idx], 0., 360.))
    # Angle relative to the observed arc's starting endpoint.
    offsets = np.mod(angles[ids] - start, 360.)
    # Avoid placing the very last sample into a phantom bin if raw_span is
    # exactly divisible by the requested bin width.
    unwrapped = start + np.minimum(offsets, max(0., raw_span - 1e-8))
    # Anchor bins to a fixed global angular grid, not the first sparse point.
    # Otherwise that first point can accidentally share a bin with dense data.
    base_bin = int(np.floor(start / float(bin_width_deg)))
    bins = (np.floor(unwrapped / float(bin_width_deg)).astype(np.int64)
            - base_bin)
    counts = np.bincount(bins, minlength=int(bins.max()) + 1)
    occupied = counts[counts > 0]
    ref = float(np.median(occupied)) if len(occupied) else 0.0
    threshold = max(int(minimum_bin_count),
                    int(np.ceil(float(density_fraction) * ref)))
    if not enabled or len(counts) <= 1:
        return dominant.copy(), raw_span, 0, 0, threshold, ref
    # Cut only the two tails. Never discard a low-density interior bin.
    left, right = 0, len(counts) - 1
    while left <= right and counts[left] < threshold:
        left += 1
    while right >= left and counts[right] < threshold:
        right -= 1
    cleaned = dominant.copy()
    if left > right:  # No bin has reliable support: status becomes UNKNOWN.
        cleaned[ids] = False
    else:
        cleaned[ids] = (bins >= left) & (bins <= right)
    removed = int(np.count_nonzero(dominant & ~cleaned))
    trimmed_bins = int(len(counts) if left > right else left + (len(counts) - 1 - right))
    return cleaned, raw_span, trimmed_bins, removed, threshold, ref

def evaluate_plane_max_gap(analysis: SphericalPlaneAnalysis,
                           max_link_gap_deg: float = 20.0,
                           edge_span_deg: float = 160.0,
                           tail_filter_enabled: bool = True,
                           tail_bin_width_deg: float = 10.0,
                           tail_density_fraction: float = 0.50,
                           tail_min_bin_count: int = 2) -> PlaneGapValidation:
    """Measure largest circular gap and its complementary span per spherical plane.

    Find the largest linked angular component first. Then, optionally remove
    only its very sparse END bins, using the median of occupied bins as the
    reference. Interior bins are never dropped by this gentle tail filter.
    Every nonempty cleaned arc is classified by span, without minimum retained
    count or fraction gates. A completely empty cleaned arc remains UNKNOWN.
    The PCL-style gap flag uses the dominant component BEFORE tail trimming,
    preserving previous diagnostic behavior. Rejecting EDGE-LIKE uses the
    cleaned span >= edge_span_deg (OUR experimental criterion).
    """
    if not (0. < max_link_gap_deg < 180. and 0. < edge_span_deg <= 360.):
        raise ValueError('Invalid maximum angular gap parameters')
    tests = []
    for i, plane in enumerate(analysis.planes):
        angles = np.mod(np.asarray(plane.angles_deg, dtype=np.float64), 360.)
        component_keep, components = _dominant_angular_component(angles, max_link_gap_deg)
        keep, raw_span, tail_bins, tail_points, density_min, density_ref = \
            trim_sparse_arc_ends(angles, component_keep,
                                 bin_width_deg=tail_bin_width_deg,
                                 density_fraction=tail_density_fraction,
                                 minimum_bin_count=tail_min_bin_count,
                                 enabled=tail_filter_enabled)
        kept = int(keep.sum())
        reliable = kept > 0
        max_gap = 360.0
        start = end = None
        span = 0.0
        if kept:
            order = np.sort(angles[keep])
            gaps = np.diff(np.r_[order, order[0] + 360.])
            gap_id = int(np.argmax(gaps))
            max_gap = float(gaps[gap_id])
            span = float(np.clip(360.0 - max_gap, 0.0, 360.0))
            start = float(order[(gap_id + 1) % kept])
            end = float(order[gap_id])
        tests.append(PlaneGapTest(
            i, len(angles), kept, keep, len(angles) - kept, components,
            max_gap, span, start, end, bool(360. - raw_span > 90.),
            bool(reliable and span >= edge_span_deg), bool(reliable),
            float(raw_span), int(tail_points), int(tail_bins),
            int(density_min), float(density_ref)))
    if not tests or any(not t.reliable for t in tests):
        status = 'REJECT' if any(t.edge_like for t in tests) else 'UNKNOWN'
    else:
        status = 'REJECT' if any(t.edge_like for t in tests) else 'CORNER'
    return PlaneGapValidation(status, tuple(tests), max_link_gap_deg, edge_span_deg)
