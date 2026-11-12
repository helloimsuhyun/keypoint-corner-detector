"""Query-local RANSAC-voxel support patches; fitted models remain immutable."""

from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .plane_projection import make_tangent_basis


def _component_labels(points, radius):
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points))).tocsr()
    return connected_components(graph, directed=False)


def projection_centroid_q10(points_uv):
    """Unweighted cosine Q10 about the original-UV centroid direction from query (0,0)."""
    uv = np.asarray(points_uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError("Projection Q10 requires finite (N,2) UV coordinates")
    lengths = np.linalg.norm(uv, axis=1)
    nonzero = lengths > 1e-7
    centroid = uv.mean(axis=0) if len(uv) else np.zeros(2)
    result = {"centroid_uv_mm": centroid.copy(), "direction_uv": None, "q10": None,
              "nonzero_mask": nonzero, "cosines": np.empty(0), "reason": None}
    if np.count_nonzero(nonzero) < 3:
        result["reason"] = "insufficient_nonzero_directions"
        return result
    norm = np.linalg.norm(centroid)
    if not np.isfinite(norm) or norm < 1e-8:
        result["reason"] = "undefined_centroid_direction"
        return result
    direction = centroid / norm
    cosines = np.clip((uv[nonzero] / lengths[nonzero, None]) @ direction, -1., 1.)
    result.update(direction_uv=direction, cosines=cosines, q10=float(np.quantile(cosines, .1)))
    return result


def reconstruct_plane_support(points_mm, query_mm, planes, cfg, *, point_tree=None, projection_only=False):
    """Recollect overlapping memberships, then hull one local connected patch per plane."""
    started = perf_counter()
    cfg.validate()
    if not isinstance(projection_only, (bool, np.bool_)):
        raise ValueError("projection_only must be boolean")
    points = np.asarray(points_mm, dtype=np.float64)
    query = np.asarray(query_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("RANSAC voxel points must be finite with shape (N, 3)")
    if query.shape != (3,) or not np.isfinite(query).all():
        raise ValueError("Coarse query must be a finite 3-vector")
    tree = cKDTree(points) if point_tree is None else point_tree
    radius = cfg.padded_support_radius_mm
    roi_ids = np.asarray(sorted(tree.query_ball_point(query, radius)), dtype=np.int64)
    roi_ids = roi_ids[np.linalg.norm(points[roi_ids] - query, axis=1) <= radius]
    patches = []
    for plane_id, plane in enumerate(planes):
        normal = np.asarray(plane["normal"], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if normal.shape != (3,) or not np.isfinite(normal).all() or norm <= 1e-12:
            raise ValueError("Plane normal must be a finite nonzero 3-vector")
        offset = float(plane["offset_mm"]) / norm
        if not np.isfinite(offset):
            raise ValueError("Plane offset must be finite")
        normal = normal / norm
        u, v, _ = make_tangent_basis(normal)
        origin = query - (normal @ query + offset) * normal
        residual = np.abs(points[roi_ids] @ normal + offset)
        ids = roi_ids[residual < cfg.inlier_threshold_mm]
        patch = {
            "plane_id": plane_id, "normal": normal, "offset_mm": offset,
            "origin_mm": origin, "basis_u": u, "basis_v": v,
            "recollected_ransac_indices": ids.copy(),
            "support_ransac_indices": np.empty(0, dtype=np.int64),
            "support_indices": np.empty(0, dtype=np.int64),
            "support_count": 0, "hull_uv": np.empty((0, 2)),
            "hull_vertices_mm": np.empty((0, 3)), "hull_equations": np.empty((0, 3)),
            "hull_area_mm2": 0., "valid": False, "status": "UNKNOWN",
            "failure_reason": None, "component_count": 0, "component_sizes": [],
            "projection_uv": np.empty((0, 2)), "projection_computed": False,
            "component_core_counts": [], "selected_component": None,
            "connectivity_uncertain": False,
        }
        patches.append(patch)
        if not len(ids):
            patch["failure_reason"] = "no_plane_support"
            continue
        count, labels = _component_labels(points[ids], cfg.plane_connect_radius_mm)
        components = [ids[labels == label] for label in range(count)]
        core_ids = np.asarray(plane["inlier_ransac_indices"], dtype=np.int64)
        overlap = [len(np.intersect1d(component, core_ids)) for component in components]
        distances = [float(np.min(np.linalg.norm(points[component] - query, axis=1)))
                     for component in components]
        anchored = [label for label in range(count) if overlap[label]]
        selected = min(anchored or range(count), key=lambda label: (
            distances[label], -overlap[label], -len(components[label]), label))
        ids = components[selected]
        uncertain = not anchored
        patch.update(component_count=count, component_sizes=[len(c) for c in components],
                     component_core_counts=overlap, selected_component=selected,
                     connectivity_uncertain=uncertain, support_ransac_indices=ids.copy(),
                     support_indices=ids.copy(), support_count=len(ids))
        if len(ids) < 3:
            patch["failure_reason"] = "insufficient_connected_support"
            continue
        uv = np.column_stack(((points[ids] - origin) @ u, (points[ids] - origin) @ v))
        patch.update(projection_uv=uv.copy(), projection_computed=True)
        if projection_only:
            patch.update(valid=not uncertain, status="UNKNOWN" if uncertain else "VALID",
                         failure_reason="ambiguous_local_connectivity" if uncertain else None)
            continue
        singular = np.linalg.svd(uv - uv.mean(axis=0), compute_uv=False)
        if singular[1] <= 1e-8 * max(1., singular[0]):
            patch["failure_reason"] = "collinear_plane_support"
            continue
        try:
            hull = ConvexHull(uv)
        except QhullError:
            patch["failure_reason"] = "convex_hull_failed"
            continue
        hull_uv = uv[hull.vertices].copy()
        patch.update(hull_uv=hull_uv,
                     hull_vertices_mm=origin + hull_uv[:, :1] * u + hull_uv[:, 1:] * v,
                     hull_equations=hull.equations.copy(), hull_area_mm2=float(hull.volume),
                     valid=not uncertain, status="UNKNOWN" if uncertain else "VALID",
                     failure_reason="ambiguous_local_connectivity" if uncertain else None)
    return {"plane_patches": patches, "support_roi_ransac_indices": roi_ids,
            "support_roi_points_mm": points[roi_ids].copy(), "support_radius_mm": radius,
            "elapsed_s": perf_counter() - started}
