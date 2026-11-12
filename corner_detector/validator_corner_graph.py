"""Endpoint-based Corner candidates and graph integration; no structural validation."""

from itertools import combinations
from time import perf_counter

import numpy as np


def generate_corners(edges, min_angle_deg=20., corner_radius_mm=5.):
    """Return unverified Corners from finite 3-D Edges, preserving input geometry.

    Endpoint IDs are 0=start and 1=end, local to each Edge ID. Optional
    fit_tolerance_mm supplies the existing Edge fitting tolerance for skew lines;
    absent tolerances require an exact intersection up to numerical roundoff.
    """
    if not np.isfinite([min_angle_deg, corner_radius_mm]).all():
        raise ValueError("Corner angle and radius must be finite")
    if not 0 < min_angle_deg <= 90 or corner_radius_mm <= 0:
        raise ValueError("Require an angle in (0, 90] degrees and a positive radius")
    prepared, identifiers = [], set()
    for edge in edges:
        identifier = edge["edge_id"]
        if isinstance(identifier, (bool, np.bool_)) or not isinstance(identifier, (int, np.integer)):
            raise ValueError("Edge IDs must be integers")
        if identifier in identifiers:
            raise ValueError("Edge IDs must be unique")
        identifiers.add(identifier)
        endpoints = np.asarray([edge["start_mm"], edge["end_mm"]], dtype=np.float64)
        tolerance = float(edge.get("fit_tolerance_mm", 0.))
        if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
            raise ValueError("Edge endpoints must be finite 3-vectors")
        length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        if not np.isfinite(length) or length <= 1e-10:
            raise ValueError("Edges must have positive finite length")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Edge fitting tolerance must be finite and nonnegative")
        if edge.get("valid", True):
            prepared.append((int(identifier), endpoints, (endpoints[1] - endpoints[0]) / length, tolerance))

    candidates = []
    cosine = np.cos(np.deg2rad(min_angle_deg))
    for left, right in combinations(sorted(prepared, key=lambda item: item[0]), 2):
        i, a, u, error_i = left
        j, b, v, error_j = right
        if abs(u @ v) > cosine + 1e-12:
            continue
        parameters, _, rank, _ = np.linalg.lstsq(np.column_stack((u, -v)), b[0] - a[0], rcond=None)
        if rank < 2:
            continue
        p, q = a[0] + parameters[0] * u, b[0] + parameters[1] * v
        residual = float(np.linalg.norm(p - q))
        if not np.isfinite(residual) or residual > max(error_i, error_j) + 1e-8:
            continue
        position = p + .5 * (q - p)
        distances = [np.linalg.norm(endpoints - position, axis=1) for endpoints in (a, b)]
        ends = [int(np.argmin(values)) for values in distances]
        gap = float(max(values[end] for values, end in zip(distances, ends)))
        if gap > corner_radius_mm:
            continue
        candidates.append({"position": position, "edge_ids": [i, j],
                           "endpoint_ids": {i: ends[0], j: ends[1]},
                           "intersection_residual": residual, "endpoint_gap": gap})

    groups = [[candidate] for candidate in candidates]
    while True:
        for i, j in combinations(range(len(groups)), 2):
            left, right = groups[i], groups[j]
            left_ends = {edge: end for item in left for edge, end in item["endpoint_ids"].items()}
            right_ends = {edge: end for item in right for edge, end in item["endpoint_ids"].items()}
            shared = left_ends.keys() & right_ends.keys()
            if not shared or any(left_ends[edge] != right_ends[edge] for edge in shared):
                continue
            if any(np.linalg.norm(a["position"] - b["position"]) > corner_radius_mm
                   for a in left for b in right):
                continue
            groups[i] = left + right
            groups.pop(j)
            break
        else:
            break

    corners = []
    for group in groups:
        representative = min(group, key=lambda item: (item["endpoint_gap"],
                                                      item["intersection_residual"], item["edge_ids"]))
        ends = {edge: end for item in group for edge, end in item["endpoint_ids"].items()}
        corners.append({**representative, "position": representative["position"].copy(),
                        "edge_ids": sorted(ends), "endpoint_ids": dict(sorted(ends.items()))})
    return corners


def empty_corner_generation(query_mm):
    return {"verified": False, "validation_status": "NOT_COMPUTED",
            "geometry_computed": False, "query_position_mm": np.asarray(query_mm).copy(),
            "corners": [], "plane_patches": [], "edges": [], "edge_candidates": [],
            "plane_adjacency": [], "failure_reason": "geometry_disabled",
            "stage_elapsed_s": {}, "point_support_checked": False}


def reconstruct_validator_corners(graph, cfg):
    """Adapt existing finite Edges to the independent Corner generator."""
    started = perf_counter()
    corners = generate_corners(graph["edges"], cfg.min_angle_deg, cfg.corner_radius_mm)
    return {"corners": corners, "geometry_computed": True, "verified": False,
            "validation_status": "CANDIDATES" if corners else "NO_CANDIDATES",
            "failure_reason": None if corners else "no_endpoint_corner_candidates",
            "elapsed_s": perf_counter() - started}
