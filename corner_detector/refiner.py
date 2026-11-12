"""Graph-local robust geometry and discrete Q10 post-detection refinement.

Graph refinement consumes the immutable output of
``analyze_plane_edge_graph`` and never recomputes clustering, boundary evidence,
or Hough voting. Q10 relocation separately supports optional spherical query CC on an
immutable analysis cloud.
All distances are millimetres.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import time

import numpy as np
from scipy.optimize import least_squares

from .connectivity import radius_components as _components


Q10_DIRECTION_MODES = ("unit_mean", "unit_median", "centroid")


@dataclass(frozen=True)
class HoughEdgeRefinerConfig:
    enabled: bool = True
    robust_loss: str = "huber"
    huber_delta_mm: float = 10.0
    plane_confidence_weight: float = 1.0
    boundary_confidence_weight: float = 1.0
    max_nfev: int = 100
    corner_intersection_mm: float = 5.0
    corner_endpoint_gap_mm: float = 12.0
    corner_merge_mm: float = 10.0

    def validate(self):
        positive = (
            self.huber_delta_mm,
            self.plane_confidence_weight,
            self.boundary_confidence_weight,
            self.corner_intersection_mm,
            self.corner_endpoint_gap_mm,
            self.corner_merge_mm,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Refiner scales and confidence weights must be positive")
        if self.robust_loss not in ("huber", "soft_l1", "linear"):
            raise ValueError("robust_loss must be huber, soft_l1, or linear")
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be positive")


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError("Degenerate direction")
    return vector / length


def _plane_basis(normal):
    normal = _normalize(normal)
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = _normalize(np.cross(normal, axis))
    return u, np.cross(normal, u)


def _line_basis(direction):
    direction = _normalize(direction)
    first, second = _plane_basis(direction)
    return first, second


def _line_distances(points, point, direction):
    points = np.asarray(points, dtype=np.float64)
    direction = _normalize(direction)
    delta = points - np.asarray(point, dtype=np.float64)
    return np.linalg.norm(delta - (delta @ direction)[:, None] * direction, axis=1)


def _robust_cost(residuals, cfg):
    residuals = np.asarray(residuals, dtype=np.float64)
    delta = float(cfg.huber_delta_mm)
    if cfg.robust_loss == "linear":
        return float(0.5 * np.sum(residuals * residuals))
    if cfg.robust_loss == "soft_l1":
        scaled = residuals / delta
        return float(delta * delta * np.sum(np.sqrt(1.0 + scaled * scaled) - 1.0))
    absolute = np.abs(residuals)
    loss = np.where(
        absolute <= delta,
        0.5 * residuals * residuals,
        delta * (absolute - 0.5 * delta),
    )
    return float(np.sum(loss))


def _solve(residual_function, variable_count, cfg):
    initial = np.zeros(variable_count, dtype=np.float64)
    before = np.asarray(residual_function(initial), dtype=np.float64)
    solved = least_squares(
        residual_function,
        initial,
        loss=cfg.robust_loss,
        f_scale=cfg.huber_delta_mm,
        max_nfev=cfg.max_nfev,
    )
    after = np.asarray(residual_function(solved.x), dtype=np.float64)
    return solved, {
        "loss": cfg.robust_loss,
        "cost_before": _robust_cost(before, cfg),
        "cost_after": _robust_cost(after, cfg),
        "residual_count": int(len(after)),
        "iterations": int(solved.nfev),
        "success": bool(solved.success),
        "message": str(solved.message),
    }


def _refine_plane(points, plane, cfg, graph_id):
    support_ids = np.unique(np.asarray(plane["support_indices"], dtype=np.int64))
    support = points[support_ids]
    center0 = np.asarray(plane["center_mm"], dtype=np.float64)
    normal0 = _normalize(plane["normal"])
    tangent_u, tangent_v = _plane_basis(normal0)

    def decode(parameters):
        normal = _normalize(
            normal0 + parameters[0] * tangent_u + parameters[1] * tangent_v
        )
        center = center0 + parameters[2] * normal0
        return center, normal

    def residual(parameters):
        center, normal = decode(parameters)
        return ((support - center) @ normal) * np.sqrt(cfg.plane_confidence_weight)

    if len(support) >= 3:
        solved, optimization = _solve(residual, 3, cfg)
        center, normal = decode(solved.x)
    else:
        center, normal = center0.copy(), normal0.copy()
        raw = residual(np.zeros(3))
        optimization = {
            "loss": cfg.robust_loss,
            "cost_before": _robust_cost(raw, cfg),
            "cost_after": _robust_cost(raw, cfg),
            "residual_count": int(len(raw)), "iterations": 0,
            "success": False, "message": "fewer than three plane points",
        }
    signed = (support - center) @ normal
    rms = float(np.sqrt(np.mean(signed * signed))) if len(signed) else float("inf")
    confidence = cfg.plane_confidence_weight / (
        1.0 + (rms / cfg.huber_delta_mm) ** 2
    )
    u, v = _plane_basis(normal)
    return {
        "plane_id": int(plane["plane_id"]),
        "graph_id": int(graph_id),
        "support_indices": support_ids,
        "initial": {
            "center_mm": center0, "normal": normal0,
            "rmse_mm": float(plane.get("rmse_mm", np.sqrt(np.mean(residual(np.zeros(3)) ** 2)))),
        },
        "refined": {
            "center_mm": center, "normal": normal, "u": u, "v": v,
            "rmse_mm": rms,
        },
        "confidence": float(confidence),
        "optimization": optimization,
    }


def _graph_components(plane_ids, edges):
    adjacency = {int(plane_id): set() for plane_id in plane_ids}
    for edge in edges:
        ids = sorted(set(int(value) for value in edge.get("plane_ids", ())))
        for left, right in combinations(ids, 2):
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
    components = []
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        stack, component = [seed], []
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _edge_source_ids(edge, plane_id, proposal_by_id):
    selected = []
    proposal_ids = []
    for proposal_id in edge.get("member_proposal_ids", ()):
        proposal = proposal_by_id.get(int(proposal_id))
        if proposal is None or int(proposal["plane_id"]) != int(plane_id):
            continue
        proposal_ids.append(int(proposal_id))
        selected.append(np.asarray(proposal["source_2d_indices"], dtype=np.int64))
    if not selected:
        return np.empty(0, dtype=np.int64), proposal_ids
    return np.unique(np.concatenate(selected)), proposal_ids


def _boundary_weights_for_ids(plane, source_ids):
    projection_ids = np.asarray(plane["projection_indices"], dtype=np.int64)
    weights = np.asarray(plane["boundary_weights"], dtype=np.float64)
    lookup = {int(scene_id): float(weight) for scene_id, weight in zip(projection_ids, weights)}
    return np.asarray([max(lookup.get(int(scene_id), 0.0), 1e-6)
                       for scene_id in source_ids], dtype=np.float64)


def _line_from_point_direction(point, direction, support):
    point = np.asarray(point, dtype=np.float64)
    direction = _normalize(direction)
    support = np.asarray(support, dtype=np.float64)
    if len(support):
        coordinate = (support - point) @ direction
        lower, upper = float(np.min(coordinate)), float(np.max(coordinate))
        start, end = point + lower * direction, point + upper * direction
        center = 0.5 * (start + end)
        rms = float(np.sqrt(np.mean(_line_distances(support, center, direction) ** 2)))
    else:
        start = end = center = point.copy()
        lower = upper = rms = 0.0
    return {
        "center_mm": center, "direction": direction,
        "start_mm": start, "end_mm": end,
        "length_mm": float(upper - lower), "rmse_mm": rms,
    }


def _refine_boundary_observation(points, source_ids, boundary_weights, plane_refined,
                                 initial_edge, cfg, graph_id, source_edge_id,
                                 proposal_ids):
    center = np.asarray(plane_refined["refined"]["center_mm"], dtype=np.float64)
    u = np.asarray(plane_refined["refined"]["u"], dtype=np.float64)
    v = np.asarray(plane_refined["refined"]["v"], dtype=np.float64)
    support = points[source_ids]
    uv = np.column_stack(((support - center) @ u, (support - center) @ v))
    edge_center = np.asarray(initial_edge["center_mm"], dtype=np.float64)
    edge_direction = _normalize(initial_edge["direction"])
    center_uv = np.asarray([(edge_center - center) @ u, (edge_center - center) @ v])
    direction_uv = np.asarray([edge_direction @ u, edge_direction @ v])
    if np.linalg.norm(direction_uv) < 1e-10:
        direction_uv = np.asarray([1.0, 0.0])
    direction_uv = direction_uv / np.linalg.norm(direction_uv)
    normal0 = np.asarray([-direction_uv[1], direction_uv[0]])
    rho0 = float(normal0 @ center_uv)
    normalized_weights = boundary_weights / max(float(np.sum(boundary_weights)), 1e-12)

    def decode(parameters):
        angle = float(parameters[0])
        cosine, sine = np.cos(angle), np.sin(angle)
        normal = np.asarray([
            cosine * normal0[0] - sine * normal0[1],
            sine * normal0[0] + cosine * normal0[1],
        ])
        return normal, rho0 + float(parameters[1])

    def residual(parameters):
        normal, rho = decode(parameters)
        return ((uv @ normal - rho)
                * np.sqrt(normalized_weights * len(uv)
                          * cfg.boundary_confidence_weight))

    solved, optimization = _solve(residual, 2, cfg)
    normal2, rho = decode(solved.x)
    direction2 = np.asarray([normal2[1], -normal2[0]])
    point2 = rho * normal2
    point3 = center + point2[0] * u + point2[1] * v
    direction3 = _normalize(direction2[0] * u + direction2[1] * v)
    refined_line = _line_from_point_direction(point3, direction3, support)
    signed = uv @ normal2 - rho
    rms = float(np.sqrt(np.average(signed * signed, weights=boundary_weights)))
    confidence = cfg.boundary_confidence_weight * float(np.mean(boundary_weights)) / (
        1.0 + (rms / cfg.huber_delta_mm) ** 2
    )
    initial_normal = normal0
    return {
        "observation_id": -1,
        "graph_id": int(graph_id),
        "source_edge_id": int(source_edge_id),
        "plane_id": int(plane_refined["plane_id"]),
        "source_proposal_ids": proposal_ids,
        "boundary_support_indices": source_ids,
        "boundary_weights": boundary_weights,
        "initial_2d": {
            "normal": initial_normal, "rho_mm": rho0,
            "direction": direction_uv, "center_uv": center_uv,
        },
        "refined_2d": {
            "normal": normal2, "rho_mm": float(rho),
            "direction": direction2,
        },
        "restored_3d": refined_line,
        "confidence": float(confidence),
        "rmse_mm": rms,
        "optimization": optimization,
    }


def _plane_intersection(left, right, reference_center, support):
    n0 = _normalize(left["refined"]["normal"])
    n1 = _normalize(right["refined"]["normal"])
    direction = np.cross(n0, n1)
    if np.linalg.norm(direction) < 1e-8:
        return None
    direction = _normalize(direction)
    matrix = np.vstack((n0, n1))
    offsets = np.asarray([
        n0 @ left["refined"]["center_mm"],
        n1 @ right["refined"]["center_mm"],
    ])
    point = matrix.T @ np.linalg.solve(matrix @ matrix.T, offsets)
    point += float((np.asarray(reference_center) - point) @ direction) * direction
    return _line_from_point_direction(point, direction, support)


def _shared_edge_residual(parameters, point0, direction0, line_u, line_v,
                          plane_pair, observations, direction_scale):
    point = point0 + parameters[0] * line_u + parameters[1] * line_v
    direction = _normalize(
        direction0 + parameters[2] * line_u + parameters[3] * line_v
    )
    residuals = []
    for plane in plane_pair:
        normal = np.asarray(plane["refined"]["normal"], dtype=np.float64)
        center = np.asarray(plane["refined"]["center_mm"], dtype=np.float64)
        scale = np.sqrt(max(float(plane["confidence"]), 1e-12) / 2.0)
        residuals.extend((
            scale * float((point - center) @ normal),
            scale * direction_scale * float(direction @ normal),
        ))
    for observation in observations:
        plane = next(item for item in plane_pair
                     if item["plane_id"] == observation["plane_id"])
        center = np.asarray(plane["refined"]["center_mm"], dtype=np.float64)
        u = np.asarray(plane["refined"]["u"], dtype=np.float64)
        v = np.asarray(plane["refined"]["v"], dtype=np.float64)
        point_uv = np.asarray([(point - center) @ u, (point - center) @ v])
        direction_uv = np.asarray([direction @ u, direction @ v])
        magnitude = float(np.linalg.norm(direction_uv))
        if magnitude < 1e-12:
            residuals.extend(np.full(len(observation["boundary_support_indices"]), 1e3))
            continue
        direction_uv /= magnitude
        normal_uv = np.asarray([-direction_uv[1], direction_uv[0]])
        support = observation["_support_uv"]
        weights = observation["boundary_weights"]
        weights = weights / max(float(np.sum(weights)), 1e-12)
        confidence = max(float(observation["confidence"]), 1e-12)
        block = ((support - point_uv) @ normal_uv)
        residuals.extend(block * np.sqrt(confidence * weights))
    return np.asarray(residuals, dtype=np.float64)


def _refine_shared_edge(points, edge, pair, observations, plane_pair, cfg,
                        graph_id, refined_edge_id):
    support_ids = np.unique(np.concatenate([
        np.asarray(item["boundary_support_indices"], dtype=np.int64)
        for item in observations
    ]))
    support = points[support_ids]
    point0 = np.asarray(edge["center_mm"], dtype=np.float64)
    direction0 = _normalize(edge["direction"])
    line_u, line_v = _line_basis(direction0)
    direction_scale = max(0.5 * float(edge["length_mm"]), cfg.huber_delta_mm)
    for observation in observations:
        plane = next(item for item in plane_pair
                     if item["plane_id"] == observation["plane_id"])
        center = np.asarray(plane["refined"]["center_mm"], dtype=np.float64)
        u = np.asarray(plane["refined"]["u"], dtype=np.float64)
        v = np.asarray(plane["refined"]["v"], dtype=np.float64)
        xyz = points[np.asarray(observation["boundary_support_indices"], dtype=np.int64)]
        observation["_support_uv"] = np.column_stack(((xyz-center)@u, (xyz-center)@v))

    def residual(parameters):
        return _shared_edge_residual(
            parameters, point0, direction0, line_u, line_v,
            plane_pair, observations, direction_scale,
        )

    solved, optimization = _solve(residual, 4, cfg)
    parameters = solved.x
    point = point0 + parameters[0] * line_u + parameters[1] * line_v
    direction = _normalize(
        direction0 + parameters[2] * line_u + parameters[3] * line_v
    )
    refined = _line_from_point_direction(point, direction, support)
    for observation in observations:
        observation.pop("_support_uv", None)
    return {
        "refined_edge_id": int(refined_edge_id),
        "graph_id": int(graph_id),
        "source_edge_id": int(edge["edge_id"]),
        "plane_ids": [int(pair[0]), int(pair[1])],
        "shared": True,
        "mode": "shared_plane_boundary_joint",
        "boundary_observation_ids": [int(item["observation_id"]) for item in observations],
        "support_indices": support_ids,
        "initial": {
            "center_mm": np.asarray(edge["center_mm"]),
            "direction": direction0,
            "start_mm": np.asarray(edge["start_mm"]),
            "end_mm": np.asarray(edge["end_mm"]),
            "length_mm": float(edge["length_mm"]),
            "rmse_mm": float(edge["rms_mm"]),
        },
        "refined": refined,
        "optimization": optimization,
    }


def _single_plane_edge(points, edge, observation, graph_id, refined_edge_id):
    return {
        "refined_edge_id": int(refined_edge_id),
        "graph_id": int(graph_id),
        "source_edge_id": int(edge["edge_id"]),
        "plane_ids": [int(observation["plane_id"])],
        "shared": False,
        "mode": "single_plane_2d_lift",
        "boundary_observation_ids": [int(observation["observation_id"])],
        "support_indices": np.asarray(observation["boundary_support_indices"], dtype=np.int64),
        "initial": {
            "center_mm": np.asarray(edge["center_mm"]),
            "direction": np.asarray(edge["direction"]),
            "start_mm": np.asarray(edge["start_mm"]),
            "end_mm": np.asarray(edge["end_mm"]),
            "length_mm": float(edge["length_mm"]),
            "rmse_mm": float(edge["rms_mm"]),
        },
        "refined": observation["restored_3d"],
        "optimization": observation["optimization"],
    }


def _finite_contains(edge, point, extension):
    direction = _normalize(edge["refined"]["direction"])
    values = [float(np.asarray(edge["refined"][key]) @ direction)
              for key in ("start_mm", "end_mm")]
    coordinate = float(np.asarray(point) @ direction)
    return min(values) - extension <= coordinate <= max(values) + extension


def _refined_corners(edges, cfg):
    raw = []
    for left, right in combinations(edges, 2):
        if left["graph_id"] != right["graph_id"]:
            continue
        p = np.asarray(left["refined"]["center_mm"], dtype=np.float64)
        q = np.asarray(right["refined"]["center_mm"], dtype=np.float64)
        d = _normalize(left["refined"]["direction"])
        e = _normalize(right["refined"]["direction"])
        matrix = np.column_stack((d, -e))
        if np.linalg.matrix_rank(matrix) < 2:
            continue
        times = np.linalg.lstsq(matrix, q-p, rcond=None)[0]
        point_left, point_right = p + times[0]*d, q + times[1]*e
        gap = float(np.linalg.norm(point_left-point_right))
        position = 0.5 * (point_left+point_right)
        if gap > cfg.corner_intersection_mm:
            continue
        if not (_finite_contains(left, position, cfg.corner_endpoint_gap_mm)
                and _finite_contains(right, position, cfg.corner_endpoint_gap_mm)):
            continue
        raw.append({
            "graph_id": int(left["graph_id"]),
            "edge_ids": [int(left["refined_edge_id"]), int(right["refined_edge_id"])],
            "source_edge_ids": sorted(set((int(left["source_edge_id"]),
                                            int(right["source_edge_id"])))),
            "plane_ids": sorted(set(left["plane_ids"] + right["plane_ids"])),
            "position_mm": position,
            "line_gap_mm": gap,
            "type": "refined_finite_edge_intersection",
        })
    raw.sort(key=lambda item: (item["line_gap_mm"], item["graph_id"]))
    corners = []
    for candidate in raw:
        match = next((item for item in corners
                      if item["graph_id"] == candidate["graph_id"]
                      and np.linalg.norm(item["position_mm"]-candidate["position_mm"])
                      <= cfg.corner_merge_mm), None)
        if match is None:
            candidate["corner_id"] = len(corners)
            corners.append(candidate)
        else:
            match["edge_ids"] = sorted(set(match["edge_ids"] + candidate["edge_ids"]))
            match["source_edge_ids"] = sorted(set(
                match["source_edge_ids"] + candidate["source_edge_ids"]
            ))
            match["plane_ids"] = sorted(set(match["plane_ids"] + candidate["plane_ids"]))
    return corners


def refine_hough_edge_graph(points_mm, detector_result, config=None):
    """Refine connected plane-edge components without mutating detector output."""
    cfg = config if config is not None else HoughEdgeRefinerConfig()
    cfg.validate()
    started = time.perf_counter()
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite points with shape (N,3)")
    planes = detector_result.get("planes", ())
    edges = detector_result.get("edges", ())
    proposals = detector_result.get("edge_proposals", ())
    plane_by_id = {int(item["plane_id"]): item for item in planes}
    proposal_by_id = {int(item["proposal_id"]): item for item in proposals}
    components = _graph_components(plane_by_id, edges)
    graph_for_plane = {
        plane_id: graph_id
        for graph_id, component in enumerate(components)
        for plane_id in component
    }
    refined_planes = {
        plane_id: _refine_plane(points, plane, cfg, graph_for_plane[plane_id])
        for plane_id, plane in plane_by_id.items()
    }
    observations = []
    refined_edges = []
    intersections = []
    edge_ids_by_graph = {graph_id: [] for graph_id in range(len(components))}

    for edge in sorted(edges, key=lambda item: int(item["edge_id"])):
        plane_ids = sorted(set(int(value) for value in edge.get("plane_ids", ())))
        if not plane_ids:
            continue
        graph_id = graph_for_plane[plane_ids[0]]
        observation_by_plane = {}
        for plane_id in plane_ids:
            source_ids, proposal_ids = _edge_source_ids(edge, plane_id, proposal_by_id)
            if not len(source_ids):
                continue
            weights = _boundary_weights_for_ids(plane_by_id[plane_id], source_ids)
            observation = _refine_boundary_observation(
                points, source_ids, weights, refined_planes[plane_id], edge, cfg,
                graph_id, int(edge["edge_id"]), proposal_ids,
            )
            observation["observation_id"] = len(observations)
            observations.append(observation)
            observation_by_plane[plane_id] = observation
        available = sorted(observation_by_plane)
        if len(available) == 1:
            refined = _single_plane_edge(
                points, edge, observation_by_plane[available[0]], graph_id,
                len(refined_edges),
            )
            refined_edges.append(refined)
            edge_ids_by_graph[graph_id].append(refined["refined_edge_id"])
            continue
        for pair in combinations(available, 2):
            pair_planes = [refined_planes[pair[0]], refined_planes[pair[1]]]
            pair_observations = [observation_by_plane[pair[0]],
                                 observation_by_plane[pair[1]]]
            support_ids = np.unique(np.concatenate([
                item["boundary_support_indices"] for item in pair_observations
            ]))
            intersection = _plane_intersection(
                pair_planes[0], pair_planes[1], edge["center_mm"], points[support_ids]
            )
            if intersection is not None:
                intersections.append({
                    "intersection_id": len(intersections),
                    "graph_id": int(graph_id),
                    "source_edge_id": int(edge["edge_id"]),
                    "plane_ids": [int(pair[0]), int(pair[1])],
                    **intersection,
                })
            refined = _refine_shared_edge(
                points, edge, pair, pair_observations, pair_planes, cfg,
                graph_id, len(refined_edges),
            )
            refined_edges.append(refined)
            edge_ids_by_graph[graph_id].append(refined["refined_edge_id"])

    corners = _refined_corners(refined_edges, cfg)
    graphs = [{
        "graph_id": graph_id,
        "plane_ids": component,
        "refined_edge_ids": edge_ids_by_graph[graph_id],
        "corner_ids": [int(item["corner_id"]) for item in corners
                       if item["graph_id"] == graph_id],
    } for graph_id, component in enumerate(components)]
    return {
        "method": "graph_local_huber_plane_boundary_edge_refiner",
        "config": asdict(cfg),
        "graphs": graphs,
        "planes": [refined_planes[key] for key in sorted(refined_planes)],
        "boundary_observations": observations,
        "plane_intersections": intersections,
        "edges": refined_edges,
        "corners": corners,
        "diagnostics": {
            "graph_count": len(graphs),
            "plane_count": len(refined_planes),
            "boundary_observation_count": len(observations),
            "refined_edge_count": len(refined_edges),
            "plane_intersection_count": len(intersections),
            "corner_count": len(corners),
            "elapsed_s": time.perf_counter() - started,
        },
    }


def mean_direction_q10(query_mm, neighbors_mm, direction_mode="unit_mean"):
    """Unweighted cosine Q10 around the selected normalized neighborhood direction.

    Median means componentwise median of unit vectors. Centroid uses original
    XYZ offsets, so distant observations affect the axis but not quantile weights.
    """
    if direction_mode not in Q10_DIRECTION_MODES:
        raise ValueError(f"Q10 direction mode must be one of {Q10_DIRECTION_MODES}")
    query, neighbors = np.asarray(query_mm, dtype=float), np.asarray(neighbors_mm, dtype=float)
    if (query.shape != (3,) or neighbors.ndim != 2 or neighbors.shape[1] != 3
            or not np.isfinite(query).all() or not np.isfinite(neighbors).all()):
        raise ValueError("Q10 requires finite XYZ coordinates")
    offsets = neighbors - query
    lengths = np.linalg.norm(offsets, axis=1)
    units = offsets[lengths > 1e-7] / lengths[lengths > 1e-7, None]
    if len(units) < 3:
        raise ValueError("Need at least three nonzero directions")
    if direction_mode == "unit_mean":
        mean = units.mean(axis=0)
    elif direction_mode == "unit_median":
        mean = np.median(units, axis=0)
    else:
        mean = offsets.mean(axis=0)
    length = np.linalg.norm(mean)
    if length < 1e-8:
        raise ValueError("Undefined Q10 direction")
    axis = mean / length
    return float(np.quantile(np.clip(units @ axis, -1., 1.), .1)), axis


class Q10PositionRefiner:
    """One frame/radius/CC configuration owns Q10 support and response caches."""

    def __init__(self, points_mm, radius_mm, step_mm, cc_factor=4., cc_enabled=True,
                 direction_mode="unit_mean"):
        from scipy.spatial import cKDTree
        self.points = np.asarray(points_mm, dtype=float)
        if (self.points.ndim != 2 or self.points.shape[1] != 3
                or not np.isfinite(self.points).all()):
            raise ValueError("Q10 analysis points must be finite (N,3)")
        if not np.isfinite([radius_mm, step_mm, cc_factor]).all() or min(radius_mm, step_mm, cc_factor) <= 0:
            raise ValueError("Q10 neighborhood, NMS radius and CC factor must be positive and finite")
        if not isinstance(cc_enabled, (bool, np.bool_)):
            raise ValueError("Q10 CC switch must be boolean")
        if direction_mode not in Q10_DIRECTION_MODES:
            raise ValueError(f"Q10 direction mode must be one of {Q10_DIRECTION_MODES}")
        self.direction_mode = direction_mode
        self.radius_mm, self.step_mm = float(radius_mm), float(step_mm)
        self.cc_factor = float(cc_factor)
        self.cc_enabled = bool(cc_enabled)
        self.tree = cKDTree(self.points)
        self.cache = {}
        self.support_cache = {}
        self.cache_hits = self.cache_misses = 0

    def evaluate(self, index, *, minimum_support=10):
        from scipy.spatial import cKDTree
        if index in self.cache:
            self.cache_hits += 1
            return self.cache[index]
        self.cache_misses += 1
        value, axis = None, None
        if index not in self.support_cache:
            ids = np.sort(np.asarray(self.tree.query_ball_point(self.points[index], self.radius_mm + 1e-8), dtype=np.int64))
            component = np.empty(0, dtype=np.int64)
            median, epsilon = np.nan, np.nan
            if not self.cc_enabled:
                component = ids
            elif len(ids) >= 10:
                median = float(np.median(cKDTree(self.points[ids]).query(self.points[ids], k=2)[0][:, 1]))
                epsilon = self.cc_factor * median
                if np.isfinite(epsilon) and epsilon > 0:
                    labels = _components(np.vstack((self.points[ids], self.points[index])), epsilon)
                    component = ids[labels[:-1] == labels[-1]]
            self.support_cache[index] = {"component_analysis_indices": component, "roi_count": len(ids),
                                         "median_nn_mm": median, "cc_radius_mm": epsilon}
        component = self.support_cache[index]["component_analysis_indices"]
        if 10 <= len(component) < minimum_support:
            # This seed's support gate is not an invalid Q10; a later seed may need the score.
            return None, len(component), np.full(3, np.nan)
        if len(component) >= 10:
            try:
                value, axis = mean_direction_q10(self.points[index], self.points[component], self.direction_mode)
            except ValueError:
                pass
        if axis is None:
            axis = np.full(3, np.nan)
        self.cache[index] = value, len(component), axis
        return self.cache[index]

    def refine(self, initial_index, *, center_mm=None):
        """Select once inside the original NMS ball; never recenter the search."""
        if (isinstance(initial_index, (bool, np.bool_)) or not isinstance(initial_index, (int, np.integer))
                or not 0 <= initial_index < len(self.points)):
            raise ValueError("Q10 starting index must address the analysis cloud")
        center = self.points[initial_index] if center_mm is None else np.asarray(center_mm, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Q10 NMS center must be a finite XYZ coordinate")
        candidates = sorted(self.tree.query_ball_point(center, self.step_mm + 1e-8))
        if not candidates:
            return {"analysis_indices": np.empty(0, dtype=np.int64), "positions_mm": center[None, :].copy(),
                    "q10_values": np.array([np.nan]), "neighbor_counts": np.array([0], dtype=np.int64),
                    "mean_axes": np.full((1, 3), np.nan), "tested_positions": np.array([0], dtype=np.int64),
                    "support_analysis_indices": [np.empty(0, dtype=np.int64)], "roi_counts": np.array([0]),
                    "cc_median_nn_mm": np.array([np.nan]), "cc_radii_mm": np.array([np.nan]),
                    "converged": False, "stop_reason": "no_analysis_point_in_nms_ball"}
        current = int(initial_index) if initial_index in candidates else min(
            candidates, key=lambda i: (np.linalg.norm(self.points[i] - center), i))
        path, values, counts, axes, tested = [], [], [], [], []
        value, count, axis = self.evaluate(current)
        path.append(current)
        values.append(np.nan if value is None else value)
        counts.append(count)
        axes.append(axis.copy())
        reason, converged = "invalid_initial_q10", False
        if value is not None and len(candidates) > 3500:
            reason = "too_many_step_candidates"
        elif value is not None:
            best, best_value = current, value
            best_distance = float(np.linalg.norm(self.points[current] - center))
            minimum = max(10, int(np.ceil(.8 * count)))
            for candidate in candidates:
                if candidate == current:
                    continue
                score, support_count, _ = self.evaluate(candidate, minimum_support=minimum)
                if score is None or support_count < minimum:
                    continue
                distance = float(np.linalg.norm(self.points[candidate] - center))
                if score > best_value + 1e-9 or (abs(score - best_value) <= 1e-9 and
                        (distance < best_distance - 1e-9 or
                         (abs(distance - best_distance) <= 1e-9 and candidate < best))):
                    best, best_value, best_distance = candidate, score, distance
            tested.append(len(candidates))
            if best != current and best_value > value + 1e-9:
                score, support_count, best_axis = self.evaluate(best)
                path.append(best)
                values.append(score)
                counts.append(support_count)
                axes.append(best_axis.copy())
            reason, converged = "fixed_nms_maximum", True
        supports = [self.support_cache.get(i, {}) for i in path]
        return {"analysis_indices": np.asarray(path, dtype=np.int64),
                "positions_mm": self.points[path].copy(), "q10_values": np.asarray(values),
                "neighbor_counts": np.asarray(counts, dtype=np.int64), "mean_axes": np.asarray(axes),
                "tested_positions": np.asarray(tested, dtype=np.int64),
                "support_analysis_indices": [s.get("component_analysis_indices", np.empty(0, dtype=np.int64)).copy() for s in supports],
                "roi_counts": np.asarray([s.get("roi_count", 0) for s in supports], dtype=np.int64),
                "cc_median_nn_mm": np.asarray([s.get("median_nn_mm", np.nan) for s in supports]),
                "cc_radii_mm": np.asarray([s.get("cc_radius_mm", np.nan) for s in supports]),
                "converged": converged, "stop_reason": reason}


__all__ = ["HoughEdgeRefinerConfig", "refine_hough_edge_graph", "Q10PositionRefiner", "mean_direction_q10"]
