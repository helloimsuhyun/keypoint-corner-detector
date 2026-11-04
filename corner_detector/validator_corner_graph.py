"""Finite-edge junctions and shared-edge-first Corner selection inside the query gate."""

from itertools import combinations
from time import perf_counter

import numpy as np

from .validator_edge_graph import edge_parameter_is_active


def empty_corner_validation(query_mm, plane_count, reason):
    return {"verified": False, "validation_status": "NOT_COMPUTED",
            "geometry_computed": False, "query_position_mm": np.asarray(query_mm).copy(),
            "corner_position_mm": None, "corner_distance_mm": None,
            "plane_count": plane_count, "plane_patches": [], "plane_nodes": [],
            "plane_adjacency": [], "edges": [], "edge_candidates": [], "vertices": [],
            "selected_vertex_id": None, "corner_edge_ids": [], "corner_edges": [],
            "failure_reason": reason, "stage_elapsed_s": {},
            "point_support_checked": False,
            "limitations": ["convex_hulls_do_not_recover_concavity",
                            "occlusion_boundaries_can_resemble_object_edges",
                            "edge_and_vertex_point_support_not_checked"]}


def _segment_distance(position, edge):
    along = np.clip((position - edge["start_mm"]) @ edge["direction"], 0., edge["length_mm"])
    return float(np.linalg.norm(position - edge["start_mm"] - along * edge["direction"]))


def _edge_reaches_vertex(position, edge, cfg):
    parameter = float((position - edge["start_mm"]) @ edge["direction"])
    tolerance = cfg.vertex_endpoint_tolerance_mm
    return (edge_parameter_is_active(edge, parameter, tolerance)
            and _segment_distance(position, edge)
            <= cfg.vertex_line_distance_mm)


def _closest_candidate(left, right, cfg):
    shared_node_id = None
    if (left["edge_type"] == right["edge_type"] == "boundary"
            and set(left["plane_ids"]) & set(right["plane_ids"])):
        left_nodes = {left[side + "_node_id"]: (left[side + "_node"], time)
                      for side, time in (("start", 0.), ("end", left["length_mm"]))}
        right_nodes = {right[side + "_node_id"]: (right[side + "_node"], time)
                       for side, time in (("start", 0.), ("end", right["length_mm"]))}
        shared = sorted(left_nodes.keys() & right_nodes.keys())
        shared_node_id = next((identifier for identifier in shared
                               if left_nodes[identifier][0]["corner_enabled"]
                               and right_nodes[identifier][0]["corner_enabled"]
                               and edge_parameter_is_active(left, left_nodes[identifier][1])
                               and edge_parameter_is_active(right, right_nodes[identifier][1])), None)
        if shared_node_id is None:
            return None
    if left["edge_type"] != "plane_intersection" and right["edge_type"] == "plane_intersection":
        left, right = right, left
    d, e = left["direction"], right["direction"]
    angle = float(np.rad2deg(np.arccos(np.clip(abs(d @ e), 0., 1.))))
    if angle < cfg.corner_min_edge_angle_deg:
        return None
    matrix = np.column_stack((d, -e))
    times = np.linalg.lstsq(matrix, right["start_mm"] - left["start_mm"], rcond=None)[0]
    a, b = left["start_mm"] + times[0] * d, right["start_mm"] + times[1] * e
    gap = float(np.linalg.norm(a - b))
    if gap > cfg.vertex_line_distance_mm:
        return None
    tolerance = cfg.vertex_endpoint_tolerance_mm
    if not all(edge_parameter_is_active(edge, time, tolerance)
               for edge, time in zip((left, right), times)):
        return None
    anchor = left["edge_id"] if left["edge_type"] == "plane_intersection" else None
    return {"position_mm": a if anchor is not None else .5 * (a + b),
            "shared_node_id": shared_node_id,
            "edge_ids": [left["edge_id"], right["edge_id"]],
            "anchor_edge_id": anchor, "anchor_parameter_mm": float(times[0]),
            "line_gap_mm": gap, "edge_angle_deg": angle}


def _refine_vertex(edge_ids, by_id, cfg):
    edges = [by_id[index] for index in edge_ids]
    intersections = [edge for edge in edges if edge["edge_type"] == "plane_intersection"]
    if len(intersections) == 1:
        anchor = intersections[0]
        a, u = anchor["start_mm"], anchor["direction"]
        numerator, denominator = 0., 0.
        for edge in edges:
            if edge["edge_id"] == anchor["edge_id"]:
                continue
            v = edge["direction"]
            projection = np.eye(3) - np.outer(v, v)
            numerator += float(u @ projection @ (edge["start_mm"] - a))
            denominator += float(u @ projection @ u)
        if denominator <= 1e-12 * max(1, len(edges) - 1):
            return None, "degenerate_anchor_ls"
        parameter = np.clip(numerator / denominator, 0., anchor["length_mm"])
        position, method = a + parameter * u, "intersection_constrained_ls"
    else:
        reference = np.mean([edge["start_mm"] for edge in edges], axis=0)
        matrix, rhs = np.zeros((3, 3)), np.zeros(3)
        for edge in edges:
            v = edge["direction"]
            projection = np.eye(3) - np.outer(v, v)
            matrix += projection
            rhs += projection @ (edge["start_mm"] - reference)
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[-1] <= 1e-12 * max(1., singular[0]):
            return None, "degenerate_multiline_ls"
        position = reference + np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        method = "multiline_3d_ls"
    if not np.isfinite(position).all() or not all(_edge_reaches_vertex(position, edge, cfg) for edge in edges):
        return None, "refined_vertex_outside_active_edges_or_residual_gate"
    return position, method


def reconstruct_validator_corners(points_mm, query_mm, support, graph, cfg):
    """Group active-edge intersections and refine on anchors before existing corner gates."""
    started = perf_counter()
    query = np.asarray(query_mm, dtype=np.float64)
    edges = graph["edges"]
    by_id = {edge["edge_id"]: edge for edge in edges}
    raw = [candidate for left, right in combinations(edges, 2)
           if (candidate := _closest_candidate(left, right, cfg)) is not None]
    groups, anchored, rejected = [], {}, []
    for candidate in raw:
        anchor = candidate["anchor_edge_id"]
        if anchor is None:
            groups.append([candidate])
        else:
            anchored.setdefault(anchor, []).append(candidate)
    for candidates in anchored.values():
        local = []
        for candidate in sorted(candidates, key=lambda item: (item["anchor_parameter_mm"], item["edge_ids"])):
            if (local and candidate["anchor_parameter_mm"] - local[0]["anchor_parameter_mm"]
                    > cfg.vertex_group_span_mm):
                groups.append(local)
                local = []
            local.append(candidate)
        if local:
            groups.append(local)
    vertices = []
    for group in groups:
        edge_ids = sorted({index for candidate in group for index in candidate["edge_ids"]})
        members = [candidate["position_mm"].copy() for candidate in group]
        position, method = _refine_vertex(edge_ids, by_id, cfg)
        if position is None or any(np.linalg.norm(position - member) > cfg.vertex_group_span_mm for member in members):
            rejected.append({"edge_ids": edge_ids, "failure_reason": method if position is None else "refinement_left_group"})
            continue
        candidate = {"position_mm": position, "edge_ids": edge_ids,
                     "member_positions_mm": members, "refinement_method": method,
                     "line_gap_mm": max(item["line_gap_mm"] for item in group),
                     "edge_angle_deg": min(item["edge_angle_deg"] for item in group)}
        for vertex in vertices:
            if not set(vertex["edge_ids"]) & set(edge_ids):
                continue
            if any(np.linalg.norm(a - b) > cfg.vertex_group_span_mm
                   for a in vertex["member_positions_mm"] for b in members):
                continue
            combined_ids = sorted(set(vertex["edge_ids"]) | set(edge_ids))
            combined_members = vertex["member_positions_mm"] + members
            refined, combined_method = _refine_vertex(combined_ids, by_id, cfg)
            if refined is None or any(np.linalg.norm(refined - member) > cfg.vertex_group_span_mm for member in combined_members):
                continue
            vertex.update(position_mm=refined, edge_ids=combined_ids,
                          member_positions_mm=combined_members, refinement_method=combined_method,
                          line_gap_mm=max(vertex["line_gap_mm"], candidate["line_gap_mm"]),
                          edge_angle_deg=min(vertex["edge_angle_deg"], candidate["edge_angle_deg"]))
            break
        else:
            vertices.append(candidate)
    for index, vertex in enumerate(vertices):
        position = vertex["position_mm"]
        distance = float(np.linalg.norm(position - query))
        inside = distance <= cfg.corner_gate_radius_mm
        inside_core = distance <= cfg.ransac_radius_mm
        incident = [by_id[edge_id] for edge_id in vertex["edge_ids"]]
        line_residuals, extensions = [], []
        for edge in incident:
            delta = position - edge["start_mm"]
            parameter = float(delta @ edge["direction"])
            line_residuals.append(float(np.linalg.norm(delta - parameter * edge["direction"])))
            extensions.append(max(0., -parameter, parameter - edge["length_mm"]))
        vertex.update(vertex_id=index, plane_ids=sorted({plane_id for edge_id in vertex["edge_ids"]
                                                        for plane_id in by_id[edge_id]["plane_ids"]}),
                      intersection_edge_count=sum(edge["edge_type"] == "plane_intersection" for edge in incident),
                      line_fit_rms_mm=float(np.sqrt(np.mean(np.square(line_residuals)))),
                      total_extension_mm=float(sum(extensions)),
                      supported=True, support_quality=min(by_id[edge_id]["support_quality"] for edge_id in vertex["edge_ids"]),
                      distance_to_query_mm=distance, within_center_gate=inside,
                      within_core_roi=bool(inside_core), valid=bool(inside and inside_core),
                      status="ACCEPT" if inside and inside_core else "REJECT",
                      ls_residuals_mm=[_segment_distance(position, by_id[edge_id]) for edge_id in vertex["edge_ids"]])
    accepted = [vertex for vertex in vertices if vertex["valid"]]
    # Ignore sub-1e-8 mm roundoff so exact intersections are ranked by extension/distance.
    selected = min(accepted, key=lambda item: (item["intersection_edge_count"] == 0,
                                              round(item["line_fit_rms_mm"], 8),
                                              round(item["total_extension_mm"], 8),
                                              item["distance_to_query_mm"], item["vertex_id"])) if accepted else None
    if selected is not None:
        status, reason = "ACCEPT", None
    elif not support["plane_patches"]:
        status, reason = "INCONCLUSIVE", "no_planes"
    elif any(not patch["valid"] for patch in support["plane_patches"]):
        status, reason = "INCONCLUSIVE", "insufficient_or_ambiguous_plane_support"
    elif all(np.linalg.norm(np.ptp(patch["hull_uv"], axis=0)) < cfg.edge_min_length_mm
             for patch in support["plane_patches"]):
        status, reason = "INCONCLUSIVE", "insufficient_observed_patch_extent"
    elif (any(not vertex["supported"] and vertex["within_center_gate"] for vertex in vertices)
          or any(not edge["valid"] and _segment_distance(query, edge) <= cfg.corner_gate_radius_mm
                 for edge in graph["edge_candidates"])):
        status, reason = "INCONCLUSIVE", "insufficient_local_edge_geometry"
    else:
        status, reason = "REJECT", "no_supported_finite_edge_junction_within_gate"
    result = empty_corner_validation(query, len(support["plane_patches"]), reason)
    result.update(vertex_rejections=rejected, vertex_candidates=raw,
                  geometry_computed=True, verified=selected is not None,
                  validation_status=status, vertices=vertices,
                  plane_nodes=[{"plane_id": patch["plane_id"], "valid": patch["valid"],
                                "edge_ids": [edge["edge_id"] for edge in edges
                                             if patch["plane_id"] in edge["plane_ids"]]}
                               for patch in support["plane_patches"]])
    if selected is not None:
        result.update(corner_position_mm=selected["position_mm"].copy(),
                      corner_distance_mm=selected["distance_to_query_mm"],
                      selected_vertex_id=selected["vertex_id"],
                      corner_edge_ids=selected["edge_ids"].copy(),
                      corner_edges=[by_id[index] for index in selected["edge_ids"]])
    result["elapsed_s"] = perf_counter() - started
    return result
