"""Hull-derived finite boundary and plane-pair edges in the RANSAC voxel domain."""

from itertools import combinations
from time import perf_counter

import numpy as np


def _hull_line_interval(patch, origin, direction):
    basis = np.column_stack((patch["basis_u"], patch["basis_v"]))
    equations = patch["hull_equations"]
    position = (origin - patch["origin_mm"]) @ basis
    tangent = direction @ basis
    values = equations[:, :2] @ position + equations[:, 2]
    slopes = equations[:, :2] @ tangent
    low, high = -np.inf, np.inf
    for value, slope in zip(values, slopes):
        if abs(slope) < 1e-12:
            if value > 1e-8:
                return None
        elif slope > 0:
            high = min(high, -value / slope)
        else:
            low = max(low, -value / slope)
    if not np.isfinite([low, high]).all() or high <= low:
        return None
    return low, high


def _hull_strip_interval(patch, origin, direction, half_width_mm):
    """Project hull/strip clipping onto the original 3-D line parameter."""
    basis = np.column_stack((patch["basis_u"], patch["basis_v"]))
    position = (origin - patch["origin_mm"]) @ basis
    tangent = direction @ basis
    norm = np.linalg.norm(tangent)
    polygon = np.asarray(patch["hull_uv"], dtype=np.float64)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or not len(polygon)
            or not np.isfinite(polygon).all() or not np.isfinite(position).all()
            or not np.isfinite(norm) or norm <= 1e-12):
        return None, None
    tangent = tangent / norm
    normal = np.array([-tangent[1], tangent[0]])
    signed = (polygon - position) @ normal
    distance = (0. if signed.min() <= 0 <= signed.max()
                else float(np.min(np.abs(signed))))
    if half_width_mm == 0:
        return _hull_line_interval(patch, origin, direction), distance
    for sign in (1., -1.):
        if not len(polygon):
            return None, distance
        values = sign * ((polygon - position) @ normal) - half_width_mm
        values[np.abs(values) <= 1e-8] = 0.
        clipped = []
        previous, previous_value = polygon[-1], values[-1]
        for current, value in zip(polygon, values):
            if (value <= 0) != (previous_value <= 0):
                fraction = previous_value / (previous_value - value)
                clipped.append(previous + fraction * (current - previous))
            if value <= 0:
                clipped.append(current)
            previous, previous_value = current, value
        polygon = np.asarray(clipped, dtype=np.float64).reshape(-1, 2)
    if not len(polygon):
        return None, distance
    projected = ((polygon - position) @ tangent) / norm
    low, high = float(projected.min()), float(projected.max())
    if not np.isfinite([low, high]).all() or high - low <= 1e-8:
        return None, distance
    return (low, high), distance


def _core_segment_interval(start, end, query, radius):
    delta = end - start
    length = np.linalg.norm(delta)
    if length <= 1e-10:
        return None
    direction = delta / length
    relative = start - query
    center = -float(relative @ direction)
    closest = relative + center * direction
    discriminant = radius * radius - float(closest @ closest)
    if discriminant <= 1e-10:
        return None
    half_length = np.sqrt(discriminant)
    low, high = max(0., center - half_length), min(length, center + half_length)
    return (low, high) if high - low > 1e-8 else None


def _edge(points, patches, start, end, edge_type, query, support_radius, cfg, *, line=None):
    original_start, original_end = start.copy(), end.copy()
    original_length = np.linalg.norm(end - start)
    interval = _core_segment_interval(start, end, query, cfg.ransac_radius_mm)
    if interval is None:
        return None
    low, high = interval
    direction = (end - start) / original_length
    start = original_start if low == 0 else original_start + low * direction
    end = original_end if high == original_length else original_start + high * direction
    length = float(np.linalg.norm(end - start))
    if length < cfg.edge_min_length_mm:
        return None
    # Keep provenance slots empty: plane memberships are not measured edge support.
    ids = np.empty(0, dtype=np.int64)
    edge = {"edge_type": edge_type, "plane_ids": [p["plane_id"] for p in patches],
            "start_mm": start.copy(), "end_mm": end.copy(),
            "original_start_mm": original_start, "original_end_mm": original_end,
            "roi_clipped_start": bool(low > 1e-8),
            "roi_clipped_end": bool(high < original_length - 1e-8),
            "center_mm": (start + end) * .5, "direction": (end - start) / length,
            "length_mm": length, "support_ransac_indices": ids,
            "support_indices": ids.copy(), "support_count": len(ids),
            "support_by_plane": [], "point_support_checked": False,
            "support_quality": 1.,
            "roi_truncated": False, "valid": True,
            "status": "VALID", "failure_reason": None,
            "fit_tolerance_mm": cfg.boundary_merge_rms_error_mm if line is not None else cfg.inlier_threshold_mm}
    if line is not None:
        edge.update(plane_id=line["plane_id"], line_id=line["id"], source_hull_edge_ids=line["source_hull_edge_ids"].copy(),
                    support_node_ids=line["support_node_ids"].copy(),
                    tls_rms_error_mm=line["tls_rms_error_mm"],
                    tls_max_error_mm=line["tls_max_error_mm"])
        for side in ("start", "end"):
            node = line[side + "_node"]
            if edge["roi_clipped_" + side]:
                position = edge[side + "_mm"]
                patch = patches[0]
                node = {"id": line["id"] + ":roi_" + side,
                        "position_2d": np.array([(position - patch["origin_mm"]) @ patch["basis_u"],
                                                  (position - patch["origin_mm"]) @ patch["basis_v"]]),
                        "position_3d": position.copy(), "source": "CLIPPED",
                        "corner_enabled": False, "parent_node_id": node["id"]}
            edge[side + "_node"] = node
            edge[side + "_node_id"] = node["id"]
            edge["roi_clipped_" + side] |= not node["corner_enabled"]
    return edge


def _clip_hull_boundary(patch, query, radius):
    """Clip original hull segments; never close the resulting open chains along the ROI."""
    vertices = patch["hull_uv"]
    origin = patch["origin_mm"]
    basis = np.column_stack((patch["basis_u"], patch["basis_v"]))
    center = (query - origin) @ basis
    section_radius2 = radius ** 2 - float(np.sum((query - origin) ** 2))
    plane_id = patch["plane_id"]

    def node(identifier, uv, source):
        return {"id": identifier, "position_2d": uv.copy(),
                "position_3d": origin + uv @ basis.T,
                "source": source, "corner_enabled": source == "ORIGINAL"}

    originals = [node(f"p{plane_id}:h{index}", uv, "ORIGINAL") for index, uv in enumerate(vertices)]
    nodes, hull_edges, clipped = {}, [], []
    for index, start in enumerate(originals):
        end = originals[(index + 1) % len(originals)]
        source = {"id": f"p{plane_id}:e{index}", "plane_id": plane_id,
                  "source_hull_edge_id": index, "start_node_id": start["id"], "end_node_id": end["id"],
                  "start_mm": start["position_3d"], "end_mm": end["position_3d"],
                  "clip_interval_mm": None}
        hull_edges.append(source)
        if section_radius2 <= 1e-10:
            continue
        a, b = start["position_2d"], end["position_2d"]
        interval = _core_segment_interval(a, b, center, np.sqrt(section_radius2))
        if interval is None:
            continue
        length = np.linalg.norm(b - a)
        direction = (b - a) / length
        low, high = interval
        source["clip_interval_mm"] = interval
        if low > 1e-8:
            start = node(source["id"] + ":start", a + low * direction, "CLIPPED")
        if high < length - 1e-8:
            end = node(source["id"] + ":end", a + high * direction, "CLIPPED")
        nodes[start["id"]], nodes[end["id"]] = start, end
        clipped.append({**source, "start_node_id": start["id"], "end_node_id": end["id"],
                        "start_mm": start["position_3d"], "end_mm": end["position_3d"]})
    return {"plane_id": plane_id, "original_nodes": originals, "nodes": nodes,
            "hull_edges": hull_edges, "clipped_edges": clipped, "lines": [],
            "roi_center_uv": center, "roi_section_radius_mm": np.sqrt(max(0., section_radius2))}


def _boundary_segments(patch, cfg, query):
    debug = _clip_hull_boundary(patch, query, cfg.ransac_radius_mm)
    nodes = debug["nodes"]

    def fit_tls(vertices):
        points = np.unique(np.asarray(vertices), axis=0)
        if len(points) < 2:
            return None
        center = points.mean(axis=0)
        delta = points - center
        _, singular, axes = np.linalg.svd(delta, full_matrices=False)
        if singular[0] <= 1e-10:
            return None
        direction = axes[0]
        # Align the eigenvector sign with the group's hull traversal.
        if direction @ (vertices[-1] - vertices[0]) < 0:
            direction = -direction
        residual = delta - (delta @ direction)[:, None] * direction
        distances = np.linalg.norm(residual, axis=1)
        return center, direction, float(np.sqrt(np.mean(distances ** 2))), float(np.max(distances))

    groups = []
    for edge in debug["clipped_edges"]:
        start, end = [nodes[edge[side + "_node_id"]]["position_2d"] for side in ("start", "end")]
        length = np.linalg.norm(end - start)
        if length > 1e-10:
            direction = (end - start) / length
            groups.append({"vertices": [start, end], "directions": [direction],
                           "direction": direction, "node_ids": [edge["start_node_id"], edge["end_node_id"]],
                           "source_hull_edge_ids": [edge["source_hull_edge_id"]]})
    cosine = np.cos(np.deg2rad(cfg.boundary_merge_angle_deg))
    while len(groups) > 1:
        merged = False
        for index in range(len(groups)):
            following = (index + 1) % len(groups)
            a, b = groups[index], groups[following]
            if a["node_ids"][-1] != b["node_ids"][0]:
                continue
            directions = a["directions"] + b["directions"]
            if any(left @ right < cosine for left, right in combinations(directions, 2)):
                continue
            combined = a["vertices"] + b["vertices"][1:]
            fit = fit_tls(combined)
            if fit is None:
                continue
            _, direction, rms_error, _ = fit
            if rms_error > cfg.boundary_merge_rms_error_mm:
                continue
            if any(left @ right < cosine for left, right in combinations(
                    (a["direction"], b["direction"], direction), 2)):
                continue
            groups[index] = {"vertices": combined, "directions": directions,
                             "direction": direction, "node_ids": a["node_ids"] + b["node_ids"][1:],
                             "source_hull_edge_ids": a["source_hull_edge_ids"] + b["source_hull_edge_ids"]}
            groups.pop(following)
            merged = True
            break
        if not merged:
            break
    segments = []
    for group in groups:
        fit = fit_tls(group["vertices"])
        if fit is None:
            continue
        center, direction, rms_error, max_error = fit
        projected = (np.asarray(group["vertices"]) - center) @ direction
        endpoints_uv = center + np.array([projected.min(), projected.max()])[:, None] * direction
        endpoints_mm = (patch["origin_mm"] + endpoints_uv[:, :1] * patch["basis_u"]
                        + endpoints_uv[:, 1:] * patch["basis_v"])
        segments.append({"id": f"p{patch['plane_id']}:l{len(segments)}", "plane_id": patch["plane_id"],
                         "start_mm": endpoints_mm[0], "end_mm": endpoints_mm[1],
                         "start_node": nodes[group["node_ids"][0]],
                         "end_node": nodes[group["node_ids"][-1]],
                         "start_node_id": group["node_ids"][0], "end_node_id": group["node_ids"][-1],
                         "support_node_ids": group["node_ids"],
                         "source_hull_edge_ids": group["source_hull_edge_ids"],
                         "tls_rms_error_mm": rms_error,
                         "tls_max_error_mm": max_error})
    debug["lines"] = segments
    return segments, debug


def _mask_boundary_overlaps(edges, cfg):
    intersections = [edge for edge in edges if edge["edge_type"] == "plane_intersection"]
    diagnostics = []
    cosine = np.cos(np.deg2rad(cfg.edge_overlap_angle_deg))
    for boundary in edges:
        boundary["active_intervals_mm"] = [(0., boundary["length_mm"])]
        boundary["masked_intervals_mm"] = []
        if boundary["edge_type"] != "boundary":
            continue
        masks, sources = [], []
        for intersection in intersections:
            if not set(boundary["plane_ids"]) & set(intersection["plane_ids"]):
                continue
            c = float(boundary["direction"] @ intersection["direction"])
            if abs(c) < cosine:
                continue
            delta = np.stack((boundary["start_mm"], boundary["end_mm"])) - intersection["start_mm"]
            projected = delta @ intersection["direction"]
            perpendicular = delta - projected[:, None] * intersection["direction"]
            if np.max(np.linalg.norm(perpendicular, axis=1)) > cfg.edge_overlap_distance_mm:
                continue
            b0, b1 = sorted(projected)
            low, high = max(0., b0), min(intersection["length_mm"], b1)
            if high - low < cfg.edge_overlap_min_length_mm:
                continue
            parameters = sorted(((low - projected[0]) / c, (high - projected[0]) / c))
            masks.append((max(0., parameters[0]), min(boundary["length_mm"], parameters[1])))
            sources.append(intersection["edge_id"])
        merged = []
        for low, high in sorted(masks):
            if merged and low <= merged[-1][1] + 1e-8:
                merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
            else:
                merged.append((low, high))
        if not merged:
            boundary["masked_by_intersection_ids"] = []
            continue
        active, cursor = [], 0.
        for low, high in merged:
            if low > cursor + 1e-8:
                active.append((cursor, low))
            cursor = max(cursor, high)
        if cursor < boundary["length_mm"] - 1e-8:
            active.append((cursor, boundary["length_mm"]))
        boundary.update(masked_intervals_mm=merged, active_intervals_mm=active,
                        masked_by_intersection_ids=sources)
        if not active:
            boundary.update(valid=False, status="INACTIVE", failure_reason="duplicate_of_intersection")
        if merged:
            diagnostics.append({"edge_id": boundary["edge_id"], "intersection_ids": sources,
                                "masked_intervals_mm": merged, "active_intervals_mm": active})
    return diagnostics


def build_validator_edge_graph(points_mm, query_mm, support, cfg):
    """Build finite hull/strip and TLS edges without point-support validation."""
    started = perf_counter()
    points = np.asarray(points_mm, dtype=np.float64)
    query = np.asarray(query_mm, dtype=np.float64)
    patches = [patch for patch in support["plane_patches"] if patch["valid"]]
    candidates, pair_diagnostics, boundary_debug = [], [], []

    def append(edge):
        if edge is not None:
            edge["source_edge_id"] = len(candidates)
            candidates.append(edge)

    for left, right in combinations(patches, 2):
        n0, n1 = left["normal"], right["normal"]
        angle = np.rad2deg(np.arccos(np.clip(abs(n0 @ n1), 0., 1.)))
        diagnostic = {"plane_ids": [left["plane_id"], right["plane_id"]],
                      "angle_deg": float(angle), "failure_reason": None,
                      "strip_half_width_mm": cfg.intersection_strip_half_width_mm,
                      "hull_line_min_distance_mm": [None, None],
                      "exact_interval_valid": [False, False],
                      "tolerance_interval_valid": [False, False],
                      "interval_exact_mm": [None, None], "interval_tolerance_mm": [None, None],
                      "common_interval_mm": None, "edge_length_mm": None,
                      "support_passed": None, "support_rejection_reason": None,
                      "point_support_checked": False,
                      "recovered_by_tolerance": False}
        pair_diagnostics.append(diagnostic)
        if angle < cfg.plane_parallel_angle_deg:
            diagnostic["failure_reason"] = "parallel_planes"
            continue
        direction = np.cross(n0, n1)
        direction /= np.linalg.norm(direction)
        matrix = np.vstack((n0, n1))
        origin = np.linalg.lstsq(matrix, -np.array([left["offset_mm"], right["offset_mm"]]), rcond=None)[0]
        origin += ((query - origin) @ direction) * direction
        diagnostic.update(intersection_line_origin_mm=origin.copy(),
                          intersection_line_direction=direction.copy())
        exact = [_hull_line_interval(patch, origin, direction) for patch in (left, right)]
        strip = [_hull_strip_interval(patch, origin, direction, cfg.intersection_strip_half_width_mm)
                 for patch in (left, right)]
        # Zero width retains the original exact clipping, including its tolerances.
        intervals = exact if cfg.intersection_strip_half_width_mm == 0 else [item[0] for item in strip]
        diagnostic.update(interval_exact_mm=exact, interval_tolerance_mm=intervals,
                          hull_line_min_distance_mm=[item[1] for item in strip],
                          exact_interval_valid=[item is not None for item in exact],
                          tolerance_interval_valid=[item is not None for item in intervals])
        if any(interval is None for interval in intervals):
            diagnostic["failure_reason"] = "intersection_outside_hull"
            continue
        low, high = max(interval[0] for interval in intervals), min(interval[1] for interval in intervals)
        diagnostic.update(support_common_interval_mm=(low, high) if high > low else None,
                          common_interval_mm=None,
                          support_edge_length_mm=float(max(0., high - low)), edge_length_mm=None)
        if high <= low or high - low < cfg.edge_min_length_mm:
            diagnostic["failure_reason"] = "no_finite_overlap"
            continue
        exact_passed = False
        if all(interval is not None for interval in exact):
            exact_low, exact_high = max(item[0] for item in exact), min(item[1] for item in exact)
            if exact_high > exact_low:
                exact_core = _core_segment_interval(origin + exact_low * direction,
                                                    origin + exact_high * direction,
                                                    query, cfg.ransac_radius_mm)
                exact_passed = exact_core is not None and exact_core[1] - exact_core[0] >= cfg.edge_min_length_mm
        diagnostic["recovered_by_tolerance"] = bool(cfg.intersection_strip_half_width_mm > 0 and not exact_passed)
        edge = _edge(points, (left, right), origin + low * direction, origin + high * direction,
                     "plane_intersection", query, support["support_radius_mm"], cfg)
        if edge is None:
            diagnostic["failure_reason"] = "outside_core_or_short_clipped_edge"
            continue
        diagnostic.update(common_interval_mm=(float((edge["start_mm"] - origin) @ direction),
                                              float((edge["end_mm"] - origin) @ direction)),
                          edge_length_mm=edge["length_mm"])
        diagnostic["failure_reason"] = edge["failure_reason"]
        edge.update({key: value for key, value in diagnostic.items()
                     if key not in ("plane_ids", "angle_deg", "failure_reason")})
        append(edge)
    for patch in patches:
        lines, debug = _boundary_segments(patch, cfg, query)
        boundary_debug.append(debug)
        for line in lines:
            edge = _edge(points, (patch,), line["start_mm"], line["end_mm"], "boundary", query,
                         support["support_radius_mm"], cfg, line=line)
            line["kept_in_graph"] = edge is not None
            if edge is not None:
                for side in ("start", "end"):
                    line[side + "_node"] = edge[side + "_node"]
                    line[side + "_node_id"] = edge[side + "_node_id"]
                    line[side + "_mm"] = edge[side + "_mm"]
            append(edge)
    edges = [{**edge, "edge_id": index, "source_edge_ids": [edge["source_edge_id"]]}
             for index, edge in enumerate(sorted(
                 (edge for edge in candidates if edge["valid"]),
                 key=lambda edge: (-edge["support_quality"], -edge["length_mm"], edge["source_edge_id"])))]
    overlap_diagnostics = _mask_boundary_overlaps(edges, cfg)
    all_edges = edges
    edges = [edge for edge in all_edges if edge["valid"]]
    adjacency = [{"plane_ids": edge["plane_ids"].copy(), "edge_id": edge["edge_id"]}
                 for edge in edges if edge["edge_type"] == "plane_intersection"]
    return {"edges": edges, "edge_candidates": candidates, "plane_adjacency": adjacency,
            "boundary_debug": boundary_debug,
            "all_edges": all_edges, "boundary_overlap_diagnostics": overlap_diagnostics,
            "plane_pair_diagnostics": pair_diagnostics, "elapsed_s": perf_counter() - started}
