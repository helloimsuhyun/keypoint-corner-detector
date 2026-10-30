"""Query-local 3-D vertex hypotheses. Does not generate or move CED queries."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .settings import DEFAULTS, ExtremityConfig
from .extremity_voting import point_direction_diagnostics, soft_direction_votes


@lru_cache(maxsize=8)
def fibonacci_directions(count=DEFAULTS.ball_directions):
    """Midpoint Fibonacci sphere; deterministic and without duplicated poles."""
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("Direction count must be a positive integer")
    k = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (k + 0.5) / count
    theta = k * (np.pi * (3.0 - np.sqrt(5.0)))
    radial = np.sqrt(1.0 - z * z)
    directions = np.column_stack((radial * np.cos(theta), radial * np.sin(theta), z))
    directions.setflags(write=False)
    return directions


def _points(points_mm):
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_mm must be finite with shape (N, 3)")
    return points


def prepare_extremity_context(scene_points_mm, *, raw_points_mm=None):
    """Cache spatial indices, not segmentation; components are query-local."""
    started = perf_counter()
    scene = _points(scene_points_mm)
    raw = scene if raw_points_mm is None else _points(raw_points_mm)
    raw_tree = cKDTree(raw)
    scene_tree = raw_tree if raw is scene else cKDTree(scene)
    return {"scene_points_mm": scene, "raw_points_mm": raw,
            "raw_tree": raw_tree, "scene_tree": scene_tree,
            "preparation_s": perf_counter() - started}


def _components(points, radius):
    if not len(points):
        return np.empty(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8),
                        (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    return connected_components(graph, directed=False)[1].astype(np.int64)


def _local_peaks(points, scores, votes, labels, cfg):
    """Component-aware NMS; equal-score neighbors prefer smaller raw index.

    Points arrive sorted by original raw index. Local maxima compare all eligible
    neighbors; greedy compares only already retained peaks. Neither moves peaks.
    """
    eligible = (votes > 0) & (scores >= cfg.min_vote_fraction)
    indices = np.flatnonzero(eligible)
    order = indices[np.lexsort((indices, -scores[indices]))]
    tree = cKDTree(points)
    suppressed = np.zeros(len(points), dtype=bool)
    selected = []
    for index in order:
        if cfg.nms_method == "greedy" and suppressed[index]:
            continue
        local = np.asarray(tree.query_ball_point(points[index], cfg.nms_radius_mm), dtype=np.int64)
        local = local[eligible[local] & (labels[local] == labels[index])]
        if cfg.nms_method == "local_maximum" and np.any(
            (scores[local] > scores[index]) |
            ((scores[local] == scores[index]) & (local < index))
        ):
            continue
        selected.append(index)
        suppressed[local] = True
    return np.asarray(selected, dtype=np.int64)


def run_extremity_refinement(scene_points_mm, query_index, cfg=None, *,
                             raw_points_mm=None, context=None, oracle_labels=None):
    """Return multiple observed vertex hypotheses for one unchanged scene query.

    Voting uses raw points, while point_index/candidate_indices target the
    supplied voxel scene for validator compatibility. raw_point_index and
    candidate_raw_indices explicitly address the raw cloud. No cross-query vote
    aggregation is performed. Every local component gets its own direction sweep.
    """
    started = perf_counter()
    cfg = ExtremityConfig() if cfg is None else cfg
    cfg.validate()
    scene = _points(scene_points_mm)
    if not isinstance(query_index, (int, np.integer)) or not 0 <= query_index < len(scene):
        raise IndexError("query_index must address scene_points_mm")
    cached = context is not None
    if context is None:
        context = prepare_extremity_context(scene, raw_points_mm=raw_points_mm)
    elif not np.array_equal(context["scene_points_mm"], scene):
        raise ValueError("Context belongs to a different scene")
    raw = context["raw_points_mm"]
    if raw_points_mm is not None and not np.array_equal(raw, raw_points_mm):
        raise ValueError("Context belongs to a different raw cloud")
    q = scene[query_index]
    stage = perf_counter()
    ids = np.asarray(context["raw_tree"].query_ball_point(
        q, max(cfg.radii_mm) + cfg.padding_mm, return_sorted=True), dtype=np.int64)
    roi = raw[ids]
    delta = roi - q
    distance2 = np.einsum("ij,ij->i", delta, delta)
    roi_s = perf_counter() - stage
    stage = perf_counter()
    if not cfg.object_clustering:
        labels = np.zeros(len(roi), dtype=np.int64)
        method = "off"
    elif oracle_labels is not None:
        oracle = np.asarray(oracle_labels)
        if oracle.shape != (len(raw),) or oracle.dtype.kind not in "iu" or np.any(oracle < 0):
            raise ValueError("oracle_labels must be nonnegative integers aligned with raw points")
        _, labels = np.unique(oracle[ids], return_inverse=True)
        method = "oracle"
    else:
        labels = _components(roi, cfg.cluster_radius_mm)
        method = "euclidean_components"
    clustering_s = perf_counter() - stage
    directions = fibonacci_directions(cfg.directions)
    hard = cfg.voting_mode == "hard"
    vote_dtype = np.int64 if hard else np.dtype(cfg.voting_dtype)
    votes = np.zeros(len(roi), dtype=vote_dtype)
    scale_support = np.zeros(len(roi), dtype=np.int64)
    opportunities = np.zeros(len(roi), dtype=np.int64)
    votes_per_scale = np.zeros((len(cfg.radii_mm), len(roi)), dtype=vote_dtype)
    detail = dict.fromkeys(("projection", "quantile", "sigmoid", "normalization", "accumulation"), 0.0)
    evaluated = possible = 0
    voting_s = aggregation_s = 0.0
    sweep_count = 0
    discarded = 0
    for si, radius in enumerate(cfg.radii_mm):
        stage = perf_counter()
        search = distance2 <= (radius + cfg.padding_mm) ** 2
        core = distance2 <= radius * radius
        for component in np.unique(labels[search]):
            local = np.flatnonzero(search & (labels == component))
            if hard:
                projection_start = perf_counter()
                projections = (delta[local].astype(cfg.voting_dtype, copy=False)
                               @ directions.astype(cfg.voting_dtype, copy=False).T)
                detail["projection"] += perf_counter() - projection_start
                winners = local[np.argmax(projections, axis=0)]
                accepted = winners[core[winners]]
                discarded += cfg.directions - len(accepted)
            else:
                local_votes, stages, used, total = soft_direction_votes(delta[local], directions, cfg)
                for key in detail:
                    detail[key] += stages[key]
                evaluated += used
                possible += total
                accepted = local[core[local]]
                discarded += float(local_votes[~core[local]].sum())
            sweep_count += 1
            voting_s += perf_counter() - stage
            stage = perf_counter()
            if hard:
                np.add.at(votes_per_scale[si], accepted, 1)
            else:
                votes_per_scale[si, accepted] += local_votes[core[local]]
            accumulation_s = perf_counter() - stage
            aggregation_s += accumulation_s
            detail["accumulation"] += accumulation_s
            stage = perf_counter()
        voting_s += perf_counter() - stage
        stage = perf_counter()
        opportunities[core] += 1
        scale_support += votes_per_scale[si] > 0
        votes += votes_per_scale[si]
        accumulation_s = perf_counter() - stage
        aggregation_s += accumulation_s
        detail["accumulation"] += accumulation_s
    stage = perf_counter()
    scores = np.divide(votes, cfg.directions * opportunities,
                       out=np.zeros(len(roi), dtype=float), where=opportunities > 0)
    nms_start = perf_counter()
    selected = _local_peaks(roi, scores, votes, labels, cfg)
    nms_s = perf_counter() - nms_start
    candidate_raw = ids[selected]
    if len(selected):
        mapping_distance, candidate_scene = context["scene_tree"].query(roi[selected])
        candidate_scene = np.asarray(candidate_scene, dtype=np.int64)
    else:
        mapping_distance = np.empty(0)
        candidate_scene = np.empty(0, dtype=np.int64)
    candidates = [
        {"point_index": int(scene_id), "raw_point_index": int(raw_id),
         "roi_point_index": int(local_id), "position_mm": raw[raw_id].copy(),
         "query_index": int(query_index), "query_position_mm": q.copy(),
         "query_offset_mm": raw[raw_id] - q,
         "query_distance_mm": float(np.linalg.norm(raw[raw_id] - q)),
         "scene_mapping_distance_mm": float(map_distance),
         "score": float(scores[local_id]),
         "raw_votes": int(votes[local_id]) if hard else float(votes[local_id]),
         "normalized_score": float(scores[local_id]), "cluster_id": int(labels[local_id]),
         "scale_support_count": int(scale_support[local_id]),
         "status": "vertex_hypothesis_unverified", "verified": False}
        for local_id, raw_id, scene_id, map_distance in
        zip(selected, candidate_raw, candidate_scene, mapping_distance)
    ]
    aggregation_s += perf_counter() - stage
    total_s = perf_counter() - started
    return {"algorithm": "local_extremity256", "config": cfg, "voting_mode": cfg.voting_mode,
            "query_index": int(query_index), "query_position_mm": q.copy(),
            "roi_points_mm": roi, "roi_raw_indices": ids, "raw_votes": votes,
            "normalized_score": scores, "opportunity_count": opportunities,
            "scale_support_count": scale_support, "votes_per_scale": votes_per_scale,
            "cluster_labels": labels, "cluster_method": method,
            "cluster_count": len(np.unique(labels)), "direction_sweep_count": sweep_count,
            "discarded_vote_count": int(discarded) if hard else None,
            "discarded_vote_mass": float(discarded),
            "sigmoid_evaluated_count": evaluated, "sigmoid_possible_count": possible,
            "candidate_count": len(selected),
            "candidate_local_indices": selected, "candidate_raw_indices": candidate_raw,
            "candidate_indices": candidate_scene, "validator_seed_indices": np.unique(candidate_scene),
            "candidates": candidates, "nms_method": cfg.nms_method,
            "timings_seconds": {"preparation": 0.0 if cached else context["preparation_s"],
                               "roi": roi_s, "clustering": clustering_s, "voting": voting_s,
                               "aggregation_nms": aggregation_s, "nms": nms_s,
                               **detail, "total": total_s},
            "context_cached": cached}


def inspect_point_directions(result, roi_point_index, *, scale_index=0):
    """Read-only Q99 membership and actual directional mass for one observed point."""
    cfg = result["config"]
    roi = result["roi_points_mm"]
    if not isinstance(roi_point_index, (int, np.integer)) or not 0 <= roi_point_index < len(roi):
        raise IndexError("roi_point_index must address the stored raw ROI")
    if not isinstance(scale_index, (int, np.integer)) or not 0 <= scale_index < len(cfg.radii_mm):
        raise IndexError("scale_index must address the configured radii")
    radius = cfg.radii_mm[scale_index]
    delta = roi - result["query_position_mm"]
    distance2 = np.einsum("ij,ij->i", delta, delta)
    cluster = result["cluster_labels"][roi_point_index]
    support = np.flatnonzero((distance2 <= (radius + cfg.padding_mm) ** 2) &
                             (result["cluster_labels"] == cluster))
    directions = fibonacci_directions(cfg.directions)
    inside_search = bool(distance2[roi_point_index] <= (radius + cfg.padding_mm) ** 2)
    inside_core = bool(distance2[roi_point_index] <= radius ** 2)
    member = np.zeros(cfg.directions, dtype=bool)
    contribution = np.zeros(cfg.directions, dtype=cfg.voting_dtype)
    projection = threshold = None
    if inside_search:
        column = int(np.searchsorted(support, roi_point_index))
        projection, threshold, contribution = point_direction_diagnostics(delta[support], directions, cfg, column)
        member = projection >= threshold
        if not inside_core:
            contribution.fill(0)
    opportunities = result["opportunity_count"][roi_point_index]
    score_contribution = (contribution.astype(float) / (cfg.directions * opportunities)
                          if opportunities else np.zeros(cfg.directions))
    return {"query_index": result["query_index"], "roi_point_index": int(roi_point_index),
            "raw_point_index": int(result["roi_raw_indices"][roi_point_index]),
            "position_mm": roi[roi_point_index].copy(), "cluster_id": int(cluster),
            "scale_index": int(scale_index), "radius_mm": radius, "padding_mm": cfg.padding_mm,
            "inside_core": inside_core,
            "inside_search": inside_search, "support_count": len(support),
            "directions": directions, "top1_mask": member, "top1_count": int(member.sum()),
            "projection": projection, "threshold": threshold,
            "direction_votes": contribution, "direction_scores": score_contribution,
            "direction_vote_sum": float(contribution.sum(dtype=np.float64)),
            "scale_raw_votes": float(result["votes_per_scale"][scale_index, roi_point_index]),
            "raw_votes": float(result["raw_votes"][roi_point_index]),
            "normalized_score": float(result["normalized_score"][roi_point_index])}


def refine_ced_candidates(scene_points_mm, candidate_indices, cfg=None, *,
                          raw_points_mm=None, oracle_labels=None):
    """One independent map per existing NMS query; no query creation or movement."""
    scene = _points(scene_points_mm)
    indices = np.asarray(candidate_indices)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError("candidate_indices must be a one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= len(scene)):
        raise IndexError("candidate_indices outside voxel scene")
    started = perf_counter()
    context = prepare_extremity_context(scene, raw_points_mm=raw_points_mm)
    records = [run_extremity_refinement(scene, int(i), cfg, context=context,
                                        oracle_labels=oracle_labels) for i in indices]
    return {"query_indices": indices.astype(np.int64, copy=True), "records": records,
            "candidate_count": sum(r["candidate_count"] for r in records),
            "preparation_s": context["preparation_s"], "elapsed_s": perf_counter() - started}


__all__ = ["ExtremityConfig", "fibonacci_directions", "prepare_extremity_context",
           "run_extremity_refinement", "refine_ced_candidates", "inspect_point_directions"]
