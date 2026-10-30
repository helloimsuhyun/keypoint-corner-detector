"""Experimental post-NMS direction-distribution features; no detector mutation."""

from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .extremity import fibonacci_directions
from .extremity_voting import point_direction_diagnostics


FEATURES = ("vote_score", "top1_fraction", "effective_fraction", "resultant",
            "moment_min", "moment_directed", "coherence", "top1_connected_fraction",
            "top1_patch_directed", "shuffled_moment_directed")
SPATIAL_METHODS = ("resultant", "moment_min", "moment_directed", "coherence",
                   "top1_connected_fraction", "top1_patch_directed")


def sphere_edges(directions):
    """Symmetric six-nearest-neighbor graph in angular distance on unit directions."""
    count = len(directions)
    if count < 2:
        return np.empty((0, 2), dtype=np.int64)
    neighbors = cKDTree(directions).query(directions, k=min(7, count))[1][:, 1:]
    edges = np.column_stack((np.repeat(np.arange(count), neighbors.shape[1]), neighbors.ravel()))
    return np.unique(np.sort(edges, axis=1), axis=0)


@lru_cache(maxsize=8)
def _sampling(count):
    directions = fibonacci_directions(count)
    return directions, sphere_edges(directions)


def angular_features(mass, top1, directions, edges):
    """Features distinguish amplitude, angular spread, and connected support."""
    mass = np.asarray(mass, dtype=float)
    top1 = np.asarray(top1, dtype=bool)
    directions = np.asarray(directions, dtype=float)
    if (mass.ndim != 2 or mass.shape != top1.shape or mass.shape[1] != len(directions)
            or directions.shape != (mass.shape[1], 3) or not np.isfinite(mass).all()
            or np.any(mass < 0) or not np.isfinite(directions).all()
            or not np.allclose(np.linalg.norm(directions, axis=1), 1)):
        raise ValueError("Expected finite nonnegative K-by-J mass and J unit 3-D directions")
    total = mass.sum(axis=1, keepdims=True)
    p = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    moment = np.einsum("kj,ja,jb->kab", p, directions, directions)
    smallest = np.maximum(np.linalg.eigvalsh(moment)[:, 0], 0)
    resultant = np.linalg.norm(p @ directions, axis=1)
    effective = np.divide(1., len(directions) * np.sum(p * p, axis=1),
                          out=np.zeros(len(p)), where=total[:, 0] > 0)
    coherence = np.zeros(len(p))
    if len(edges):
        a, b = p[:, edges[:, 0]], p[:, edges[:, 1]]
        denominator = np.sum(a * a + b * b, axis=1)
        coherence = np.divide(2 * np.sum(a * b, axis=1), denominator,
                              out=coherence, where=denominator > 0)
    source = np.r_[edges[:, 0], edges[:, 1]]
    target = np.r_[edges[:, 1], edges[:, 0]]
    graph = coo_matrix((np.ones(len(source)), (source, target)),
                       shape=(len(directions), len(directions))).tocsr()
    connected_fraction, patch_directed = np.zeros(len(p)), np.zeros(len(p))
    for i, mask in enumerate(top1):
        active = np.flatnonzero(mask)
        if not len(active):
            continue
        _, labels = connected_components(graph[active][:, active], directed=False)
        largest = active[labels == np.argmax(np.bincount(labels))]
        patch = directions[largest]
        connected_fraction[i] = len(patch) / len(directions)
        area_moment = patch.T @ patch / len(directions)
        patch_directed[i] = max(np.linalg.eigvalsh(area_moment)[0], 0) * np.linalg.norm(patch.mean(axis=0))
    # Fixed permutation preserves amplitudes/counts while destroying angular placement.
    permutation = np.random.default_rng(20261029).permutation(len(directions))
    shuffled = p[:, permutation]
    shuffled_moment = np.einsum("kj,ja,jb->kab", shuffled, directions, directions)
    shuffled_score = (np.maximum(np.linalg.eigvalsh(shuffled_moment)[:, 0], 0)
                      * np.linalg.norm(shuffled @ directions, axis=1))
    return dict(top1_fraction=top1.mean(axis=1), effective_fraction=effective, resultant=resultant,
                moment_min=smallest, moment_directed=smallest * resultant, coherence=coherence,
                top1_connected_fraction=connected_fraction, top1_patch_directed=patch_directed,
                shuffled_moment_directed=shuffled_score)


def candidate_direction_features(result):
    """One extra component/scale sweep for all existing NMS candidates together."""
    cfg = result["config"]
    selected = result["candidate_local_indices"]
    directions, edges = _sampling(cfg.directions)
    mass = np.zeros((len(selected), cfg.directions))
    top1 = np.zeros_like(mass, dtype=bool)
    started = perf_counter()
    delta = result["roi_points_mm"] - result["query_position_mm"]
    distance2 = np.einsum("ij,ij->i", delta, delta)
    labels = result["cluster_labels"]
    for radius in cfg.radii_mm:
        core = distance2 <= radius ** 2
        for component in np.unique(labels[selected[core[selected]]]):
            positions = np.flatnonzero(core[selected] & (labels[selected] == component))
            support = np.flatnonzero((distance2 <= (radius + cfg.padding_mm) ** 2) & (labels == component))
            columns = np.searchsorted(support, selected[positions])
            projection, threshold, contribution = point_direction_diagnostics(delta[support], directions, cfg, columns)
            mass[positions] += contribution.T
            top1[positions] |= (projection >= threshold[:, None]).T
    trace_s = perf_counter() - started
    np.testing.assert_allclose(mass.sum(axis=1), result["raw_votes"][selected], rtol=3e-6, atol=3e-6)
    started = perf_counter()
    features = angular_features(mass, top1, directions, edges)
    features["vote_score"] = result["normalized_score"][selected].copy()
    return features, {"trace_s": trace_s, "features_s": perf_counter() - started}
