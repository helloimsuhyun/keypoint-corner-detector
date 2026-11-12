"""Python port of Merium88's edgedetection.m / corner_detection.m.

Reference commit: e8aa7a4254cf67c3bdd6a82a7efb67d51c47994c.
All coordinates here are metres. MATLAB toolbox bitwise parity is unverified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
from scipy.io import savemat
from scipy.spatial import cKDTree

if __package__:
    from .config import AhmedConfig, AUTHOR_AHMED
else:
    from config import AhmedConfig, AUTHOR_AHMED


def grid_average(points, step):
    """Average each bounding-box-anchored voxel; enumerate X fastest."""
    cells = np.floor((points - points.min(axis=0)) / step).astype(np.int64)
    _, inverse, counts = np.unique(cells[:, ::-1], axis=0, return_inverse=True, return_counts=True)
    return np.column_stack([np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)])


def knn(points, queries, k):
    """Include self, sort by distance then input ID, including boundary ties."""
    if len(points) < k:
        raise ValueError(f"Ahmed {k}-NN stage requires at least {k} points, got {len(points)}")
    tree = cKDTree(points)
    search_k = min(k + 1, len(points))
    distances, indices = tree.query(queries, k=search_k)
    distances = distances.reshape(len(queries), search_k)
    indices = indices.reshape(len(queries), search_k)
    for row in range(len(queries)):
        ids = indices[row]
        if search_k > k and distances[row, k - 1] == distances[row, k]:
            ids = np.asarray(tree.query_ball_point(queries[row], np.nextafter(distances[row, k], np.inf)))
        squared = np.sum((points[ids] - queries[row]) ** 2, axis=1)
        order = np.lexsort((ids, squared))[:k]
        indices[row, :k] = ids[order]
        distances[row, :k] = np.sqrt(squared[order])
    return indices[:, :k], distances[:, :k]


def pca_vectors(neighborhoods, component):
    # edgedetection subtracts zscore's mean, then pca centers a second time.
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    centered -= centered.mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    vectors = vt[:, component, :].copy()
    largest = np.argmax(np.abs(vectors), axis=1)
    vectors *= np.sign(vectors[np.arange(len(vectors)), largest])[:, None]
    return vectors


def remove_duplicate_thinned_points(points):
    # MATLAB deletes every member of an equal-position pair, not just repeats.
    _, inverse, counts = np.unique(points, axis=0, return_inverse=True, return_counts=True)
    return points[counts[inverse] == 1]


def edge_detection(cloud, cfg=AUTHOR_AHMED):
    indices, distances = knn(cloud, cloud, cfg.edge_neighbors)
    centroid = cloud[indices].mean(axis=1)
    displacement = np.linalg.norm(cloud - centroid, axis=1)
    edge_ids = np.flatnonzero(displacement > distances[:, 1] * cfg.edge_threshold)
    stages = {"initial_cloud_m": cloud.copy(), "edge_initial_indices": edge_ids,
              "centroid_distance_m": displacement, "nearest_neighbor_distance_m": distances[:, 1]}
    if not len(edge_ids):
        return {**stages, "cloud_m": cloud.copy(), "edge_indices": edge_ids,
                "edge_points_m": np.empty((0, 3)), "curvature_vectors": np.empty((0, 3)),
                "surface_normals": np.empty((0, 3))}
    edge_cloud = cloud[edge_ids]
    indices, _ = knn(edge_cloud, edge_cloud, cfg.thinning_neighbors)
    thin_edge = remove_duplicate_thinned_points(edge_cloud[indices].mean(axis=1))
    if not len(thin_edge):
        return {**stages, "cloud_m": cloud.copy(), "edge_indices": np.empty(0, dtype=np.int64),
                "edge_points_m": np.empty((0, 3)), "curvature_vectors": np.empty((0, 3)),
                "surface_normals": np.empty((0, 3))}
    edge_cloud = grid_average(thin_edge, cfg.voxel_mm / 1000)
    cloud = grid_average(cloud, cfg.voxel_mm / 2000)
    indices, _ = knn(cloud, edge_cloud, 1)
    edge_ids = indices[:, 0]
    edge_points = cloud[edge_ids]
    indices, _ = knn(edge_points, edge_points, cfg.pca_neighbors)
    curve = pca_vectors(edge_points[indices], 0)
    indices, _ = knn(cloud, cloud, cfg.pca_neighbors)
    normals = pca_vectors(cloud[indices], 2)
    return {**stages, "cloud_m": cloud, "edge_indices": edge_ids, "edge_points_m": edge_points,
            "curvature_vectors": curve, "surface_normals": normals}


def kmeans_plus(points, k, rng):
    """Squared-Euclidean k-means++, one replicate, singleton empty clusters."""
    centers = [points[int(rng.random_sample() * len(points))].copy()]
    for _ in range(1, k):
        squared = np.min(np.sum((points[:, None] - np.asarray(centers)) ** 2, axis=2), axis=1)
        total = squared.sum()
        index = (int(np.searchsorted(np.cumsum(squared), rng.random_sample() * total, side="right"))
                 if total > 0 else int(rng.random_sample() * len(points)))
        centers.append(points[min(index, len(points) - 1)].copy())
    centers = np.asarray(centers)
    labels = np.full(len(points), -1, dtype=np.int64)
    for _ in range(100):
        squared = np.sum((points[:, None] - centers) ** 2, axis=2)
        assigned = np.argmin(squared, axis=1)
        valid = labels >= 0
        tied = valid & (squared[np.arange(len(points)), np.maximum(labels, 0)] == squared.min(axis=1))
        assigned[tied] = labels[tied]
        counts = np.bincount(assigned, minlength=k)
        for cluster in np.flatnonzero(counts == 0):
            eligible = counts[assigned] > 1
            farthest = np.argmax(np.where(eligible, squared[np.arange(len(points)), assigned], -np.inf))
            counts[assigned[farthest]] -= 1
            assigned[farthest] = cluster
            counts[cluster] += 1
        centers = np.asarray([points[assigned == j].mean(axis=0) for j in range(k)])
        if np.array_equal(assigned, labels):
            return assigned
        labels = assigned
    warnings.warn("Ahmed k-means did not converge in 100 iterations", RuntimeWarning)
    return labels


def cluster_corner(curve, labels, level, angle_min_deg=50., angle_max_deg=150.):
    groups = [curve[labels == j] for j in range(int(labels.max()) + 1)]
    means = [group.mean(axis=0) for group in groups]
    sizes = np.array([len(group) for group in groups])
    pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    balanced = any(abs(int(sizes[i]) - int(sizes[j])) < level for i, j in pairs)
    angles = [np.arctan2(np.linalg.norm(np.cross(means[i], means[j])), means[i] @ means[j]) * 180 / 3.142
              for i, j in pairs]
    return balanced and all(angle_min_deg < angle < angle_max_deg for angle in angles)


def corner_detection(edge_points, curve, seed=1, cfg=AUTHOR_AHMED):
    if not len(edge_points):
        return np.zeros((10, 0), dtype=bool)
    neighbors, _ = knn(edge_points, edge_points, cfg.corner_neighbors)
    rng = np.random.RandomState(seed)
    masks = np.zeros((10, len(edge_points)), dtype=bool)
    for level in range(1, 11):
        for i, ids in enumerate(neighbors):
            patch = edge_points[ids]
            dimensions = np.count_nonzero(np.ptp(patch, axis=0) > cfg.span_mm / 1000)
            if dimensions >= 2:
                labels = kmeans_plus(curve[ids], dimensions, rng)
                masks[level - 1, i] = cluster_corner(curve[ids], labels, level,
                                                    cfg.angle_min_deg, cfg.angle_max_deg)
    return masks


def run_ahmed(points_m, seed=1, cfg=AUTHOR_AHMED):
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Ahmed input must be finite nonempty (N,3) XYZ in metres")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be a uint32 integer")
    cloud = grid_average(points, cfg.voxel_mm / 1000)
    stages = edge_detection(cloud, cfg)
    masks = corner_detection(stages["edge_points_m"], stages["curvature_vectors"], seed, cfg)
    return {**stages, "level_masks": masks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--level", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--voxel-mm", type=float, default=10.)
    parser.add_argument("--edge-neighbors", type=int, default=400)
    parser.add_argument("--edge-threshold", type=float, default=4.5)
    parser.add_argument("--thinning-neighbors", type=int, default=150)
    parser.add_argument("--corner-neighbors", type=int, default=20)
    parser.add_argument("--pca-neighbors", type=int, default=10)
    parser.add_argument("--span-mm", type=float, default=5.)
    parser.add_argument("--angle-min-deg", type=float, default=50.)
    parser.add_argument("--angle-max-deg", type=float, default=150.)
    args = parser.parse_args(argv)
    cfg = AhmedConfig(args.voxel_mm, args.edge_neighbors, args.edge_threshold,
                      args.thinning_neighbors, args.corner_neighbors, args.pca_neighbors, args.span_mm,
                      args.angle_min_deg, args.angle_max_deg)
    result = run_ahmed(np.loadtxt(args.input, ndmin=2), args.seed, cfg)
    edge = result["edge_points_m"]
    levels = np.empty((1, 10), dtype=object)
    for i, mask in enumerate(result["level_masks"]):
        levels[0, i] = edge[mask]
    output = Path(args.output)
    np.savetxt(output, levels[0, args.level - 1], delimiter=",", header="x_m,y_m,z_m", comments="", fmt="%.17g")
    savemat(output.parent / "ahmed_all_levels.mat", {"predictions": levels, "level": args.level, "seed": args.seed,
                                                   "implementation": "python_port"})
    np.savez_compressed(output.parent / "ahmed_stages.npz", **result)
    print(f"Ahmed Python port: {len(result['initial_cloud_m'])} initial voxels, "
          f"{len(result['edge_initial_indices'])} edge candidates, {len(edge)} thinned edges", flush=True)
    print("Corner counts L=1..10:", result["level_masks"].sum(axis=1).tolist(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
