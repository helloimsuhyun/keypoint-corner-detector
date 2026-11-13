"""Normal second-moment Harris 3-D responses on observed points (mm)."""

import numpy as np
from scipy.spatial import cKDTree


def normal_harris_response(normals):
    """PCL HARRIS equation, using the uncentered mean of unit normal products."""
    moment = normals.T @ normals / len(normals)
    return float(.04 + np.linalg.det(moment) - .04 * np.trace(moment)**2)


def compute_harris3d(points_mm, radius_mm, *, query_indices=None, min_neighbors=3,
                     is_cancelled=None):
    """Estimate PCA normals, then Harris responses; undefined support stays NaN."""
    xyz = np.asarray(points_mm, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Harris requires finite (N,3) points in mm')
    if not np.isfinite(radius_mm) or radius_mm <= 0 or min_neighbors < 3:
        raise ValueError('Harris radius must be finite >0; minimum support must be >=3')
    ids = np.arange(len(xyz)) if query_indices is None else np.asarray(query_indices)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= len(xyz)):
        raise ValueError('Harris query indices must index the supplied points')
    tree = cKDTree(xyz)
    normals = np.full_like(xyz, np.nan)
    for i, point in enumerate(xyz):
        if i % 256 == 0 and is_cancelled is not None and is_cancelled():
            return None
        neighbors = tree.query_ball_point(point, radius_mm)
        if len(neighbors) < min_neighbors:
            continue
        centered = xyz[neighbors] - xyz[neighbors].mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(neighbors))
        if eigenvalues[1] > 1e-12 * eigenvalues[2] and eigenvalues[2] > 0:
            normals[i] = eigenvectors[:, 0]
    responses = np.full(len(ids), np.nan)
    valid = np.isfinite(normals).all(axis=1)
    for row, i in enumerate(ids):
        if row % 256 == 0 and is_cancelled is not None and is_cancelled():
            return None
        if not valid[i]:
            continue
        neighbors = np.asarray(tree.query_ball_point(xyz[i], radius_mm), dtype=np.int64)
        neighbors = neighbors[valid[neighbors]]
        if len(neighbors) >= min_neighbors:
            responses[row] = normal_harris_response(normals[neighbors])
    return responses
