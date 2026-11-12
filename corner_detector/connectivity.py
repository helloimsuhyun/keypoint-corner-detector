"""Euclidean radius-graph connectivity on unchanged XYZ samples (mm)."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def radius_components(points, epsilon):
    pairs = cKDTree(points).query_pairs(epsilon, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points)))
    return connected_components(graph, directed=False)[1]
