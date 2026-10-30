"""Required synthetic coverage for the whole-cloud CED corner detector."""

from __future__ import annotations

import unittest

import numpy as np

from corner_detector.candidate_extraction import extract_corner_candidates
from corner_detector.edge_suppression import dense_ced_weighted_pca_scores


def _unique(points):
    return np.unique(np.asarray(points, dtype=np.float64), axis=0)


def _plane(noise=0.0, seed=0):
    grid = np.arange(-8.0, 8.01)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    if noise:
        points += np.random.default_rng(seed).normal(0.0, noise, points.shape)
    return points


def _edge(noise=0.0, seed=0):
    axis = np.arange(-8.0, 8.01)
    side = np.arange(0.0, 8.01)
    x, y = np.meshgrid(axis, side, indexing="ij")
    xy = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    x, z = np.meshgrid(axis, side, indexing="ij")
    xz = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    points = _unique(np.vstack((xy, xz)))
    if noise:
        points += np.random.default_rng(seed).normal(0.0, noise, points.shape)
    return points


def _corner(face_count):
    side = np.arange(0.0, 10.01)
    a, b = np.meshgrid(side, side, indexing="ij")
    faces = [np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size)))]
    if face_count >= 2:
        faces.append(np.column_stack((a.ravel(), np.zeros(a.size), b.ravel())))
    if face_count >= 3:
        faces.append(np.column_stack((np.zeros(a.size), a.ravel(), b.ravel())))
    return _unique(np.vstack(faces))


def _uneven_corner():
    points = _corner(3)
    rng = np.random.default_rng(19)
    keep_probability = np.where(np.max(points, axis=1) <= 5.0, 1.0, 0.4)
    keep = rng.random(len(points)) < keep_probability
    keep |= np.linalg.norm(points, axis=1) <= 2.1
    return points[keep]


class CedCornerSyntheticCoverageTest(unittest.TestCase):
    def test_required_structures_keep_index_and_numeric_contracts(self):
        cases = {
            "plane": _plane(),
            "straight-edge": _edge(),
            "1-face-corner": _corner(1),
            "2-face-corner": _corner(2),
            "3-face-corner": _corner(3),
            "noisy-plane": _plane(0.12, 3),
            "noisy-edge": _edge(0.12, 5),
            "uneven-density": _uneven_corner(),
        }
        for name, points in cases.items():
            with self.subTest(structure=name):
                scores = dense_ced_weighted_pca_scores(points, 2.5, 5.0, 4)
                self.assertTrue(np.all(np.isfinite(scores["ced_score"])))
                self.assertTrue(np.all(np.isfinite(scores["bilateral_ced_score"])))
                self.assertTrue(np.all(np.isfinite(scores["pca_ratio_raw"])))
                self.assertTrue(np.all(np.isfinite(scores["pca_ratio_saturated"])))
                self.assertTrue(np.all(np.isfinite(scores["fusion_score"])))
                np.testing.assert_array_equal(
                    scores["pca_query_mask"],
                    scores["ced_valid_mask"] & scores["bilateral_valid_mask"],
                )
                extraction = extract_corner_candidates(points, scores, 2.5)
                indices = extraction["candidate_indices"]
                self.assertTrue(np.all((0 <= indices) & (indices < len(points))))
                if "corner" in name:
                    corner = np.linalg.norm(points, axis=1) <= 2.1
                    self.assertTrue(np.any(scores["ced_valid_mask"] & corner))
                    self.assertTrue(np.any(extraction["candidate_mask"] & corner))


if __name__ == "__main__":
    unittest.main()
