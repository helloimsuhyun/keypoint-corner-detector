"""Numeric contracts of the Python port of the author's MATLAB pipeline."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import loadmat

from corner_dataset_tool.baselines.ahmed import (
    cluster_corner, corner_detection, grid_average, kmeans_plus, knn,
    pca_vectors, remove_duplicate_thinned_points, run_ahmed,
)
from corner_dataset_tool.run_baselines import main
from corner_dataset_tool.baselines.config import AhmedConfig


class AhmedPythonTest(unittest.TestCase):
    def test_tunable_angle_bounds_keep_strict_original_predicate(self):
        curve = np.eye(3)[:2]
        labels = np.arange(2)
        angle = 90 * np.pi / 3.142
        self.assertTrue(cluster_corner(curve, labels, 1))
        self.assertFalse(cluster_corner(curve, labels, 1, angle, 150))
        self.assertFalse(cluster_corner(curve, labels, 1, 50, angle))
        self.assertTrue(cluster_corner(curve, labels, 1, 80, 100))

    def test_tunable_patch_span_changes_only_eligibility(self):
        edge = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0]])
        small = corner_detection(edge, np.eye(3), cfg=AhmedConfig(corner_neighbors=3))
        large = corner_detection(edge, np.eye(3), cfg=AhmedConfig(corner_neighbors=3, span_mm=10))
        self.assertFalse(small[0].any())
        self.assertTrue(small[1:].all())
        self.assertFalse(large.any())

    def line(self):
        return np.column_stack((np.arange(400.), np.zeros(400), np.ones(400)))

    def test_grid_average_bounds_centroids_and_x_fastest_order(self):
        points = np.array([[-.5, 0, 1], [-.375, 0, 1], [-.25, 0, 1], [-.5, .25, 1], [0, 0, 1]])
        np.testing.assert_array_equal(grid_average(points, .25),
                                      [[-.4375, 0, 1], [-.25, 0, 1], [0, 0, 1], [-.5, .25, 1]])

    def test_knn_includes_self_and_resolves_ties_by_smallest_input_index(self):
        points = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0], [0, 0, 0]])
        ids, distance = knn(points, points[4:5], 3)
        np.testing.assert_array_equal(ids, [[4, 0, 1]])
        np.testing.assert_array_equal(distance, [[0, 1, 1]])
        with self.assertRaisesRegex(ValueError, "400-NN"):
            knn(points, points, 400)

    def test_thinning_deletes_all_members_of_duplicate_positions(self):
        points = np.array([[0., 0, 0], [1., 2, 3], [0., 0, 0], [4., 5, 6], [1., 2, 3]])
        np.testing.assert_array_equal(remove_duplicate_thinned_points(points), [[4, 5, 6]])

    def test_pca_centering_and_loading_sign_match_documented_matlab_convention(self):
        axis = np.array([-1., 2., .25])
        line = np.arange(10)[:, None] * axis + [100, -50, 90]
        np.testing.assert_allclose(pca_vectors(line[None], 0)[0], axis / np.linalg.norm(axis), atol=1e-14)
        x, y = np.meshgrid(np.arange(4.), np.arange(3.))
        plane = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 5.)))
        np.testing.assert_allclose(pca_vectors(plane[None], 2)[0], [0, 0, 1], atol=1e-14)

    def test_kmeans_plus_partitions_axis_vectors_and_repairs_empty_clusters(self):
        points = np.repeat(np.eye(3), [7, 7, 6], axis=0)
        a = kmeans_plus(points, 3, np.random.RandomState(1))
        b = kmeans_plus(points, 3, np.random.RandomState(1))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), 3)
        for cluster in range(3):
            self.assertEqual(len(np.unique(points[a == cluster], axis=0)), 1)
        duplicate = kmeans_plus(np.ones((20, 3)), 3, np.random.RandomState(1))
        self.assertTrue(np.all(np.bincount(duplicate, minlength=3) > 0))

    def test_corner_uses_strict_size_gate_or_for_three_clusters_and_angle_gate(self):
        labels = np.repeat(np.arange(3), [2, 4, 14])
        curve = np.eye(3)[labels]
        self.assertFalse(cluster_corner(curve, labels, 2))
        self.assertTrue(cluster_corner(curve, labels, 3))
        labels = np.repeat([0, 1], 10)
        self.assertTrue(cluster_corner(np.eye(3)[labels], labels, 1))
        self.assertFalse(cluster_corner(np.array([[1., 0, 0], [-1., 0, 0]])[labels], labels, 10))

    def test_corner_repeats_kmeans_for_every_patch_at_every_level(self):
        points = np.column_stack((np.arange(20) * .001, np.arange(20)[::-1] * .001, np.zeros(20)))
        curve = np.repeat(np.eye(3)[:2], 10, axis=0)

        def classify(vectors, k, rng):
            self.assertEqual(k, 2)
            rng.random_sample()
            return np.argmax(vectors, axis=1)

        with patch("corner_dataset_tool.baselines.ahmed.kmeans_plus", side_effect=classify) as cluster:
            masks = corner_detection(points, curve, 1)
        self.assertEqual(cluster.call_count, 200)
        np.testing.assert_array_equal(masks, np.ones((10, 20), dtype=bool))

    def test_full_pipeline_line_has_exact_strict_edges_and_no_corners(self):
        points = self.line()
        original = points.copy()
        result = run_ahmed(points, 1)
        expected = np.r_[np.arange(195), np.arange(205, 400)]
        np.testing.assert_array_equal(result["edge_initial_indices"], expected)
        np.testing.assert_array_equal(result["nearest_neighbor_distance_m"], np.ones(400))
        np.testing.assert_allclose(result["centroid_distance_m"], np.abs(np.arange(400) - 199.5))
        np.testing.assert_allclose(result["curvature_vectors"], np.tile([1, 0, 0], (len(result["edge_points_m"]), 1)))
        self.assertFalse(result["level_masks"].any())
        np.testing.assert_array_equal(points, original)

    def test_bad_input_and_insufficient_support_are_explicit_failures(self):
        with self.assertRaisesRegex(ValueError, "400-NN"):
            run_ahmed(self.line()[:399])
        invalid = self.line()
        invalid[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            run_ahmed(invalid)
        with self.assertRaisesRegex(ValueError, "uint32"):
            run_ahmed(self.line(), -1)

    def test_runner_executes_python_without_matlab_or_upstream_setup_and_saves_all_levels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = root / "scene_0001"
            scene.mkdir()
            np.savez(scene / "points.npz", xyz_mm=self.line().astype(np.float32) * 1000,
                     rgb=np.zeros((400, 3), np.uint8), pixel_index=np.arange(400))
            original = (scene / "points.npz").read_bytes()
            output = root / "results"
            with patch("corner_dataset_tool.run_baselines.shutil.which", side_effect=AssertionError("MATLAB must not be checked")), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = main([str(scene), "--method", "ahmed", "--mode", "scene", "--preset", "author", "--output", str(output)])
            self.assertEqual(status, 0)
            directory = output / "scene_0001/scene/ahmed"
            info = json.loads((directory / "run.json").read_text())
            self.assertEqual(info["source"]["implementation"], "python_port")
            self.assertFalse(info["source"]["matlab_parity_verified"])
            self.assertEqual(info["prediction_count"], 0)
            self.assertTrue(info["commands"][0][1].endswith("ahmed.py"))
            levels = loadmat(directory / "ahmed_all_levels.mat")["predictions"]
            self.assertEqual(levels.shape, (1, 10))
            self.assertTrue(all(levels[0, i].shape == (0, 3) for i in range(10)))
            with np.load(directory / "ahmed_stages.npz") as data:
                self.assertEqual(len(data["edge_initial_indices"]), 390)
            self.assertEqual((scene / "points.npz").read_bytes(), original)
            with patch("corner_dataset_tool.run_baselines.shutil.which", return_value=None), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main([str(scene), "--method", "ahmed", "--mode", "scene", "--ahmed-backend", "matlab",
                      "--output", str(root / "matlab_missing")])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root / "matlab_missing").exists())


if __name__ == "__main__":
    unittest.main()
