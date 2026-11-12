"""Numerical regression tests for guided validator geometry."""

from dataclasses import replace
import unittest

import numpy as np

from corner_detector.settings import GuidedValidatorConfig, configure, parse_args
from corner_detector.validator_plane_support import reconstruct_plane_support
from corner_detector.validator_edge_graph import _boundary_segments, build_validator_edge_graph
from corner_detector.validator_corner_graph import reconstruct_validator_corners


class ValidatorCornerGeometryTest(unittest.TestCase):
    def test_plane_support_selects_one_component_independently_of_corner_radius(self):
        points = np.array([[-5., 0., 0.], [-4., 0., 0.], [-4.5, 1., 0.],
                           [6., 0., 0.], [7., 0., 0.], [6.5, 1., 0.]])
        for core_ids, expected_component, valid in (
                (np.arange(6), 0, True),
                (np.arange(3, 6), 1, True),
                (np.empty(0, dtype=np.int64), 0, False)):
            plane = {"normal": np.array([0., 0., 1.]), "offset_mm": 0.,
                     "inlier_ransac_indices": core_ids.copy()}
            for gate in (1., 10.):
                with self.subTest(core_ids=core_ids.tolist(), gate=gate):
                    cfg = GuidedValidatorConfig(boundary_merge_angle_deg=30.,
                                                plane_connect_radius_mm=2., corner_radius_mm=gate)
                    support = reconstruct_plane_support(points, np.zeros(3), [plane], cfg)
                    patch = support["plane_patches"][0]
                    self.assertEqual(patch["component_count"], 2)
                    self.assertEqual(patch["component_sizes"], [3, 3])
                    self.assertEqual(patch["component_core_counts"],
                                     [len(np.intersect1d(np.arange(start, start + 3), core_ids))
                                      for start in (0, 3)])
                    self.assertEqual(patch["selected_component"], expected_component)
                    expected_ids = np.arange(3 * expected_component, 3 * expected_component + 3)
                    np.testing.assert_array_equal(patch["support_ransac_indices"], expected_ids)
                    np.testing.assert_array_equal(patch["recollected_ransac_indices"], np.arange(6))
                    np.testing.assert_allclose(np.unique(patch["hull_vertices_mm"], axis=0),
                                               np.unique(points[expected_ids], axis=0))
                    self.assertAlmostEqual(patch["hull_area_mm2"], .5)
                    self.assertEqual(patch["valid"], valid)
                    self.assertEqual(patch["connectivity_uncertain"], not valid)
                    self.assertEqual(patch["status"], "VALID" if valid else "UNKNOWN")
                    self.assertEqual(patch["failure_reason"], None if valid else "ambiguous_local_connectivity")
                    np.testing.assert_array_equal(plane["normal"], [0., 0., 1.])
                    np.testing.assert_array_equal(plane["inlier_ransac_indices"], core_ids)

    def test_tls_rms_setting_cli_and_validation(self):
        self.assertGreaterEqual(GuidedValidatorConfig().boundary_merge_rms_error_mm, 0.)
        for mode in ("validator", "total"):
            with self.subTest(mode=mode):
                cfg, _ = configure(parse_args([
                    "--source", "realsense", "--mode", mode,
                    "--validator-boundary-merge-rms-error-mm", "1.25",
                ]))
                self.assertEqual(cfg.validator.boundary_merge_rms_error_mm, 1.25)
        for value in (-1., np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GuidedValidatorConfig(boundary_merge_rms_error_mm=value).validate()
        GuidedValidatorConfig(boundary_merge_rms_error_mm=0.).validate()

    def test_tls_merge_uses_vertex_rms_instead_of_maximum_error(self):
        patch = {"plane_id": 0, "origin_mm": np.zeros(3), "valid": True,
                 "basis_u": np.array([1., 0., 0.]), "basis_v": np.array([0., 1., 0.]),
                 "hull_uv": np.array([[-30., 0.], [0., -6.], [30., 0.], [30., 20.], [-30., 20.]])}
        query = np.zeros(3)
        cfg = GuidedValidatorConfig(boundary_merge_angle_deg=30., boundary_merge_rms_error_mm=3.5, edge_min_length_mm=5.)
        lines, _ = _boundary_segments(patch, cfg, query)
        self.assertEqual(len(lines), 4)
        line = next(line for line in lines if line["source_hull_edge_ids"] == [0, 1])
        # Unique vertices have perpendicular errors 2, 4, 2 mm after TLS.
        self.assertAlmostEqual(line["tls_rms_error_mm"], np.sqrt(8.))
        self.assertAlmostEqual(line["tls_max_error_mm"], 4.)
        self.assertGreater(line["tls_max_error_mm"], cfg.boundary_merge_rms_error_mm)
        np.testing.assert_allclose(line["start_mm"], [-30., -2., 0.], atol=1e-10)
        np.testing.assert_allclose(line["end_mm"], [30., -2., 0.], atol=1e-10)
        self.assertEqual(line["support_node_ids"], ["p0:h0", "p0:h1", "p0:h2"])
        self.assertTrue(line["start_node"]["corner_enabled"])
        self.assertTrue(line["end_node"]["corner_enabled"])

        for threshold, count in ((np.sqrt(8.), 4), (np.nextafter(np.sqrt(8.), 0.), 5)):
            with self.subTest(threshold=threshold):
                fitted, _ = _boundary_segments(patch, replace(cfg, boundary_merge_rms_error_mm=threshold), query)
                self.assertEqual(len(fitted), count)

        graph = build_validator_edge_graph(
            np.empty((0, 3)), query,
            {"plane_patches": [patch], "support_radius_mm": cfg.padded_support_radius_mm}, cfg)
        edge = next(edge for edge in graph["edges"] if edge["source_hull_edge_ids"] == [0, 1])
        self.assertAlmostEqual(edge["tls_rms_error_mm"], np.sqrt(8.))
        self.assertAlmostEqual(edge["tls_max_error_mm"], 4.)

    def test_geometry_angle_cli_settings_are_independent(self):
        for mode in ("validator", "total"):
            with self.subTest(mode=mode):
                cfg, _ = configure(parse_args([
                    "--source", "realsense", "--mode", mode,
                    "--validator-boundary-merge-angle-deg", "10",
                    "--validator-min-angle-deg", "40",
                ]))
                self.assertEqual((cfg.validator.boundary_merge_angle_deg,
                                  cfg.validator.min_angle_deg), (10., 40.))

    def test_geometry_angles_must_be_finite_and_in_positive_90_degree_range(self):
        cfg = GuidedValidatorConfig(boundary_merge_angle_deg=30., boundary_merge_rms_error_mm=3.5, edge_min_length_mm=5.)
        for name in ("boundary_merge_angle_deg", "min_angle_deg"):
            for value in (-1., 0., 91., np.nan, np.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(cfg, **{name: value}).validate()
            replace(cfg, **{name: 90.}).validate()

    def test_tls_merging_uses_only_boundary_merge_angle(self):
        patch = {"plane_id": 0, "origin_mm": np.zeros(3),
                 "basis_u": np.array([1., 0., 0.]), "basis_v": np.array([0., 1., 0.]),
                 "hull_uv": np.array([[0., 0.], [10., 0.], [20., 3.], [20., 20.], [0., 20.]])}
        query = np.zeros(3)
        cfg = GuidedValidatorConfig(boundary_merge_angle_deg=30., boundary_merge_rms_error_mm=3.5, edge_min_length_mm=5.)
        lines, _ = _boundary_segments(patch, cfg, query)
        self.assertEqual(len(lines), 4)
        strict, _ = _boundary_segments(patch, replace(cfg, boundary_merge_angle_deg=10.), query)
        self.assertEqual(len(strict), 5)
        independent, _ = _boundary_segments(
            patch, replace(cfg, min_angle_deg=10.), query)
        self.assertEqual(len(independent), len(lines))
        for actual, expected in zip(independent, lines):
            np.testing.assert_allclose(actual["start_mm"], expected["start_mm"])
            np.testing.assert_allclose(actual["end_mm"], expected["end_mm"])

    def test_thin_hull_retains_both_boundaries_with_shared_endpoint_ids(self):
        patch = {"plane_id": 0, "origin_mm": np.zeros(3), "valid": True,
                 "basis_u": np.array([1., 0., 0.]), "basis_v": np.array([0., 1., 0.]),
                 "hull_uv": np.array([[-30., 0.], [30., 0.], [0., 1.]])}
        cfg = GuidedValidatorConfig(boundary_merge_angle_deg=30., boundary_merge_rms_error_mm=3.5, edge_min_length_mm=5.)
        graph = build_validator_edge_graph(
            np.empty((0, 3)), np.zeros(3),
            {"plane_patches": [patch], "support_radius_mm": cfg.padded_support_radius_mm}, cfg)
        self.assertEqual(len(graph["edge_candidates"]), 2)
        self.assertEqual(len(graph["all_edges"]), 2)
        self.assertEqual(len(graph["edges"]), 2)
        lower, upper = graph["edges"]
        self.assertEqual([lower["source_hull_edge_ids"], upper["source_hull_edge_ids"]], [[0], [1, 2]])
        self.assertEqual({lower["start_node_id"], lower["end_node_id"]},
                         {upper["start_node_id"], upper["end_node_id"]})
        np.testing.assert_allclose(lower["start_mm"], [-30., 0., 0.])
        np.testing.assert_allclose(lower["end_mm"], [30., 0., 0.])
        np.testing.assert_allclose(upper["start_mm"], [30., 1. / 3., 0.], atol=1e-10)
        np.testing.assert_allclose(upper["end_mm"], [-30., 1. / 3., 0.], atol=1e-10)
        for index, edge in enumerate(graph["edges"]):
            self.assertEqual(edge["edge_id"], index)
            self.assertEqual(edge["source_edge_ids"], [edge["source_edge_id"]])
            self.assertEqual(edge["active_intervals_mm"], [(0., 60.)])

    def test_plane_intersection_still_masks_overlapping_boundaries(self):
        x, y = np.meshgrid(np.arange(-30., 31., 5.), np.arange(0., 21., 5.))
        horizontal = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        vertical = np.column_stack((x.ravel(), np.zeros(x.size), y.ravel()))
        points = np.vstack((horizontal, vertical))
        planes = [{"normal": normal, "offset_mm": 0.,
                   "inlier_ransac_indices": np.flatnonzero(np.abs(points @ normal) < .1)}
                  for normal in (np.array([0., 0., 1.]), np.array([0., 1., 0.]))]
        cfg = GuidedValidatorConfig(inlier_threshold_mm=.1)
        support = reconstruct_plane_support(points, np.zeros(3), planes, cfg)
        graph = build_validator_edge_graph(points, np.zeros(3), support, cfg)
        self.assertEqual(len(graph["edge_candidates"]), 9)
        self.assertEqual(len(graph["all_edges"]), 9)
        intersections = [edge for edge in graph["edges"] if edge["edge_type"] == "plane_intersection"]
        self.assertEqual(len(intersections), 1)
        np.testing.assert_allclose(intersections[0]["length_mm"], 60.)
        masked = [edge for edge in graph["all_edges"] if not edge["valid"]]
        self.assertEqual(len(masked), 2)
        self.assertEqual(len(graph["edges"]), 7)
        for edge in masked:
            self.assertEqual(edge["edge_type"], "boundary")
            self.assertEqual(edge["failure_reason"], "duplicate_of_intersection")
            self.assertEqual(edge["active_intervals_mm"], [])
            self.assertEqual(edge["masked_by_intersection_ids"], [intersections[0]["edge_id"]])

    def test_supported_patch_returns_all_four_unverified_endpoint_corners(self):
        x, y = np.meshgrid(np.arange(0., 41., 5.), np.arange(0., 41., 5.))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        cfg = GuidedValidatorConfig(boundary_merge_angle_deg=30., boundary_merge_rms_error_mm=3.5, edge_min_length_mm=5.)
        core_ids = np.flatnonzero(np.linalg.norm(points, axis=1) <= cfg.ransac_radius_mm)
        plane = {"normal": np.array([0., 0., 1.]), "offset_mm": 0.,
                 "inlier_ransac_indices": core_ids}
        original = plane["normal"].copy()
        support = reconstruct_plane_support(points, np.zeros(3), [plane], cfg)
        graph = build_validator_edge_graph(points, np.zeros(3), support, cfg)
        result = reconstruct_validator_corners(graph, cfg)
        self.assertEqual(result["validation_status"], "CANDIDATES")
        self.assertFalse(result["verified"])
        self.assertEqual(len(result["corners"]), 4)
        np.testing.assert_allclose(np.unique([corner["position"] for corner in result["corners"]], axis=0),
                                   [[0., 0., 0.], [0., 40., 0.], [40., 0., 0.], [40., 40., 0.]], atol=1e-8)
        for corner in result["corners"]:
            self.assertEqual(len(corner["edge_ids"]), 2)
            self.assertEqual(set(corner["edge_ids"]), set(corner["endpoint_ids"]))
            self.assertLessEqual(corner["endpoint_gap"], 1e-8)
        np.testing.assert_array_equal(plane["normal"], original)


if __name__ == "__main__":
    unittest.main()
