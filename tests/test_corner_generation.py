"""Numerical contracts for the independent endpoint Corner generator."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from corner_detector.settings import GuidedValidatorConfig, ValidatorDetectorConfig, configure, parse_args
from corner_detector.validator_corner_graph import generate_corners, reconstruct_validator_corners
from corner_detector.validator_pipeline import (
    _final_geometry, prepare_validator_voxel_context, run_validator_candidates, run_validator_pipeline,
)
from corner_detector.visualization import view3d


def edge(identifier, start, end, tolerance=0.):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = float(np.linalg.norm(end - start))
    return {"edge_id": identifier, "start_mm": start, "end_mm": end,
            "direction": (end - start) / length, "length_mm": length,
            "fit_tolerance_mm": tolerance, "valid": True,
            "edge_type": "boundary", "plane_ids": [0],
            "active_intervals_mm": [(0., length)]}


class CornerGenerationTest(unittest.TestCase):
    def test_angle_uses_unoriented_directions_and_includes_threshold(self):
        horizontal = edge(0, [0, 0, 0], [10, 0, 0])
        for angle, expected in ((0., 0), (10., 0), (20., 1), (90., 1), (160., 1), (170., 0), (180., 0)):
            with self.subTest(angle=angle):
                direction = 10. * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0.])
                self.assertEqual(len(generate_corners([horizontal, edge(1, [0, 0, 0], direction)])), expected)

    def test_endpoint_radius_excludes_interior_crossings_and_includes_threshold(self):
        interior = [edge(0, [-10, 0, 0], [10, 0, 0]), edge(1, [0, -10, 0], [0, 10, 0])]
        self.assertEqual(generate_corners(interior), [])
        endpoints = [edge(0, [-15, 0, 0], [-5, 0, 0]), edge(1, [0, 0, 0], [0, 10, 0])]
        corner, = generate_corners(endpoints)
        np.testing.assert_allclose(corner["position"], [0, 0, 0], atol=1e-12)
        self.assertEqual(corner["endpoint_ids"], {0: 1, 1: 0})
        self.assertEqual(corner["endpoint_gap"], 5.)
        self.assertEqual(generate_corners(endpoints, corner_radius_mm=np.nextafter(5., 0.)), [])

    def test_skew_intersection_uses_existing_fit_tolerance_and_midpoint(self):
        edges = [edge(0, [0, 0, 0], [10, 0, 0], 1.), edge(1, [0, 0, 1], [0, 10, 1])]
        corner, = generate_corners(edges)
        np.testing.assert_allclose(corner["position"], [0, 0, .5])
        self.assertEqual(corner["intersection_residual"], 1.)
        self.assertEqual(corner["endpoint_gap"], .5)
        self.assertEqual(generate_corners([{**edges[0], "fit_tolerance_mm": .9}, edges[1]]), [])
        self.assertEqual(generate_corners([{**item, "fit_tolerance_mm": 0.} for item in edges]), [])

    def test_merging_keeps_smallest_gap_candidate_without_averaging(self):
        edges = [edge(0, [0, 0, 0], [20, 0, 0]),
                 edge(1, [-1, 0, 0], [-1, 10, 0]), edge(2, [2, 0, 0], [2, 10, 0])]
        original = deepcopy(edges)
        corner, = generate_corners(edges)
        np.testing.assert_allclose(corner["position"], [-1, 0, 0])
        self.assertEqual(corner["edge_ids"], [0, 1, 2])
        self.assertEqual(corner["endpoint_ids"], {0: 0, 1: 0, 2: 0})
        self.assertEqual(corner["endpoint_gap"], 1.)
        for actual, expected in zip(edges, original):
            np.testing.assert_array_equal(actual["start_mm"], expected["start_mm"])
            np.testing.assert_array_equal(actual["end_mm"], expected["end_mm"])
        reversed_corner, = generate_corners(edges[::-1])
        np.testing.assert_array_equal(reversed_corner["position"], corner["position"])
        self.assertEqual(reversed_corner["endpoint_ids"], corner["endpoint_ids"])

    def test_shared_edge_opposite_endpoints_do_not_merge(self):
        edges = [edge(0, [0, 0, 0], [4, 0, 0]),
                 edge(1, [0, 0, 0], [0, 10, 0]), edge(2, [4, 0, 0], [4, 10, 0])]
        corners = generate_corners(edges)
        self.assertEqual(len(corners), 2)
        self.assertEqual([item["endpoint_ids"][0] for item in corners], [0, 1])

    def test_nearby_candidates_without_shared_edge_remain_separate(self):
        edges = [edge(0, [0, 0, 0], [10, 0, 0]), edge(1, [0, 0, 0], [0, 10, 0]),
                 edge(2, [0, 0, 1], [10, 0, 1]), edge(3, [0, 0, 1], [0, 10, 1])]
        corners = generate_corners(edges)
        self.assertEqual([item["edge_ids"] for item in corners], [[0, 1], [2, 3]])

    def test_complete_group_diameter_prevents_chained_merge(self):
        edges = [edge(0, [0, 0, 0], [100, 0, 0])]
        edges += [edge(i, [x, 0, 0], [x, 10, 0]) for i, x in enumerate((-4, 0, 4), 1)]
        corners = generate_corners(edges)
        self.assertEqual([item["edge_ids"] for item in corners], [[0, 1, 2], [0, 3]])
        np.testing.assert_allclose([item["position"] for item in corners], [[0, 0, 0], [4, 0, 0]])

    def test_all_corners_are_unverified_and_batch_output_preserves_every_corner(self):
        edges = [edge(0, [100, 0, 0], [120, 0, 0]), edge(1, [100, 0, 0], [100, 10, 0]),
                 edge(2, [120, 0, 0], [120, 10, 0])]
        cfg = GuidedValidatorConfig()
        generated = reconstruct_validator_corners({"edges": edges}, cfg)
        self.assertFalse(generated["verified"])
        self.assertEqual(generated["validation_status"], "CANDIDATES")
        self.assertEqual(len(generated["corners"]), 2)
        final_edges, corners = _final_geometry({**generated, "query_index": 7, "edges": edges})
        self.assertEqual(len(corners), 2)
        self.assertEqual(len(final_edges), 4)
        self.assertEqual([item["corner_id"] for item in corners], [0, 1])
        np.testing.assert_allclose([item["position_mm"] for item in corners], [[100, 0, 0], [120, 0, 0]])
        self.assertTrue(all(item["query_index"] == 7 for item in final_edges + corners))
        self.assertEqual(reconstruct_validator_corners({"edges": []}, cfg)["validation_status"], "NO_CANDIDATES")

    def test_corner_settings_propagate_and_old_fields_are_removed(self):
        cfg = GuidedValidatorConfig(min_angle_deg=20., corner_radius_mm=5.)
        self.assertEqual((cfg.min_angle_deg, cfg.corner_radius_mm), (20., 5.))
        removed = ("corner_min_edge_angle_deg", "vertex_line_distance_mm", "vertex_endpoint_tolerance_mm",
                   "vertex_group_span_mm", "corner_gate_radius_mm")
        self.assertFalse(set(removed) & vars(cfg).keys())
        for mode in ("validator", "total"):
            configured, _ = configure(parse_args(["--source", "realsense", "--mode", mode,
                                                  "--validator-min-angle-deg", "25",
                                                  "--validator-corner-radius-mm", "3"]))
            self.assertEqual((configured.validator.min_angle_deg, configured.validator.corner_radius_mm), (25., 3.))
        for changes in ({"min_angle_deg": 0.}, {"min_angle_deg": 91.}, {"min_angle_deg": np.nan},
                        {"corner_radius_mm": 0.}, {"corner_radius_mm": np.inf}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes).validate()

    def test_query_and_compact_batch_return_all_corners_without_selection(self):
        x, y = np.meshgrid(np.arange(0., 41., 5.), np.arange(0., 41., 5.))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        cfg = ValidatorDetectorConfig(validator=GuidedValidatorConfig(
            max_planes=1, min_inliers=3, ransac_iterations=16, min_angle_deg=20., corner_radius_mm=5.,
            boundary_merge_angle_deg=30., boundary_merge_rms_error_mm=3.5, edge_min_length_mm=5.))
        context = prepare_validator_voxel_context(points, points, cfg)
        result = run_validator_pipeline(points, 0, cfg, voxel_context=context)
        self.assertEqual(result["validation_status"], "CANDIDATES")
        self.assertFalse(result["verified"])
        self.assertEqual(len(result["corners"]), 4)
        self.assertEqual(len(result["roi_debug_before"]["corners"]), 4)
        self.assertNotIn("selected_vertex_id", result)
        self.assertNotIn("corner_position_mm", result)
        np.testing.assert_allclose(np.unique([item["position"] for item in result["corners"]], axis=0),
                                   [[0., 0., 0.], [0., 40., 0.], [40., 0., 0.], [40., 40., 0.]], atol=1e-8)
        batch = run_validator_candidates(points, [0], cfg, voxel_context=context, include_debug=False)
        self.assertEqual(len(batch["final_corners"]), 4)
        self.assertEqual(len(batch["final_edges"]), 8)
        self.assertEqual(len(batch["records"][0]["corners"]), 4)
        self.assertNotIn("ransac_points_mm", batch["records"][0])

    def test_invalid_inputs_are_rejected(self):
        valid = edge(0, [0, 0, 0], [10, 0, 0])
        for edges in ([valid, valid], [{**valid, "edge_id": .5}],
                      [{**valid, "end_mm": np.zeros(3)}], [{**valid, "end_mm": [np.nan, 0, 0]}],
                      [{**valid, "fit_tolerance_mm": -1.}]):
            with self.subTest(edges=edges), self.assertRaises(ValueError):
                generate_corners(edges)
        for angle, radius in ((0, 5), (91, 5), (20, 0), (np.nan, 5), (20, np.inf)):
            with self.subTest(angle=angle, radius=radius), self.assertRaises(ValueError):
                generate_corners([], angle, radius)

    def test_final_preview_draws_all_corners_and_endpoint_links(self):
        edges = [edge(0, [0, 0, 0], [20, 0, 0]), edge(1, [0, 0, 0], [0, 10, 0]),
                 edge(2, [20, 0, 0], [20, 10, 0])]
        result = {"edges": edges, "corners": generate_corners(edges)}
        original = deepcopy(result["corners"])
        fake, scene = Mock(), Mock()
        with patch.object(view3d, "require_open3d", return_value=fake):
            names = view3d._add_validator_graph_step(scene, result, 4)
        self.assertIn("corner_candidate_0", names)
        self.assertIn("corner_candidate_1", names)
        self.assertIn("endpoint_gap_c0_e0", names)
        self.assertIn("endpoint_gap_c1_e0", names)
        for actual, expected in zip(result["corners"], original):
            np.testing.assert_array_equal(actual["position"], expected["position"])
            self.assertEqual(actual["endpoint_ids"], expected["endpoint_ids"])


if __name__ == "__main__":
    unittest.main()
