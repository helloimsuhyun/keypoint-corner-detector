"""Numerical regression fixture; not executed during the requested implementation."""

import unittest

import numpy as np

from corner_detector.settings import GuidedValidatorConfig
from corner_detector.validator_plane_support import reconstruct_plane_support
from corner_detector.validator_edge_graph import build_validator_edge_graph
from corner_detector.validator_corner_graph import reconstruct_validator_corners


class ValidatorCornerGeometryTest(unittest.TestCase):
    def test_supported_boundary_junction_returns_corner_and_incident_edges(self):
        x, y = np.meshgrid(np.arange(0., 41., 5.), np.arange(0., 41., 5.))
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        cfg = GuidedValidatorConfig()
        core_ids = np.flatnonzero(np.linalg.norm(points, axis=1) <= cfg.ransac_radius_mm)
        plane = {"normal": np.array([0., 0., 1.]), "offset_mm": 0.,
                 "inlier_ransac_indices": core_ids}
        original = plane["normal"].copy()
        support = reconstruct_plane_support(points, np.zeros(3), [plane], cfg)
        graph = build_validator_edge_graph(points, np.zeros(3), support, cfg)
        result = reconstruct_validator_corners(points, np.zeros(3), support, graph, cfg)
        self.assertEqual(result["validation_status"], "ACCEPT")
        self.assertTrue(result["verified"])
        np.testing.assert_allclose(result["corner_position_mm"], [0., 0., 0.], atol=1e-8)
        self.assertGreaterEqual(len(result["corner_edges"]), 2)
        self.assertEqual(result["corner_edge_ids"], [edge["edge_id"] for edge in result["corner_edges"]])
        for edge in result["corner_edges"]:
            self.assertTrue(edge["valid"])
            self.assertEqual(edge["edge_type"], "boundary")
            self.assertTrue(np.all(edge["support_ransac_indices"] < len(points)))
        np.testing.assert_array_equal(plane["normal"], original)


if __name__ == "__main__":
    unittest.main()
