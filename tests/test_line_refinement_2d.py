import unittest

import numpy as np

from corner_detector.line_refinement_2d import (
    LineRefinement2DConfig,
    _duplicates,
    refine_hough_lines_2d,
)


def _seed(rho, theta_deg, support):
    theta = float(np.deg2rad(theta_deg) % np.pi)
    return {
        "rho": float(rho),
        "rho_mm": float(rho),
        "theta_rad": theta,
        "normal": np.array([np.cos(theta), np.sin(theta)]),
        "vote_count": int(len(support)),
        "weighted_vote": float(len(support)),
        "indices": np.asarray(support, dtype=np.int64),
    }


def _run(uv, weights, mask, seeds, *, origin=(0.0, 0.0), min_length=10.0):
    return refine_hough_lines_2d(
        uv,
        weights,
        mask,
        seeds,
        roi_center_uv=np.asarray(origin, dtype=float),
        core_mask=np.ones(len(uv), dtype=bool),
        rho_bin_mm=1.0,
        hough_nms_rho_mm=6.0,
        hough_nms_angle_deg=10.0,
        hough_theta_step_deg=0.5,
        min_points=5,
        min_length_mm=min_length,
        config=LineRefinement2DConfig(max_iterations=60),
    )


class ContinuousLineRefinementTests(unittest.TestCase):
    def test_multiple_seeds_converge_and_final_nms_keeps_one_edge(self):
        uv = np.array([
            (x, y) for x in np.arange(0.0, 16.0) for y in np.arange(-20.0, 21.0)
        ])
        weights = np.exp(-0.5 * (uv[:, 0] / 1.2) ** 2)
        voters = uv[:, 0] <= 2.0
        ids = np.flatnonzero(voters)
        result = _run(
            uv, weights, voters,
            [_seed(2.0, 2.0, ids), _seed(-1.0, 178.0, ids), _seed(4.0, 177.0, ids)],
            min_length=15.0,
        )
        candidates = result["refined_segments_before_nms"]
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            theta = candidate["final_theta_rad"]
            intercept_x = candidate["final_rho_mm"] / np.cos(theta)
            self.assertLess(abs(intercept_x), 1.0)
            self.assertLess(candidate["final_loss"], candidate["initial_loss"])
            self.assertEqual(candidate["optimization_path"].shape[1], 2)
            self.assertGreater(candidate["length_mm"], 35.0)
        self.assertEqual(len(result["refined_segments_after_nms"]), 1)
        self.assertEqual(result["final_nms_suppressed"], 2)
        self.assertIsNone(result["score_filter_threshold"])

    def test_corner_keeps_two_perpendicular_edges(self):
        uv = np.array([
            (x, y) for x in np.arange(0.0, 21.0) for y in np.arange(0.0, 21.0)
        ])
        weights = np.maximum(
            np.exp(-0.5 * (uv[:, 0] / 1.2) ** 2),
            np.exp(-0.5 * (uv[:, 1] / 1.2) ** 2),
        )
        voters = (uv[:, 0] <= 2.0) | (uv[:, 1] <= 2.0)
        ids = np.flatnonzero(voters)
        result = _run(
            uv, weights, voters,
            [_seed(2.0, 3.0, ids), _seed(2.0, 87.0, ids)],
            origin=(10.0, 10.0),
        )
        final = result["refined_segments_after_nms"]
        self.assertEqual(len(final), 2)
        directions = [item["direction"] for item in final]
        self.assertLess(abs(float(directions[0] @ directions[1])), 0.1)

    def test_noise_missing_points_and_interior_seed_rejection(self):
        rng = np.random.default_rng(4)
        uv = np.array([
            (x, y)
            for x in np.arange(0.0, 16.0)
            for y in np.arange(-20.0, 21.0)
            if not (-3.0 < y < 3.0)
        ])
        uv = uv + rng.normal(0.0, 0.08, uv.shape)
        weights = np.exp(-0.5 * (uv[:, 0] / 1.2) ** 2)
        voters = uv[:, 0] <= 2.0
        ids = np.flatnonzero(voters)
        result = _run(
            uv, weights, voters,
            [_seed(1.0, 2.0, ids), _seed(30.0, 0.0, ids)],
        )
        candidates = result["refined_segments_before_nms"]
        self.assertTrue(candidates[0]["score_pass"])
        self.assertFalse(candidates[1]["score_pass"])
        self.assertIn(candidates[1]["rejection_reason"], {
            "insufficient_boundary_support", "segment_too_short",
        })
        self.assertEqual(
            [item["seed_index"] for item in result["refined_segments_after_nms"]],
            [0],
        )

    def test_final_nms_does_not_remove_disjoint_collinear_segments(self):
        base = {
            "direction": np.array([0.0, 1.0]),
            "normal": np.array([1.0, 0.0]),
            "point_uv": np.array([0.0, 0.0]),
            "start_uv": np.array([0.0, -20.0]),
            "end_uv": np.array([0.0, -10.0]),
        }
        separated = {
            **base,
            "start_uv": np.array([0.2, 10.0]),
            "end_uv": np.array([0.2, 20.0]),
            "point_uv": np.array([0.2, 0.0]),
        }
        self.assertFalse(_duplicates(base, separated, 5.0, 2.0, 0.5))


if __name__ == "__main__":
    unittest.main()
