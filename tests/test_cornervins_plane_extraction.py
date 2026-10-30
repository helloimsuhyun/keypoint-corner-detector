import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from corner_detector.cornervins_plane_extraction import (
    CornerVINSPlaneConfig,
    PlaneExtractionResult,
    PlanePatch,
    as_plane_cluster_analysis,
    extract_planes,
)
from corner_detector.settings import configure, parse_args
from corner_detector.validator_pipeline import _cornervins_clustering


class CornerVINSPlaneExtractionContractTest(unittest.TestCase):
    def test_cornervins_is_the_automatic_plane_extractor(self):
        args = parse_args([
            "--source", "realsense",
            "--no-3d-view",
            "--cornervins-region-mode", "paper",
        ])
        config, _noise = configure(args)

        self.assertEqual(config.cornervins_region_mode, "paper")
        self.assertEqual(config.cornervins_min_region_points_paper, 20_000)
        self.assertEqual(config.cornervins_ahc_block_size, 3)
        self.assertEqual(config.cornervins_ransac_threshold_mm, 15.0)
        self.assertEqual(config.projection_halo_mm, 50.0)
        self.assertEqual(config.support_radius_mm, 150.0)
        self.assertFalse(hasattr(config, "plane_extraction_mode"))
        self.assertFalse(hasattr(config, "plane_cluster_enabled"))

    def test_unorganized_input_has_no_knn_fallback(self):
        with self.assertRaisesRegex(ValueError, "unorganized/KNN fallback"):
            extract_planes(np.zeros((3, 3), dtype=np.float64))

    def test_validator_runs_ahc_on_query_core_pixels_and_caches_per_query(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        organized = points.reshape(2, 2, 3)
        context = {
            "organized_points_mm": organized,
            "valid_mask": np.ones((2, 2), dtype=bool),
            "organized_to_scene_index": np.arange(4).reshape(2, 2),
        }
        cfg = SimpleNamespace(
            query_radius_mm=1.1,
            projection_halo_mm=0.9,
            plane_edge_plane_inlier_mm=1.0,
            cornervins_ahc_block_size=2,
            cornervins_ahc_mse_threshold_mm2=None,
            cornervins_ahc_refinement=True,
            cornervins_region_mode="roi_test",
            cornervins_min_region_points_paper=20_000,
            cornervins_min_region_points_roi=100,
            cornervins_downsample_leaf_mm=50.0,
            cornervins_ransac_threshold_mm=1.0,
            cornervins_ransac_min_inlier_ratio=0.9,
            cornervins_ransac_max_trials=32,
            cornervins_ransac_seed=0,
            cornervins_merge_angle_deg=5.0,
            cornervins_merge_distance_mm=20.0,
        )
        empty = PlaneExtractionResult(
            patches=(),
            ahc_regions=(),
            ahc_membership_image=np.full((2, 2), -1, dtype=np.int32),
            final_membership_image=np.full((2, 2), -1, dtype=np.int32),
            diagnostics={
                "valid_organized_point_count": 2,
                "plane_count": 0,
                "timings_seconds": {"total": 0.0},
            },
        )

        with patch(
            "corner_detector.validator_pipeline.extract_planes",
            return_value=empty,
        ) as mocked:
            first = _cornervins_clustering(points, 0, cfg, context)
            second = _cornervins_clustering(points, 0, cfg, context)

        self.assertIs(first["cornervins_result"], second["cornervins_result"])
        self.assertEqual(mocked.call_count, 1)
        np.testing.assert_array_equal(
            mocked.call_args.kwargs["valid_mask"],
            np.asarray([[True, True], [False, False]]),
        )
        self.assertEqual(first["organized_extraction_scope"], "query_core")
        self.assertEqual(first["organized_extraction_pixel_count"], 2)
        self.assertEqual(first["projection_halo_mm"], 0.9)

    def test_adapter_preserves_scene_indices_and_separates_core_from_halo(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.5, 0.0, 0.2],
            [2.1, 0.0, 0.0],
        ])
        patch = PlanePatch(
            plane_id=0,
            normal=np.asarray([0.0, 0.0, 1.0]),
            offset=0.0,
            center=np.asarray([0.75, 0.25, 0.0]),
            point_indices=np.arange(4, dtype=np.int64),
            inlier_indices=np.arange(4, dtype=np.int64),
            hull_vertices_3d=points[:4].copy(),
            residual_rmse=0.0,
            inlier_ratio=1.0,
        )
        result = PlaneExtractionResult(
            patches=(patch,),
            ahc_regions=(),
            ahc_membership_image=np.empty((0, 0), dtype=np.int32),
            final_membership_image=np.empty((0, 0), dtype=np.int32),
            diagnostics={
                "timings_seconds": {"total": 0.0},
                "plane_count": 1,
            },
        )

        analysis = as_plane_cluster_analysis(
            result,
            points,
            query_index=0,
            query_radius_mm=1.1,
            roi_indices=np.asarray([0, 1, 2]),
            projection_halo_mm=0.9,
            projection_inlier_mm=0.1,
        )

        plane = analysis["global_planes"][0]
        np.testing.assert_array_equal(plane["support_indices"], [0, 1, 2])
        np.testing.assert_array_equal(plane["projection_indices"], [0, 1, 2, 3])
        np.testing.assert_array_equal(
            analysis["projection_support_indices"], [0, 1, 2, 3]
        )

    def test_projection_halo_cli_is_absolute_mm_and_rejects_negative(self):
        args = parse_args([
            "--source", "realsense", "--no-3d-view",
            "--projection-halo-mm", "25",
        ])
        config, _noise = configure(args)
        self.assertEqual(config.query_radius_mm, 100.0)
        self.assertEqual(config.projection_halo_mm, 25.0)
        self.assertEqual(config.support_radius_mm, 125.0)

        invalid = parse_args([
            "--source", "realsense", "--no-3d-view",
            "--projection-halo-mm", "-1",
        ])
        with self.assertRaisesRegex(ValueError, "projection-halo-mm"):
            configure(invalid)


if __name__ == "__main__":
    unittest.main()
