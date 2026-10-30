"""All input sources must feed the same formal whole-cloud candidate mode."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from corner_detector import runner
from corner_detector.settings import parse_args


class HeatmapSourceCompatibilityTest(unittest.TestCase):
    def test_fine_is_default_and_uses_the_clicked_query_path(self):
        args = parse_args(["--source", "realsense"])
        self.assertEqual(args.mode, "fine")
        self.assertTrue(args.plane_edge_graph_enabled)
        self.assertTrue(args.plane_edge_debug_2d)
        frame = SimpleNamespace(source="realsense", points_mm=np.zeros((1, 3)))
        frozen = {}
        with patch.object(runner, "require_open3d"), \
                patch.object(runner, "tk", object()), \
                patch.object(runner, "capture_realsense_point_cloud", return_value=frame), \
                patch.object(runner, "apply_noise_to_frame", return_value=frame), \
                patch.object(runner, "prepare_frozen_from_frame", return_value=(frozen, frame.points_mm)), \
                patch.object(runner, "run_validator_mode", return_value="quit") as run_query, \
                patch.object(runner, "close_query_views"), redirect_stdout(StringIO()):
            runner.main(["--source", "realsense", "--mode", "fine"])
        self.assertIs(run_query.call_args.args[0], frame)
        self.assertIs(run_query.call_args.args[1], frozen)

    def test_missing_kinect_input_exits_cleanly_and_hides_opt_in_logs(self):
        missing = Path("/definitely/missing/7scenes/seq-03.zip")
        output = StringIO()
        with patch.object(runner, "require_open3d"), \
                patch.object(runner, "tk", object()), \
                patch.object(
                    runner, "load_kinect_point_cloud",
                    side_effect=FileNotFoundError(missing),
                ), redirect_stdout(output):
            with self.assertRaises(SystemExit) as caught:
                runner.main([
                    "--source", "kinect", "--mode", "fine",
                    "--input", str(missing),
                ])
        message = str(caught.exception)
        self.assertIn("[INPUT ERROR]", message)
        self.assertIn(str(missing), message)
        self.assertIn("--input", message)
        self.assertNotIn("[multiscale]", output.getvalue())
        self.assertNotIn("[CED-3D]", output.getvalue())

    def test_stl_kinect_and_realsense_share_the_heatmap_candidate_path(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        cfg = SimpleNamespace(
            voxel_3d_mm=2.0,
            centroid_radius_mm=20.0,
            pca_radius_mm=36.0,
            pca_radius_factor=1.8,
            bilateral_radius_mm=20.0,
            bilateral_sigma_spatial_mm=10.0,
            bilateral_sigma_ced=0.05,
            pca_ratio_saturation_tau=None,
            fusion_score_threshold=0.0,
            nms_radius_mm=10.0,
            min_neighbors=3,
        )
        noise = SimpleNamespace()

        for source in ("stl", "kinect", "realsense"):
            with self.subTest(source=source):
                args = SimpleNamespace(
                    source=source,
                    mode="detector",
                    input="dummy-input",
                    stl_unit_mm=1.0,
                    width=640,
                    height=480,
                    fov_deg=60.0,
                    min_depth_m=0.01,
                    max_depth_m=5.0,
                    frame_index=0,
                    fps=30,
                )
                frame = SimpleNamespace(source=source, points_mm=points)
                with patch.object(runner, "parse_args", return_value=args), \
                        patch.object(runner, "configure", return_value=(cfg, noise)), \
                        patch.object(runner, "require_open3d"), \
                        patch.object(runner, "capture_stl_point_cloud", return_value=frame) as stl, \
                        patch.object(runner, "load_kinect_point_cloud", return_value=frame) as kinect, \
                        patch.object(runner, "capture_realsense_point_cloud", return_value=frame) as realsense, \
                        patch.object(runner, "apply_noise_to_frame", return_value=frame), \
                        patch.object(
                            runner, "prepare_frozen_from_frame",
                            return_value=({}, points),
                        ), \
                        patch.object(runner, "run_dense_heatmap_mode") as heatmap:
                    runner.main([])

                heatmap.assert_called_once_with(points, cfg)
                calls = {
                    "stl": stl.call_count,
                    "kinect": kinect.call_count,
                    "realsense": realsense.call_count,
                }
                self.assertEqual(calls[source], 1)
                self.assertEqual(sum(calls.values()), 1)

    def test_realsense_heatmap_forwards_registered_rgb_for_candidate_view(self):
        points = np.array(
            [[0.0, 0.0, 100.0], [1.0, 0.0, 100.0], [0.0, 1.0, 100.0]],
        )
        colors = np.array(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8,
        )
        frame = SimpleNamespace(
            source="realsense-live",
            points_mm=points,
            colors_rgb=colors,
        )
        args = SimpleNamespace(
            source="realsense",
            mode="detector",
            input=None,
            stl_unit_mm=1.0,
            width=640,
            height=480,
            fov_deg=60.0,
            min_depth_m=0.01,
            max_depth_m=5.0,
            frame_index=0,
            fps=30,
        )
        cfg = SimpleNamespace(
            voxel_3d_mm=2.0,
            centroid_radius_mm=20.0,
            pca_radius_mm=36.0,
            pca_radius_factor=1.8,
            bilateral_radius_mm=20.0,
            bilateral_sigma_spatial_mm=10.0,
            bilateral_sigma_ced=0.05,
            pca_ratio_saturation_tau=None,
            fusion_score_threshold=0.0,
            nms_radius_mm=10.0,
            min_neighbors=3,
        )
        with patch.object(runner, "parse_args", return_value=args), \
                patch.object(runner, "configure", return_value=(cfg, SimpleNamespace())), \
                patch.object(runner, "require_open3d"), \
                patch.object(runner, "capture_realsense_point_cloud", return_value=frame), \
                patch.object(runner, "apply_noise_to_frame", return_value=frame), \
                patch.object(
                    runner, "prepare_frozen_from_frame", return_value=({}, points),
                ), \
                patch.object(runner, "run_dense_heatmap_mode") as heatmap:
            runner.main([])

        heatmap.assert_called_once_with(
            points,
            cfg,
            rgb_points_mm=points,
            rgb_colors_rgb=colors,
        )


if __name__ == "__main__":
    unittest.main()
