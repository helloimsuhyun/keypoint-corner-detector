"""Headless contracts for active visualization management."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from corner_detector.visualization import query_ui
from corner_detector.visualization import view3d
from corner_detector.visualization import manager
from corner_detector.extremity import run_extremity_refinement
from corner_detector.visualization.view3d import (
    centroid_saliency_colors,
    cornervins_ahc_primitives,
    cornervins_scene_roi_primitives,
)


class ManagerTest(unittest.TestCase):
    def ball_result(self):
        points = np.column_stack((np.arange(0, 10, 3), np.zeros((4, 2))))
        return points, run_extremity_refinement(points, 1)

    def test_ball_raw_heatmap_uses_all_raw_votes_without_score_or_nms_filtering(self):
        points, result = self.ball_result()
        result["raw_votes"] = np.array([0, 1, 4, 100])
        result["normalized_score"] = np.array([1.0, 0.9, 0.8, 0.7])
        before = result["raw_votes"].copy()
        data = view3d.extremity_primitives(points, result, color_mode="raw_heatmap")
        expected = np.vstack(([0.28, 0.28, 0.30], view3d.centroid_saliency_colors(
            [0.01, 0.04, 1])))
        np.testing.assert_array_equal(data["point_colors"], expected)
        self.assertFalse(np.allclose(data["point_colors"][0], data["point_colors"][1]))
        np.testing.assert_array_equal(result["raw_votes"], before)

    def test_ball_stage_views_preserve_scene_and_only_nms_adds_peak_markers(self):
        points, result = self.ball_result()
        original = {k: v.copy() for k, v in result.items() if isinstance(v, np.ndarray)}
        fake = Mock()
        views = [Mock(), Mock(), Mock()]
        for view in views:
            view.create_window.return_value = True
        fake.visualization.Visualizer.side_effect = views
        fake.geometry.PointCloud.side_effect = Mock
        fake.geometry.LineSet.side_effect = Mock
        fake.geometry.TriangleMesh.create_sphere.side_effect = Mock
        fake.utility.Vector3dVector.side_effect = lambda value: np.array(value, copy=True)
        fake.utility.Vector2iVector.side_effect = lambda value: np.array(value, copy=True)
        with patch.object(view3d, "require_open3d", return_value=fake):
            opened = view3d.open_extremity_views(points, result)
        self.assertEqual(opened, tuple(views))
        for index, view in enumerate(views):
            self.assertIn(f"BALL {index + 1}/3", view.create_window.call_args.kwargs["window_name"])
            self.assertEqual(view.create_window.call_args.kwargs["left"], 10 + index * 510)
            cloud = view.add_geometry.call_args_list[0].args[0]
            np.testing.assert_array_equal(cloud.points, points)
        self.assertIn("RAW votes LINEAR [0,", views[1].create_window.call_args.kwargs["window_name"])
        self.assertEqual(views[0].add_geometry.call_count, 4)
        self.assertEqual(views[1].add_geometry.call_count, 4)
        self.assertEqual(views[2].add_geometry.call_count, 4 + result["candidate_count"])
        np.testing.assert_array_equal(views[2].add_geometry.call_args_list[0].args[0].colors,
                                      np.tile([0.42, 0.43, 0.46], (len(points), 1)))
        for key, value in original.items():
            np.testing.assert_array_equal(result[key], value)

    def test_ball_open_failure_cleans_up_previous_and_partial_windows(self):
        points, result = self.ball_result()
        previous = Mock()
        first, failed = Mock(), Mock()
        first.create_window.return_value = True
        failed.create_window.return_value = False
        fake = Mock()
        fake.visualization.Visualizer.side_effect = [first, failed]
        with patch.object(view3d, "require_open3d", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "Failed to create ball"):
                view3d.open_extremity_views(points, result, (previous, None))
        for view in (previous, first, failed):
            view.destroy_window.assert_called_once()

    def test_ball_manager_opens_three_views_and_respects_no_3d(self):
        points, result = self.ball_result()
        old = (Mock(), None, Mock())
        new = (Mock(), Mock(), Mock())
        frozen = {"points_raw_mm": points, "extremity_views": old}
        with patch.object(manager, "open_extremity_views", return_value=new) as open_views:
            manager.prepare_extremity_view(frozen, result)
            open_views.assert_called_once_with(points, result, old)
            self.assertIs(frozen["extremity_result"], result)
            self.assertIs(frozen["extremity_views"], new)
        with patch.object(manager, "open_extremity_views") as open_views:
            frozen = {"points_raw_mm": points}
            manager.prepare_extremity_view(frozen, result, show_3d=False)
            open_views.assert_not_called()
            self.assertNotIn("extremity_views", frozen)

    def test_ball_manager_polls_and_closes_all_three_views_without_discarding_batch(self):
        first, closed, last = Mock(), Mock(), Mock()
        first.poll_events.return_value = last.poll_events.return_value = True
        closed.poll_events.return_value = False
        batch = object()
        frozen = {"extremity_views": (first, closed, last), "extremity_batch": batch}
        manager.poll_view_events(frozen)
        self.assertEqual(frozen["extremity_views"], (first, None, last))
        first.update_renderer.assert_called_once()
        last.update_renderer.assert_called_once()
        closed.update_renderer.assert_not_called()
        manager.close_query_views(frozen)
        for view in (first, closed, last):
            view.destroy_window.assert_called_once()
        self.assertNotIn("extremity_views", frozen)
        self.assertIs(frozen["extremity_batch"], batch)

    def test_ball_overview_uses_original_nms_heatmap_and_both_index_spaces(self):
        raw = np.array([[100, 0, 0], [200, 0, 0], [201, 0, 0], [400, 0, 0]])
        scene = np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]])
        values = np.array([0.1, 0.4, 0.2, 10.0])
        valid = np.array([True, True, True, False])
        scores = dict(ced_score=values.copy(), ced_valid_mask=valid.copy(),
                      pca_ratio_saturated=values.copy(), pca_valid_mask=valid.copy(),
                      fusion_score=values.copy(), fusion_valid_mask=valid.copy())
        records = [{"candidate_raw_indices": np.array([1, 2])},
                   {"candidate_raw_indices": np.array([2, 3])}]
        batch = {"query_indices": np.array([1, 0]), "records": records, "ced_score_result": scores}
        before = [r["candidate_raw_indices"].copy() for r in records]
        fake = MagicMock()
        fake.visualization.Visualizer.return_value.create_window.return_value = True
        fake.utility.Vector3dVector.side_effect = lambda value: np.array(value, copy=True)
        spheres = [Mock() for _ in range(5)]
        fake.geometry.TriangleMesh.create_sphere.side_effect = spheres
        with patch.object(view3d, "require_open3d", return_value=fake):
            opened = view3d.open_extremity_overview(raw, scene, batch)
        cloud = opened.add_geometry.call_args_list[0].args[0]
        np.testing.assert_array_equal(cloud.points, scene)
        np.testing.assert_array_equal(cloud.colors, view3d.dense_heatmap_colors(
            values, valid, vmax=np.quantile(values[valid], 0.99)))
        np.testing.assert_array_equal(scores["fusion_score"], values)
        np.testing.assert_array_equal(scores["fusion_valid_mask"], valid)
        positions = [marker.translate.call_args.args[0] for marker in spheres]
        np.testing.assert_array_equal(positions[:2], scene[[1, 0]])
        np.testing.assert_array_equal(positions[2:], raw[[1, 2, 3]])
        # Two nearby raw vertices remain distinct; only repeated raw ID 2 is hidden.
        self.assertIn("queries=2 raw peaks=3", opened.create_window.call_args.kwargs["window_name"])
        self.assertIn("Fusion Q99=0.396", opened.create_window.call_args.kwargs["window_name"])
        self.assertEqual(fake.geometry.TriangleMesh.create_sphere.call_count, 5)
        for record, expected in zip(records, before):
            np.testing.assert_array_equal(record["candidate_raw_indices"], expected)
        np.testing.assert_array_equal(batch["query_indices"], [1, 0])
        opened.run.assert_not_called()

    def test_ball_overview_handles_empty_results_and_cleans_up_on_failure(self):
        empty = np.empty((0, 3))
        scores = dict(ced_score=np.empty(0), ced_valid_mask=np.empty(0, dtype=bool),
                      pca_ratio_saturated=np.empty(0), pca_valid_mask=np.empty(0, dtype=bool),
                      fusion_score=np.empty(0), fusion_valid_mask=np.empty(0, dtype=bool))
        batch = {"query_indices": np.array([], dtype=int), "records": [], "ced_score_result": scores}
        fake = Mock()
        opened = fake.visualization.Visualizer.return_value
        opened.create_window.return_value = True
        with patch.object(view3d, "require_open3d", return_value=fake):
            self.assertIs(view3d.open_extremity_overview(empty, empty, batch), opened)
            fake.geometry.TriangleMesh.create_sphere.assert_not_called()
            opened.create_window.return_value = False
            with self.assertRaisesRegex(RuntimeError, "ball overview"):
                view3d.open_extremity_overview(empty, empty, batch)
        opened.destroy_window.assert_called_once()

    def test_ball_overview_manager_reuses_cache_and_respects_no_3d(self):
        points, result = self.ball_result()
        batch = {"query_indices": np.array([1]), "records": [result]}
        frozen = {"points_raw_mm": points, "points_voxel_mm": points, "extremity_batch": batch}
        with patch.object(manager, "open_extremity_overview") as open_view:
            manager.prepare_extremity_overview(frozen, show_3d=False)
            open_view.assert_not_called()
            manager.prepare_extremity_overview(frozen)
            manager.prepare_extremity_overview(frozen)
            open_view.assert_called_once_with(points, points, batch)
            self.assertIs(frozen["extremity_overview"], open_view.return_value)
        self.assertIs(frozen["extremity_batch"], batch)

    def test_ball_overview_survives_reselection_and_closes_with_frame(self):
        overview, local = Mock(), Mock()
        overview.poll_events.return_value = True
        batch = object()
        frozen = {"extremity_overview": overview, "extremity_views": (local,),
                  "extremity_batch": batch}
        manager.close_query_views(frozen, keep_extremity_overview=True)
        local.destroy_window.assert_called_once()
        overview.destroy_window.assert_not_called()
        manager.poll_view_events(frozen)
        overview.update_renderer.assert_called_once()
        manager.close_query_views(frozen)
        overview.destroy_window.assert_called_once()
        self.assertNotIn("extremity_overview", frozen)
        self.assertIs(frozen["extremity_batch"], batch)

    def test_ball_ui_opens_overview_before_waiting_for_click(self):
        frozen = {"extremity_batch": object()}
        cfg = SimpleNamespace(mode="ball", show_3d=True)
        with patch.object(query_ui, "prepare_extremity_overview") as prepare, \
                patch.object(query_ui, "run_frozen_click_stage", return_value="quit") as stage:
            def wait_for_click(*args, **kwargs):
                prepare.assert_called_once_with(frozen, show_3d=True)
                return "quit"
            stage.side_effect = wait_for_click
            self.assertEqual(query_ui.run_validator_mode(object(), frozen, cfg), "quit")

    def test_open3d_proxy_is_defined_and_loads_on_first_attribute_access(self):
        geometry = object()
        fake_open3d = SimpleNamespace(geometry=geometry)

        with patch.object(
            view3d,
            "require_open3d",
            return_value=fake_open3d,
        ) as require:
            self.assertIs(view3d.o3d.geometry, geometry)

        require.assert_called_once_with()

    def test_validator_opens_frozen_rgb_click_stage_directly(self):
        frame = object()
        frozen = {
            "selected_pixel": (10, 20),
            "pending_query": (30, 40),
        }
        cfg = object()

        with patch.object(
            query_ui,
            "run_frozen_click_stage",
            return_value="quit",
        ) as run_stage:
            action = query_ui.run_validator_mode(frame, frozen, cfg)

        self.assertEqual(action, "quit")
        self.assertNotIn("selected_pixel", frozen)
        self.assertNotIn("pending_query", frozen)
        run_stage.assert_called_once_with(
            frozen,
            cfg,
            allow_reselection=True,
        )

    def test_cornervins_final_views_keep_full_scene_and_exact_roi(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        analysis = {
            "query_position_mm": points[0],
            "query_radius_mm": 1.1,
            "roi_point_indices": np.asarray([0, 1], dtype=np.int64),
        }

        primitive = cornervins_scene_roi_primitives(
            points, analysis, circle_samples=12,
        )

        np.testing.assert_array_equal(primitive["points_mm"], points)
        np.testing.assert_array_equal(primitive["roi_point_indices"], [0, 1])
        self.assertFalse(
            np.allclose(primitive["point_colors"][1], primitive["point_colors"][2])
        )
        np.testing.assert_allclose(
            np.linalg.norm(
                primitive["sphere_points_mm"]
                - primitive["query_position_mm"],
                axis=1,
            ),
            1.1,
        )

    def test_cornervins_ahc_view_contains_exact_query_radius(self):
        organized = np.asarray([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ])
        analysis = {
            "query_position_mm": np.zeros(3),
            "query_radius_mm": 1.1,
            "organized_extraction_radius_mm": 1.1,
            "cornervins_result": SimpleNamespace(
                ahc_regions=(np.asarray([0], dtype=np.int64),),
                final_membership_image=np.asarray([[0, 0], [-1, -1]], dtype=np.int32),
                refined_membership_image=np.asarray([[0, 0], [-1, -1]], dtype=np.int32),
                ransac_patches=(),
            ),
        }

        primitive = cornervins_ahc_primitives(
            organized, analysis, circle_samples=12,
        )

        np.testing.assert_array_equal(
            primitive["inside_roi_mask"], [True, True, False, False],
        )
        self.assertEqual(primitive["sphere_points_mm"].shape, (36, 3))
        self.assertEqual(primitive["sphere_lines"].shape, (36, 2))
        np.testing.assert_allclose(
            np.linalg.norm(
                primitive["sphere_points_mm"]
                - primitive["query_position_mm"],
                axis=1,
            ),
            1.1,
        )
        self.assertFalse(
            np.allclose(primitive["point_colors"][1], primitive["point_colors"][2])
        )

        refined = cornervins_ahc_primitives(
            organized, analysis, stage="refined", circle_samples=12,
        )
        self.assertEqual(refined["stage"], "refined")
        self.assertEqual(refined["region_count"], 1)
        self.assertFalse(
            np.allclose(refined["point_colors"][1], refined["point_colors"][2])
        )

        analysis["cornervins_result"].ransac_patches = (
            SimpleNamespace(
                center=np.asarray([0.5, 0.0, 0.0]),
                normal=np.asarray([0.0, 0.0, 1.0]),
                organized_inlier_indices=np.asarray([0, 1], dtype=np.int64),
            ),
        )
        ransac = cornervins_ahc_primitives(
            organized, analysis, stage="ransac", circle_samples=12,
        )
        self.assertEqual(ransac["stage"], "ransac")
        self.assertEqual(ransac["region_count"], 1)
        np.testing.assert_allclose(ransac["plane_centers_mm"], [[0.5, 0.0, 0.0]])
        np.testing.assert_allclose(ransac["plane_normals"], [[0.0, 0.0, 1.0]])

    def test_centroid_heatmap_uses_fixed_score_scale(self):
        colors = centroid_saliency_colors([0.0, 0.5, 1.0])
        self.assertEqual(colors.shape, (3, 3))
        self.assertFalse(np.allclose(colors[0], colors[-1]))
        np.testing.assert_allclose(colors, centroid_saliency_colors([0.0, 0.5, 1.0]))



if __name__ == "__main__":
    unittest.main()
