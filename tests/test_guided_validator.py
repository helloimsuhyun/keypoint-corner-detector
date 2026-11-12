"""Numerical and native-view interaction contracts for the guided validator."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from corner_detector.guided_ransac import extract_guided_planes, guided_probabilities
from corner_detector.candidate_extraction import greedy_nms_indices
from corner_detector.settings import GuidedValidatorConfig, ValidatorDetectorConfig, configure, parse_args
from corner_detector.validator_pipeline import prepare_validator_voxel_context, run_validator_pipeline
from corner_detector import validator_query
from corner_detector.visualization import manager, query_ui, view3d


def plane_cloud(count):
    rng = np.random.default_rng(7)
    clouds = []
    for axis in range(count):
        points = rng.uniform(-20, 20, (160, 3))
        points[:, axis] = 0
        clouds.append(points)
    return np.vstack(clouds)


class GuidedIntegrationTest(unittest.TestCase):
    def config(self, *extra):
        return configure(parse_args(["--source", "realsense", "--mode", "validator", "--no-validator-geometry", *extra]))[0]

    def test_default_stage_lengths_are_independent_of_coarse_voxel(self):
        cfg = self.config("--voxel-3d-mm", "5")
        self.assertEqual(cfg.validator, replace(GuidedValidatorConfig(), geometry_enabled=False))
        other, _ = configure(parse_args(["--source", "realsense", "--mode", "detector",
                                         "--voxel-3d-mm", "5"]))
        for name in vars(other):
            if name != "validator":
                self.assertEqual(getattr(cfg, name), getattr(other, name))
        self.assertEqual(self.config("--validator-ransac-voxel-mm", "4",
                                     "--validator-ransac-radius-x", "2.2").validator.ransac_radius_mm, 8.8)


    def test_ransac_lengths_use_local_voxel_and_fine_settings_are_removed(self):
        cfg = self.config("--voxel-3d-mm", "7", "--validator-ransac-voxel-mm", "4",
                          "--validator-ransac-radius-x", "3", "--validator-inlier-threshold-x", ".5",
                          "--validator-guided-sigma-x", "2")
        self.assertEqual((cfg.validator.ransac_radius_mm, cfg.validator.inlier_threshold_mm,
                          cfg.validator.guided_sigma_mm), (12., 2., 8.))
        self.assertFalse(any(name.startswith("fine_") for name in vars(cfg.validator)))
        with redirect_stdout(StringIO()), patch("sys.stderr", new=StringIO()), self.assertRaises(SystemExit):
            self.config("--validator-fine-voxel-mm", "2")

    def test_raw_crop_precedes_voxel_even_when_sizes_equal_coarse_and_reuses_cache(self):
        scene = np.array([[0., 0, 0], [10., 0, 0]])
        raw = np.array([[0., 0, 0], [.1, 0, 0], [2., 0, 0], [2.1, 0, 0], [20., 0, 0]])
        cfg = self.config("--voxel-3d-mm", "1", "--validator-ransac-voxel-mm", "1",
                          "--validator-ransac-radius-x", "2", "--validator-max-planes", "0")
        context = prepare_validator_voxel_context(scene, raw, cfg)
        from corner_detector.preprocessing import voxel_downsample_3d
        with patch("corner_detector.preprocessing.voxel_downsample_3d", wraps=voxel_downsample_3d) as down:
            result = run_validator_pipeline(scene, 0, cfg, voxel_context=context)
            np.testing.assert_array_equal(down.call_args_list[0].args[0], raw[:3])
            self.assertEqual(down.call_args_list[0].args[1], 1.)
            calls = down.call_count
            again = run_validator_pipeline(scene, 0, cfg, voxel_context=context)
            self.assertEqual(down.call_count, calls)
        points = result["roi_points_mm"]
        np.testing.assert_allclose(points[np.argsort(points[:, 0])], [[.05, 0, 0], [2., 0, 0]])
        np.testing.assert_array_equal(result["roi_raw_point_indices"], [0, 1, 2])
        np.testing.assert_array_equal(result["query_position_mm"], scene[0])
        np.testing.assert_array_equal(result["ransac_scene_indices"], [0, 0])
        np.testing.assert_array_equal(again["roi_points_mm"], result["roi_points_mm"])
        self.assertTrue(result["planes_computed"])
        self.assertNotIn("fine_result", result)
        with self.assertRaises(ValueError):
            run_validator_pipeline(scene + 1, 0, cfg, voxel_context=context)
        np.testing.assert_array_equal(raw[:, 0], [0., .1, 2., 2.1, 20.])

    def test_halo_is_separate_from_core_voxel_centroids_and_ransac_input(self):
        scene = np.array([[0., 0, 0], [10., 0, 0]])
        raw = np.array([[0., 0, 0], [1.9, 0, 0], [2.1, 0, 0], [3.9, 0, 0], [4.1, 0, 0]])
        cfg = self.config("--validator-ransac-voxel-mm", "1", "--validator-ransac-radius-x", "2",
                          "--validator-plane-support-radius-mm", "4", "--validator-max-planes", "0")
        cfg = replace(cfg, validator=replace(cfg.validator, geometry_enabled=True))
        context = prepare_validator_voxel_context(scene, raw, cfg)
        from corner_detector.validator_pipeline import _query_voxel_cloud
        cloud = _query_voxel_cloud(context, 0, cfg.validator)
        np.testing.assert_array_equal(cloud["raw_core_point_indices"], [0, 1])
        np.testing.assert_allclose(np.sort(cloud["core_points_mm"][:, 0]), [0., 1.9])
        np.testing.assert_allclose(np.sort(cloud["points_mm"][cloud["core_voxel_count"]:, 0]), [2.1, 3.9])
        self.assertEqual(cloud["halo_voxel_count"], 2)
        self.assertTrue(np.all(np.linalg.norm(cloud["core_points_mm"], axis=1) <= 2.))

    def test_empty_query_roi_returns_empty_computed_result(self):
        cfg = self.config("--validator-ransac-radius-x", "1")
        result = run_validator_pipeline(np.zeros((1, 3)), 0, cfg, raw_points_mm=np.array([[1000., 0, 0]]))
        self.assertTrue(result["planes_computed"])
        self.assertEqual(result["roi_points_mm"].shape, (0, 3))
        self.assertEqual(result["planes"], [])
        self.assertEqual(result["ransac_scene_indices"].shape, (0,))

    def test_interactive_coarse_click_immediately_runs_ransac_without_moving_query(self):
        cfg = self.config()
        scene = np.array([[.25, 0, 0]])
        raw = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0]])
        frozen = {"points_voxel_mm": scene, "points_raw_mm": raw,
                  "validator_query_indices": np.array([0])}
        with patch.object(validator_query, "prepare_guided_validator_result"), redirect_stdout(StringIO()):
            result = validator_query.inspect_validator_query(0, frozen, cfg)
        self.assertTrue(result["planes_computed"])
        np.testing.assert_array_equal(result["query_position_mm"], scene[0])
        self.assertNotIn("validator_fine_result", frozen)

    def test_coarse_heatmap_retains_detector_frame_scale(self):
        points = np.column_stack((np.arange(4.), np.zeros((4, 2))))
        values, valid = np.array([.02, .08, .2, 99.]), np.array([True, True, True, False])
        scores = {key: values.copy() for key in ("ced_score", "pca_ratio_saturated", "fusion_score")}
        scores.update({key: valid.copy() for key in ("ced_valid_mask", "pca_valid_mask", "fusion_valid_mask")})
        frozen = {"points_voxel_mm": points, "validator_query_indices": np.array([0]),
                  "validator_ced_scores": scores}
        fake, _, scenes = self.open3d_stub()
        with patch.object(view3d, "require_open3d", return_value=fake):
            display = manager.prepare_guided_validator_view(frozen)
            vmax = np.quantile(values[valid], .99)
            self.assertEqual(display["fusion_vmax"], vmax)
            cloud = scenes[0].add_geometry.call_args_list[0].args[1]
            np.testing.assert_array_equal(cloud.colors, view3d.dense_heatmap_colors(values, valid, vmax=vmax))
            manager.close_query_views(frozen)

    def test_greedy_nms_keeps_second_peak_after_an_intermediate_point_is_suppressed(self):
        points = np.column_stack(([0., 1., 2.], np.zeros((3, 2))))
        np.testing.assert_array_equal(greedy_nms_indices(points, np.array([3., 2., 1.]),
                                                       np.ones(3, bool), 1.1), [0, 2])

    def test_query_generation_disables_only_display_and_post_nms_validator(self):
        cfg = self.config()
        scores = {"candidate_extraction": {"candidate_indices": np.array([4, 2])}}
        points = np.zeros((5, 3))
        with patch.object(validator_query, "run_dense_heatmap_mode", return_value=scores) as dense, \
                patch.object(validator_query, "run_validator_pipeline") as planes:
            result = validator_query.prepare_validator_queries(points, cfg)
        passed = dense.call_args.args[1]
        self.assertIsNone(passed.validator)
        self.assertFalse(passed.show_3d)
        self.assertEqual(passed.nms_radius_mm, cfg.nms_radius_mm)
        self.assertIs(result, scores)
        np.testing.assert_array_equal(result["candidate_extraction"]["candidate_indices"], [4, 2])
        planes.assert_not_called()

    def test_rgb_click_selects_existing_scene_query_and_logs_result(self):
        cfg = self.config("--no-3d-view")
        raw = plane_cloud(1)
        frozen = {"points_raw_mm": raw, "points_voxel_mm": np.vstack(([[0., 0, 0], [500., 0, 0]], raw)),
                  "validator_query_indices": np.array([0]), "pointcloud_frame": object()}
        with patch.object(validator_query, "query_from_pixel", return_value=SimpleNamespace(point_mm=[2., 0, 0])), \
                redirect_stdout(StringIO()) as output:
            result = validator_query.inspect_validator_pixel(10, 20, frozen, cfg)
        self.assertEqual(result["query_index"], 0)
        self.assertIn("[VALIDATOR]", output.getvalue())
        self.assertIn("Inlier Ratio:", output.getvalue())
        self.assertIs(frozen["guided_validator_result"], result)
        with self.assertRaises(ValueError):
            validator_query.inspect_validator_query(1, frozen, cfg)
        frozen["validator_query_indices"] = np.array([], dtype=int)
        with redirect_stdout(StringIO()):
            self.assertIsNone(validator_query.inspect_validator_pixel(10, 20, frozen, cfg))

    def test_visual_overlap_uses_minimum_residual_and_never_mutates_membership(self):
        raw = np.array([[0.1, 1., 0.2], [0.3, 1., 0.1], [9., 9., 9.]])
        planes = [{"normal": np.array([1., 0, 0]), "offset_mm": 0., "inlier_mask": np.array([True, True])},
                  {"normal": np.array([0., 0, 1]), "offset_mm": 0., "inlier_mask": np.array([True, True])}]
        result = {"planes": planes, "roi_points_mm": raw[:2].copy(), "roi_ransac_indices": np.array([0, 1]),
                  "query_position_mm": np.zeros(3), "radius_mm": 30., "query_index": 0}
        before = [p["inlier_mask"].copy() for p in planes]
        data = view3d.guided_validator_primitives(result)
        np.testing.assert_array_equal(data["point_colors"], [[1, 0, 0], [0, 1, 0]])
        for disk, plane, mask in zip(data["disk_vertices_mm"], planes, before):
            np.testing.assert_allclose(np.linalg.norm(disk, axis=1), 30)
            np.testing.assert_allclose(disk @ plane["normal"], 0, atol=1e-12)
            np.testing.assert_array_equal(plane["inlier_mask"], mask)

    def open3d_stub(self):
        fake = Mock()
        widgets = [Mock() for _ in range(8)]
        scenes = [Mock() for _ in range(8)]
        fake.visualization.gui.Application.instance.create_window.side_effect = [Mock() for _ in range(4)]
        fake.visualization.gui.SceneWidget.side_effect = widgets
        fake.visualization.gui.Label.side_effect = Mock
        fake.visualization.rendering.Open3DScene.side_effect = scenes
        fake.visualization.rendering.MaterialRecord.side_effect = Mock
        fake.geometry.PointCloud.side_effect = Mock
        fake.geometry.TriangleMesh.side_effect = Mock
        fake.geometry.TriangleMesh.create_sphere.side_effect = Mock
        fake.geometry.TriangleMesh.create_arrow.side_effect = Mock
        fake.geometry.LineSet.side_effect = Mock
        for name in ("Vector3dVector", "Vector3iVector", "Vector2iVector"):
            getattr(fake.utility, name).side_effect = lambda value: np.array(value, copy=True)
        return fake, widgets, scenes

    def test_geometry_has_its_own_window_and_never_overlays_plane_panel(self):
        points = np.array([[0., 0, 0], [10., 0, 0], [0., 10., 0], [0., 0, 10.]])
        result = {"query_index": 0, "query_position_mm": np.zeros(3),
                  "coarse_query_position_mm": np.zeros(3), "radius_mm": 30.,
                  "ransac_voxel_mm": 5., "roi_raw_point_indices": np.arange(4),
                  "roi_points_mm": points.copy(), "planes_computed": True,
                  "planes": [{"normal": np.array([0., 0, 1.]), "offset_mm": 0.,
                              "inlier_mask": np.array([True, True, True, False])},
                             {"normal": np.array([0., 1., 0]), "offset_mm": 0.,
                              "inlier_mask": np.array([True, True, False, True])}],
                  "geometry_computed": True, "validation_status": "CANDIDATES", "corner_radius_mm": 5.,
                  "support_roi_points_mm": points.copy(), "support_radius_mm": 50.,
                  "support_roi_ransac_indices": np.arange(4),
                  "plane_patches": [{"plane_id": 0, "valid": True,
                                     "support_ransac_indices": np.arange(3),
                                     "hull_vertices_mm": points[:3].copy()},
                                    {"plane_id": 1, "valid": True,
                                     "support_ransac_indices": np.array([0, 1, 3]),
                                     "hull_vertices_mm": points[[0, 1, 3]].copy()}],
                  "edge_candidates": [
                      {"source_edge_id": 3, "edge_type": "boundary", "plane_ids": [0],
                       "valid": False, "start_mm": points[1].copy(), "end_mm": points[2].copy()},
                      {"source_edge_id": 4, "edge_type": "plane_intersection", "plane_ids": [0, 1],
                       "valid": False, "start_mm": points[0].copy(), "end_mm": points[1].copy()}],
                  "edges": [
                      {"edge_id": index, "edge_type": kind, "plane_ids": plane_ids,
                       "start_mm": points[0].copy(), "end_mm": points[endpoint].copy()}
                      for index, kind, plane_ids, endpoint in (
                          (0, "boundary", [0], 2), (1, "boundary", [1], 3),
                          (2, "plane_intersection", [0, 1], 1))],
                  "corners": [{"position": np.zeros(3), "edge_ids": [0, 1, 2],
                               "endpoint_ids": {0: 0, 1: 0, 2: 0},
                               "intersection_residual": 0., "endpoint_gap": 0.}]}
        frozen = {"points_voxel_mm": points, "validator_query_indices": np.array([0]),
                  "validator_scene_colors": np.full(points.shape, .4)}
        fake, widgets, scenes = self.open3d_stub()
        with patch.object(view3d, "require_open3d", return_value=fake), \
                patch.object(view3d, "update_validator_roi_view"):
            display = manager.prepare_guided_validator_view(frozen)
            stages = display["geometry_windows"]
            self.assertEqual([stage["number"] for stage in stages], [4, 5, 6])
            self.assertEqual([len(stage["panels"]) for stage in stages], [1, 3, 1])
            self.assertEqual(display["app"].create_window.call_count, 4)
            manager.prepare_guided_validator_result(frozen, result)
            plane_names = {call.args[0] for call in scenes[2].add_geometry.call_args_list}
            self.assertEqual(plane_names, {"roi", "radius", "coarse_query",
                                          "plane_0", "outline_0", "normal_0",
                                          "plane_1", "outline_1", "normal_1"})
            names = [{call.args[0] for call in scene.add_geometry.call_args_list} for scene in scenes]
            self.assertEqual({name for name in names[3] if name.startswith("finite_edge_")}, {"finite_edge_2"})
            self.assertEqual({name for name in names[4] if name.startswith("finite_edge_")}, {"finite_edge_0"})
            self.assertEqual({name for name in names[5] if name.startswith("finite_edge_")}, {"finite_edge_1"})
            self.assertEqual(names[6], set())
            self.assertIn("not observed", stages[1]["panels"][2]["label"].text)
            self.assertTrue({"finite_edge_0", "finite_edge_1", "finite_edge_2", "corner_vertex_0"} <= names[7])
            self.assertIn("unknown_edge_4", names[3])
            self.assertNotIn("unknown_edge_3", names[3])
            self.assertIn("unknown_edge_3", names[4])
            self.assertNotIn("unknown_edge_4", names[4])
            for scene_names in names[3:7]:
                self.assertNotIn("corner_vertex_0", scene_names)
                self.assertNotIn("core_radius", scene_names)
            cloud = next(call.args[1] for call in scenes[4].add_geometry.call_args_list
                         if call.args[0] == "support_context")
            np.testing.assert_array_equal(cloud.points, points[:3])
            for scene_index, edge_id, expected in ((3, 2, [0., .9, 1.]), (4, 0, [1., .6, .1]),
                                                    (5, 1, [1., .6, .1]), (7, 2, [.1, 1., .2])):
                line = next(call.args[1] for call in scenes[scene_index].add_geometry.call_args_list
                            if call.args[0] == f"finite_edge_{edge_id}")
                line.paint_uniform_color.assert_called_with(expected)
            material = next(call.args[2] for call in scenes[2].add_geometry.call_args_list
                            if call.args[0] == "plane_0")
            self.assertEqual(material.base_color[3], .60)
            np.testing.assert_allclose(scenes[2].camera.look_at.call_args.args[1], [51., 39., 45.])
            np.testing.assert_allclose(scenes[3].camera.look_at.call_args.args[1], [85., 65., 75.])
            calls = [scene.add_geometry.call_count for scene in scenes[3:]]
            manager.prepare_guided_validator_result(frozen, {**result, "planes_computed": False})
            self.assertEqual([scene.add_geometry.call_count for scene in scenes[3:]], calls)
            for stage in stages:
                for panel in stage["panels"]:
                    self.assertIn("NOT_COMPUTED", panel["label"].text)
            manager.close_query_views(frozen)
            display["window"].close.assert_called_once()
            for stage in stages:
                stage["window"].close.assert_called_once()
        np.testing.assert_array_equal(result["planes"][0]["inlier_mask"], [True, True, True, False])
        np.testing.assert_array_equal(result["plane_patches"][0]["hull_vertices_mm"], points[:3])


    def test_open3d_uploads_scene_once_and_replaces_only_roi(self):
        raw = plane_cloud(3)
        scene = np.vstack((np.zeros((1, 3)), raw))
        result = run_validator_pipeline(scene, 0, self.config("--no-validator-geometry"), raw_points_mm=raw)
        for plane in result["planes"]:
            native_ids = plane["inlier_ransac_indices"]
            np.testing.assert_array_equal(result["ransac_points_mm"][native_ids],
                                          result["roi_points_mm"][plane["inlier_mask"]])
            np.testing.assert_array_equal(plane["inlier_point_indices"],
                                          result["ransac_scene_indices"][native_ids])
        before = [plane["inlier_mask"].copy() for plane in result["planes"]]
        frozen = {"points_raw_mm": raw, "points_voxel_mm": scene,
                  "validator_query_indices": np.array([0]), "validator_scene_colors": np.full(scene.shape, .4),
                  "validator_fusion_vmax": .2}
        fake, widgets, scenes = self.open3d_stub()
        with patch.object(view3d, "require_open3d", return_value=fake):
            display = manager.prepare_guided_validator_view(frozen)
            overview_camera_calls = widgets[0].setup_camera.call_count
            overview_uploads = scenes[0].add_geometry.call_count
            overview_cloud = scenes[0].add_geometry.call_args_list[0].args[1]
            np.testing.assert_array_equal(overview_cloud.points, frozen["points_voxel_mm"])
            np.testing.assert_array_equal(overview_cloud.colors, frozen["validator_scene_colors"])
            manager.prepare_guided_validator_result(frozen, result)
            roi_cloud = scenes[1].add_geometry.call_args_list[0].args[1]
            np.testing.assert_array_equal(roi_cloud.points, result["roi_points_mm"])
            roi_camera_calls = widgets[1].setup_camera.call_count
            self.assertIn("Raw core:", display["labels"][1].text)
            first_count = scenes[2].add_geometry.call_count
            self.assertEqual(first_count, 3 + 3 * len(result["planes"]))
            cloud = scenes[2].add_geometry.call_args_list[0].args[1]
            np.testing.assert_array_equal(cloud.points, result["roi_points_mm"])
            self.assertEqual(len(cloud.colors), len(result["roi_ransac_indices"]))
            for call in scenes[2].add_geometry.call_args_list:
                if call.args[0].startswith("plane_"):
                    self.assertEqual(call.args[2].shader, "defaultLitTransparency")
                    self.assertEqual(call.args[2].base_color[3], 0.60)
                    plane = result["planes"][int(call.args[0].split("_")[-1])]
                    center = call.args[1].vertices[0]
                    self.assertAlmostEqual(center @ plane["normal"] + plane["offset_mm"], 0.)
                if call.args[0].startswith("outline_"):
                    self.assertEqual(call.args[2].shader, "unlitLine")
                    self.assertEqual(call.args[2].line_width, 3.0)
                    self.assertEqual(len(call.args[1].lines), 98)
            manager.prepare_guided_validator_result(frozen, result)
            self.assertEqual(scenes[2].clear_geometry.call_count, 2)
            self.assertEqual(scenes[2].add_geometry.call_count, first_count * 2)
            self.assertEqual(scenes[0].add_geometry.call_count, overview_uploads)
            self.assertEqual(widgets[0].setup_camera.call_count, overview_camera_calls)
            self.assertEqual(widgets[1].setup_camera.call_count, roi_camera_calls)
            self.assertFalse(any(call.args[0].startswith("plane_") for call in scenes[1].add_geometry.call_args_list))
            pending = {**result, "planes": [], "planes_computed": False}
            manager.prepare_guided_validator_result(frozen, pending)
            self.assertEqual(scenes[2].clear_geometry.call_count, 3)
            self.assertEqual(scenes[2].add_geometry.call_count, first_count * 2)
            self.assertIn("select a Coarse query", display["labels"][2].text)
            manager.close_query_views(frozen, keep_guided_validator=True)
            self.assertIs(frozen["guided_validator_display"], display)
            manager.close_query_views(frozen)
            manager.close_query_views(frozen)
            display["window"].close.assert_called_once()
            self.assertNotIn("guided_validator_display", frozen)
        for mask, plane in zip(before, result["planes"]):
            np.testing.assert_array_equal(plane["inlier_mask"], mask)

    def test_open3d_picker_handles_camera_screen_coordinates_and_hidden_points(self):
        camera = Mock()
        # Perspective projection: w = -z, so positive z is behind the camera.
        projection = np.eye(4)
        projection[3] = [0, 0, -1, 0]
        camera.get_projection_matrix.return_value = projection
        camera.get_view_matrix.return_value = np.eye(4)
        widget = SimpleNamespace(scene=SimpleNamespace(camera=camera),
                                 frame=SimpleNamespace(x=10, y=20, width=400, height=200))
        display = {"overview": widget, "query_xyz": np.array([[0., 0, -1], [0., 0, 1], [.5, .5, -1]]),
                   "query_indices": np.array([7, 3, 5])}
        self.assertEqual(manager.pick_guided_validator_query(display, 210, 120), 7)
        self.assertEqual(manager.pick_guided_validator_query(display, 310, 70), 5)
        self.assertIsNone(manager.pick_guided_validator_query(display, -100, -100))
        display["query_xyz"] = np.empty((0, 3))
        self.assertIsNone(manager.pick_guided_validator_query(display, 210, 120))

    def test_renderer_process_routes_events_results_and_terminates_stalled_teardown(self):
        context = Mock()
        parent, child = Mock(), Mock()
        context.Pipe.return_value = (parent, child)
        process = context.Process.return_value
        values = np.array([.1, .4, 10., 200.])
        valid = np.array([True, True, True, False])
        scores = {"ced_score": values.copy(), "ced_valid_mask": valid.copy(),
                  "pca_ratio_saturated": values.copy(), "pca_valid_mask": valid.copy(),
                  "fusion_score": values.copy(), "fusion_valid_mask": valid.copy()}
        frozen = {"points_raw_mm": plane_cloud(1), "points_voxel_mm": np.zeros((4, 3)),
                  "validator_query_indices": np.array([0]), "validator_ced_scores": scores}
        with patch("multiprocessing.get_context", return_value=context) as get_context:
            manager.start_guided_validator_view(frozen)
        get_context.assert_called_once_with("spawn")
        expected_colors = view3d.dense_heatmap_colors(values, valid, vmax=np.quantile(values[valid], .99))
        np.testing.assert_array_equal(context.Process.call_args.kwargs["args"][-2], expected_colors)
        self.assertEqual(context.Process.call_args.kwargs["args"][-1], np.quantile(values[valid], .99))
        np.testing.assert_array_equal(scores["fusion_score"], values)
        np.testing.assert_array_equal(scores["fusion_valid_mask"], valid)
        np.testing.assert_array_equal(frozen["validator_query_indices"], [0])
        process.start.assert_called_once()
        child.close.assert_called_once()
        parent.poll.side_effect = [True, True, False]
        parent.recv.side_effect = [("query", 0), ("action", "recapture")]
        manager.poll_view_events(frozen)
        self.assertEqual(frozen["pending_validator_query"], 0)
        self.assertEqual(frozen["validator_view_action"], "recapture")
        result = {"planes": []}
        manager.prepare_guided_validator_result(frozen, result)
        parent.send.assert_called_with(("result", result))
        process.is_alive.return_value = True
        manager.close_query_views(frozen)
        parent.send.assert_called_with(("close", None))
        process.terminate.assert_called_once()
        self.assertEqual(process.join.call_count, 2)
        parent.close.assert_called_once()
        self.assertNotIn("guided_validator_process", frozen)

    def test_window_click_rotation_keys_and_user_close_cleanup(self):
        frozen = {"points_raw_mm": plane_cloud(1), "points_voxel_mm": np.zeros((1, 3)),
                  "validator_query_indices": np.array([0]), "validator_scene_colors": np.array([[.1, .4, .8]])}
        fake, widgets, _ = self.open3d_stub()
        request_action = Mock()
        gui = fake.visualization.gui
        with patch.object(view3d, "require_open3d", return_value=fake), \
                patch.object(query_ui, "pick_guided_validator_query", return_value=0):
            query_ui.open_guided_validator_window(frozen, request_action)
            display = frozen["guided_validator_display"]
            mouse = widgets[0].set_on_mouse.call_args.args[0]
            def event(kind, x=20, y=20):
                return SimpleNamespace(type=kind, x=x, y=y, is_button_down=lambda _: True)
            mouse(event(gui.MouseEvent.Type.BUTTON_DOWN))
            mouse(event(gui.MouseEvent.Type.DRAG, 40, 40))
            mouse(event(gui.MouseEvent.Type.BUTTON_UP))
            self.assertNotIn("pending_validator_query", frozen)
            mouse(event(gui.MouseEvent.Type.BUTTON_DOWN))
            mouse(event(gui.MouseEvent.Type.BUTTON_UP))
            self.assertEqual(frozen["pending_validator_query"], 0)
            widgets[1].set_on_mouse.assert_not_called()
            widgets[2].set_on_mouse.assert_not_called()
            widgets[2].set_on_key.assert_called_once()
            for widget in widgets[3:]:
                widget.set_on_mouse.assert_not_called()
                widget.set_on_key.assert_called_once()
            key = widgets[1].set_on_key.call_args.args[0]
            key(SimpleNamespace(type=gui.KeyEvent.DOWN, key=gui.KeyName.R))
            request_action.assert_called_once_with("recapture")
            stages = display["geometry_windows"]
            geometry_close = stages[0]["window"].set_on_close.call_args.args[0]
            self.assertTrue(geometry_close())
            self.assertFalse(display["closed"])
            self.assertFalse(stages[1]["closed"])
            self.assertFalse(stages[2]["closed"])
            close_callback = display["window"].set_on_close.call_args.args[0]
            self.assertTrue(close_callback())
            stages[0]["window"].close.assert_not_called()
            stages[1]["window"].close.assert_called_once()
            stages[2]["window"].close.assert_called_once()
            display["app"].run_one_tick.return_value = False
            manager.poll_view_events(frozen)
            self.assertNotIn("guided_validator_display", frozen)
            manager.close_query_views(frozen)
            display["window"].close.assert_not_called()


class GuidedNumericsTest(unittest.TestCase):
    def offset_planes(self, count):
        rng = np.random.default_rng(7)
        planes = []
        for axis in range(count):
            points = rng.uniform(-15, 15, (160, 3))
            points[:, axis] = (-3., 4., 7.)[axis]
            planes.append(points)
        return np.vstack(planes)

    def test_one_two_three_unanchored_planes_disjoint_support_and_reproducibility(self):
        cfg = GuidedValidatorConfig(inlier_threshold_mm=.01)
        for count in (1, 2, 3):
            with self.subTest(count=count):
                raw = self.offset_planes(count)
                before = raw.copy()
                result = extract_guided_planes(raw, np.zeros(3), cfg)
                self.assertEqual(len(result["planes"]), count)
                covered = np.zeros(len(raw), dtype=bool)
                for plane in result["planes"]:
                    mask = plane["inlier_mask"]
                    self.assertFalse(np.any(covered & mask))
                    self.assertEqual(plane["inlier_count"], 160)
                    self.assertGreater(plane["query_to_plane_distance_mm"], 2.9)
                    self.assertLess(plane["inlier_rms_mm"], 1e-12)
                    np.testing.assert_array_equal(plane["inlier_ransac_indices"], np.flatnonzero(mask))
                    self.assertAlmostEqual(plane["inlier_ratio"], 1 / count)
                    residual = np.abs(raw[mask] @ plane["normal"] + plane["offset_mm"])
                    self.assertTrue(np.all(residual < cfg.inlier_threshold_mm))
                    covered |= mask
                repeated = extract_guided_planes(raw, np.zeros(3), cfg)
                for plane, other in zip(result["planes"], repeated["planes"]):
                    np.testing.assert_array_equal(plane["normal"], other["normal"])
                    np.testing.assert_array_equal(plane["inlier_ransac_indices"], other["inlier_ransac_indices"])
                np.testing.assert_array_equal(raw, before)
                print(f"[GUIDED synthetic] {count} planes: " + ", ".join(
                    f"inliers={p['inlier_count']} RMS={p['inlier_rms_mm']:.3g} mm" for p in result["planes"]))

    def test_guided_probability_formula_underflow_and_uniform_fallback(self):
        points = np.array([[0., 0, 0], [10., 0, 0], [20., 0, 0]])
        weights = np.exp(-.5 * (points[:, 0] / 10.) ** 2)
        np.testing.assert_allclose(guided_probabilities(points, np.zeros(3), 10.), weights / weights.sum())
        far = np.array([[1000., 0, 0], [1000.001, 0, 0]])
        prob = guided_probabilities(far, np.zeros(3), 1.)
        self.assertAlmostEqual(prob.sum(), 1.)
        self.assertTrue(np.all(prob > 0))
        self.assertGreater(prob[0], prob[1])
        np.testing.assert_allclose(guided_probabilities(np.full((3, 3), 1e300), np.zeros(3), 1e-300), [1/3] * 3)

    def test_rms_breaks_count_ties_and_only_first_sample_is_guided(self):
        raw = np.array([[2., 0, 0], [0., 2, 0], [0., 0, 0], [0., 0, 4], [0., 2, 4]])
        rng = Mock()
        rng.choice.side_effect = [0, np.array([0, 1]), 1, np.array([1, 2])]
        cfg = GuidedValidatorConfig(max_planes=1, ransac_iterations=2, min_inliers=3,
                                    min_inlier_ratio=0., inlier_threshold_mm=10.)
        with patch("corner_detector.guided_ransac.np.random.default_rng", return_value=rng):
            plane = extract_guided_planes(raw, np.zeros(3), cfg)["planes"][0]
        np.testing.assert_allclose(np.abs(plane["normal"]), [1, 0, 0])
        self.assertEqual(plane["inlier_count"], len(raw))
        for call in rng.choice.call_args_list[::2]:
            self.assertIn("p", call.kwargs)
        for call in rng.choice.call_args_list[1::2]:
            self.assertEqual(call.kwargs, {"size": 2, "replace": False})
            self.assertEqual(call.args[0], len(raw) - 1)

    def test_count_beats_nearby_weighted_support(self):
        near = np.array([[x, y, 0.] for x in (2., 3., 4.) for y in (2., 3., 4.)])
        far = np.array([[0., y, z] for y in (18., 19., 20., 21.) for z in (10., 11., 12.)])
        raw = np.vstack((near, far))
        rng = Mock()
        rng.choice.side_effect = [0, np.array([0, 2]), 9, np.array([9, 11])]
        cfg = GuidedValidatorConfig(max_planes=1, ransac_iterations=2, min_inliers=3,
                                    min_inlier_ratio=0., inlier_threshold_mm=.01)
        with patch("corner_detector.guided_ransac.np.random.default_rng", return_value=rng):
            plane = extract_guided_planes(raw, np.zeros(3), cfg)["planes"][0]
        self.assertEqual(plane["inlier_count"], 12)
        np.testing.assert_allclose(np.abs(plane["normal"]), [1, 0, 0])

    def test_minimum_is_based_on_original_roi_after_removal(self):
        raw = self.offset_planes(2)
        raw = np.vstack((raw[:100], raw[160:190]))
        cfg = GuidedValidatorConfig(min_inlier_ratio=.3, inlier_threshold_mm=.01)
        result = extract_guided_planes(raw, np.zeros(3), cfg)
        self.assertEqual(result["min_required"], 39)
        self.assertEqual(len(result["planes"]), 1)
        self.assertEqual(result["planes"][0]["inlier_count"], 100)

    def test_strict_threshold_and_raw_roi_indices(self):
        raw = np.array([[100., 0, 0], [0., 0, 0], [1., 0, 0], [0., 1, 0],
                        [0., 0, 1.], [0., 0, .999], [30., 0, 0], [30.00001, 0, 0]])
        rng = Mock()
        rng.choice.side_effect = [0, np.array([0, 1])]
        cfg = GuidedValidatorConfig(max_planes=1, ransac_iterations=1, min_inliers=3,
                                    min_inlier_ratio=0., inlier_threshold_mm=1., ransac_radius_mm=30.)
        with patch("corner_detector.guided_ransac.np.random.default_rng", return_value=rng):
            result = extract_guided_planes(raw, np.zeros(3), cfg)
        np.testing.assert_array_equal(result["roi_ransac_indices"], [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(result["planes"][0]["inlier_ransac_indices"], [1, 2, 3, 5, 6])

    def test_degenerate_small_empty_clouds_and_zero_limit(self):
        cfg = GuidedValidatorConfig(min_inliers=3)
        clouds = [np.empty((0, 3)), np.zeros((2, 3)), np.zeros((30, 3)),
                  np.column_stack((np.linspace(-10, 10, 30), np.zeros((30, 2)))),
                  np.random.default_rng(1).normal(size=(30, 3)) * 1e-12]
        for raw in clouds:
            self.assertEqual(extract_guided_planes(raw, np.zeros(3), cfg)["planes"], [])
        self.assertEqual(extract_guided_planes(self.offset_planes(1), np.zeros(3),
                                              replace(cfg, max_planes=0))["planes"], [])

    def test_configuration_validation(self):
        for changes in ({"ransac_radius_mm": 0.}, {"guided_sigma_mm": np.nan},
                        {"ransac_voxel_mm": -1.},
                        {"max_planes": 4}, {"min_inliers": 2}, {"min_inlier_ratio": 1.1},
                        {"ransac_iterations": 1.5}, {"random_seed": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(GuidedValidatorConfig(), **changes).validate()


if __name__ == "__main__":
    unittest.main()
