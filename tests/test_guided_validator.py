"""Numerical and native-view interaction contracts for the guided validator."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from corner_detector.guided_ransac import extract_guided_planes, guided_probabilities
from corner_detector.validator_fine import find_fine_queries
from corner_detector.edge_suppression import dense_ced_weighted_pca_scores
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
        return configure(parse_args(["--source", "realsense", "--mode", "validator", *extra]))[0]

    def test_default_stage_lengths_are_independent_of_coarse_voxel(self):
        cfg = self.config("--voxel-3d-mm", "5")
        self.assertEqual(cfg.validator, GuidedValidatorConfig())
        other, _ = configure(parse_args(["--source", "realsense", "--mode", "detector",
                                         "--voxel-3d-mm", "5"]))
        for name in vars(other):
            if name != "validator":
                self.assertEqual(getattr(cfg, name), getattr(other, name))
        self.assertEqual(self.config("--validator-ransac-radius-x", "2.2").validator.ransac_radius_mm, 22)

    def test_fine_and_ransac_lengths_use_their_own_voxels(self):
        cfg = self.config("--voxel-3d-mm", "7", "--validator-fine-voxel-mm", "2",
                          "--validator-ransac-voxel-mm", "4", "--validator-fine-nms-radius-x", "1.5",
                          "--validator-fine-ced-radius-x", "3")
        stage = cfg.validator
        self.assertEqual((stage.fine_search_radius_mm, stage.fine_ced_radius_mm, stage.fine_nms_radius_mm),
                         (18., 6., 3.))
        self.assertEqual((stage.fine_pca_radius_mm, stage.fine_bilateral_radius_mm,
                          stage.fine_bilateral_sigma_spatial_mm), (9., 6., 3.))
        self.assertEqual((stage.ransac_radius_mm, stage.inlier_threshold_mm, stage.guided_sigma_mm),
                         (24., 2., 4.))
        self.assertEqual((stage.max_planes, stage.ransac_iterations, stage.min_inliers,
                          stage.min_inlier_ratio, stage.random_seed), (3, 256, 20, .08, 42))
        changed = self.config("--validator-fine-voxel-mm", "2", "--validator-ransac-voxel-mm", "4",
                              "--validator-fine-search-radius-x", "5", "--validator-guided-sigma-x", "2")
        self.assertEqual(changed.validator.fine_search_radius_mm, 10.)
        self.assertEqual(changed.validator.guided_sigma_mm, 8.)

    def test_fine_scoring_cli_controls_are_independent_and_voxel_scaled(self):
        cfg = self.config("--validator-fine-voxel-mm", "2", "--validator-fine-pca-radius-x", "5",
            "--validator-fine-bilateral-radius-x", "4", "--validator-fine-bilateral-sigma-spatial-x", "2",
            "--validator-fine-bilateral-sigma-ced", ".2", "--validator-fine-min-neighbors", "8",
            "--validator-fine-pca-ratio-saturation-tau", ".3", "--validator-fine-fusion-threshold-mode", "fixed",
            "--validator-fine-fusion-score-threshold", ".15", "--validator-fine-nms-method", "local_maximum")
        fine = cfg.validator
        self.assertEqual((fine.fine_pca_radius_mm, fine.fine_bilateral_radius_mm,
                          fine.fine_bilateral_sigma_spatial_mm), (10., 8., 4.))
        self.assertEqual((fine.fine_bilateral_sigma_ced, fine.fine_min_neighbors,
                          fine.fine_pca_ratio_saturation_tau), (.2, 8, .3))
        self.assertEqual((fine.fine_fusion_threshold_mode, fine.fine_fusion_score_threshold,
                          fine.fine_nms_method), ("fixed", .15, "local_maximum"))
        self.assertEqual(cfg.pca_radius_factor, 1.5)

    def test_fine_peak_relative_cli_and_default_leave_coarse_threshold_unchanged(self):
        default = self.config()
        self.assertEqual(default.validator.fine_fusion_threshold_mode, "peak_relative")
        self.assertEqual(default.validator.fine_fusion_peak_relative_ratio, .5)
        cfg = self.config("--validator-fine-fusion-threshold-mode", "peak-relative",
                          "--validator-fine-fusion-peak-relative-ratio", ".7")
        self.assertEqual(cfg.validator.fine_fusion_peak_relative_ratio, .7)
        self.assertEqual(cfg.fusion_threshold_mode, default.fusion_threshold_mode)
        self.assertEqual(cfg.fusion_score_threshold, default.fusion_score_threshold)

    def test_stage_context_downsamples_independently_and_reuses_equal_voxels(self):
        scene = np.array([[0., 0, 0], [50., 0, 0]])
        raw = np.array([[1., 0, 0], [2., 0, 0], [48., 0, 0], [49., 0, 0]])
        fine = np.array([[1., 0, 0], [49., 0, 0]])
        ransac = np.array([[2., 0, 0], [3., 0, 0], [48., 0, 0]])
        cfg = self.config("--validator-fine-voxel-mm", "2", "--validator-ransac-voxel-mm", "4")
        with patch("corner_detector.preprocessing.voxel_downsample_3d", side_effect=[fine, ransac]) as down:
            context = prepare_validator_voxel_context(scene, raw, cfg)
        self.assertEqual([call.args[1] for call in down.call_args_list], [2., 4.])
        np.testing.assert_array_equal(context["fine"]["points_mm"], fine)
        np.testing.assert_array_equal(context["ransac"]["points_mm"], ransac)
        np.testing.assert_array_equal(context["fine"]["scene_indices"], [0, 1])

        np.testing.assert_array_equal(context["ransac"]["scene_indices"], [0, 0, 1])
        equal = self.config("--validator-fine-voxel-mm", "2", "--validator-ransac-voxel-mm", "2")
        with patch("corner_detector.preprocessing.voxel_downsample_3d", return_value=fine) as down:
            context = prepare_validator_voxel_context(scene, raw, equal)
        down.assert_called_once()
        self.assertIs(context["fine"], context["ransac"])
        with patch("corner_detector.preprocessing.voxel_downsample_3d") as down:
            context = prepare_validator_voxel_context(scene, raw, self.config("--validator-fine-voxel-mm", "10"))
        down.assert_not_called()
        self.assertIs(context["fine"]["points_mm"], scene)
        np.testing.assert_array_equal(context["fine"]["scene_indices"], [0, 1])

    def test_real_stage_voxels_produce_expected_centroids(self):
        raw = np.array([[0., 0, 0], [.1, 0, 0], [2., 0, 0], [2.1, 0, 0]])
        scene = np.array([[1.05, 0, 0]])
        cfg = self.config("--voxel-3d-mm", "20", "--validator-fine-voxel-mm", "1",
                          "--validator-ransac-voxel-mm", "10")
        context = prepare_validator_voxel_context(scene, raw, cfg)
        fine = context["fine"]["points_mm"]
        np.testing.assert_allclose(fine[np.argsort(fine[:, 0])], [[.05, 0, 0], [2.05, 0, 0]])
        np.testing.assert_allclose(context["ransac"]["points_mm"], [[1.05, 0, 0]])

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
        widgets = [Mock(), Mock(), Mock()]
        scenes = [Mock(), Mock(), Mock()]
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

    def test_open3d_uploads_scene_once_and_replaces_only_roi(self):
        raw = plane_cloud(3)
        scene = np.vstack((np.zeros((1, 3)), raw))
        result = run_validator_pipeline(scene, 0, self.config(), raw_points_mm=raw)
        before = [plane["inlier_mask"].copy() for plane in result["planes"]]
        frozen = {"points_raw_mm": raw, "points_voxel_mm": scene,
                  "validator_query_indices": np.array([0]), "validator_scene_colors": np.full(scene.shape, .4)}
        fake, widgets, scenes = self.open3d_stub()
        with patch.object(view3d, "require_open3d", return_value=fake):
            display = manager.prepare_guided_validator_view(frozen)
            overview_camera_calls = widgets[0].setup_camera.call_count
            overview_uploads = scenes[0].add_geometry.call_count
            overview_cloud = scenes[0].add_geometry.call_args_list[0].args[1]
            np.testing.assert_array_equal(overview_cloud.points, frozen["points_voxel_mm"])
            np.testing.assert_array_equal(overview_cloud.colors, frozen["validator_scene_colors"])
            manager.prepare_guided_validator_result(frozen, result)
            fine = result["fine_result"]
            fine_scores = fine["scores"]
            fine_cloud = scenes[1].add_geometry.call_args_list[0].args[1]
            np.testing.assert_array_equal(fine_cloud.points, fine_scores["query_position_mm"])
            np.testing.assert_array_equal(fine_cloud.colors, view3d.dense_heatmap_colors(
                fine_scores["fusion_score"], fine_scores["fusion_valid_mask"],
                vmax=view3d.dense_heatmap_vmax(fine_scores)["fusion_score"]))
            fine_camera_calls = widgets[1].setup_camera.call_count
            fine_upload_count = sum(call.args[0] == "fine_heatmap" for call in scenes[1].add_geometry.call_args_list)
            first_count = scenes[2].add_geometry.call_count
            self.assertEqual(first_count, 4 + 3 * len(result["planes"]))
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
            self.assertEqual(widgets[1].setup_camera.call_count, fine_camera_calls)
            self.assertEqual(sum(call.args[0] == "fine_heatmap" for call in scenes[1].add_geometry.call_args_list),
                             fine_upload_count)
            self.assertFalse(any(call.args[0].startswith("plane_") for call in scenes[1].add_geometry.call_args_list))
            pending = {**result, "planes": [], "planes_computed": False}
            manager.prepare_guided_validator_result(frozen, pending)
            self.assertEqual(scenes[2].clear_geometry.call_count, 3)
            self.assertEqual(scenes[2].add_geometry.call_count, first_count * 2)
            self.assertIn("click a Fine NMS", display["labels"][2].text)
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
        np.testing.assert_array_equal(context.Process.call_args.kwargs["args"][-1], expected_colors)
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
                patch.object(query_ui, "pick_guided_validator_query", return_value=0), \
                patch.object(query_ui, "pick_validator_fine_query", return_value=7):
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
            display["fine_result"] = {"coarse_query_index": 0}
            fine_mouse = widgets[1].set_on_mouse.call_args.args[0]
            fine_mouse(event(gui.MouseEvent.Type.BUTTON_DOWN))
            fine_mouse(event(gui.MouseEvent.Type.BUTTON_UP))
            self.assertEqual(frozen["pending_validator_fine_query"], (0, 7))
            widgets[2].set_on_mouse.assert_not_called()
            widgets[2].set_on_key.assert_called_once()
            key = widgets[1].set_on_key.call_args.args[0]
            key(SimpleNamespace(type=gui.KeyEvent.DOWN, key=gui.KeyName.R))
            request_action.assert_called_once_with("recapture")
            close_callback = display["window"].set_on_close.call_args.args[0]
            self.assertTrue(close_callback())
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
                        {"fine_voxel_mm": 0.}, {"ransac_voxel_mm": -1.},
                        {"fine_pca_radius_mm": 0.}, {"fine_min_neighbors": 2},
                        {"fine_bilateral_sigma_ced": 0.}, {"fine_pca_ratio_saturation_tau": -1.},
                        {"fine_pca_ratio_saturation_tau": np.nan}, {"fine_fusion_threshold_mode": "bad"},
                        {"fine_fusion_score_threshold": 2.}, {"fine_nms_method": "bad"},
                        {"fine_fusion_peak_relative_ratio": -.1}, {"fine_fusion_peak_relative_ratio": 1.1},
                        {"fine_fusion_peak_relative_ratio": np.nan},
                        {"max_planes": 4}, {"min_inliers": 2}, {"min_inlier_ratio": 1.1},
                        {"ransac_iterations": 1.5}, {"random_seed": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(GuidedValidatorConfig(), **changes).validate()


class FineCedTest(unittest.TestCase):
    def cfg(self):
        return ValidatorDetectorConfig(centroid_radius_mm=10., pca_radius_factor=1.5,
            bilateral_radius_mm=10., bilateral_sigma_spatial_mm=5., min_neighbors=3,
            validator=GuidedValidatorConfig(fine_voxel_mm=10., ransac_voxel_mm=10.,
                fine_search_radius_mm=6., fine_ced_radius_mm=3.,
                fine_nms_radius_mm=3., fine_min_neighbors=3, fine_pca_radius_mm=4.5,
                fine_bilateral_radius_mm=3., fine_bilateral_sigma_spatial_mm=1.5,
                fine_fusion_threshold_mode="global_mean"))

    def test_halo_scores_match_full_original_cloud_and_differ_from_clipped_roi(self):
        raw = np.random.default_rng(12).uniform(-12, 12, (1200, 3))
        cfg = self.cfg()
        fine = find_fine_queries(raw, np.zeros(3), cfg)
        reference = dense_ced_weighted_pca_scores(raw, 3., 4.5, 3,
            bilateral_radius_mm=3., bilateral_sigma_spatial_mm=1.5,
            bilateral_sigma_ced=cfg.validator.fine_bilateral_sigma_ced,
            pca_ratio_saturation_tau=cfg.validator.fine_pca_ratio_saturation_tau)
        ids = fine["evaluation_fine_indices"]
        self.assertLess(len(fine["support_fine_indices"]), len(raw))
        self.assertGreater(len(fine["support_fine_indices"]), len(ids))
        for key in ("ced_score", "bilateral_ced_score", "pca_ratio_raw", "fusion_score"):
            np.testing.assert_allclose(fine["scores"][key], reference[key][ids], atol=1e-12)
        clipped = dense_ced_weighted_pca_scores(raw[ids], 3., 4.5, 3)
        self.assertGreater(np.max(np.abs(clipped["ced_score"] - fine["scores"]["ced_score"])), 1e-3)
        np.testing.assert_array_equal(fine["scores"]["original_point_indices"], ids)

    def test_greedy_nms_keeps_second_peak_after_an_intermediate_point_is_suppressed(self):
        points = np.array([[0., 0, 0], [2., 0, 0], [4., 0, 0]])
        np.testing.assert_array_equal(greedy_nms_indices(points, [1., .8, .7], [True]*3, 3.), [0, 2])
        np.testing.assert_array_equal(greedy_nms_indices(points, [1., 1., 1.], [True]*3, 3.), [0, 2])

    def test_coarse_tuning_does_not_change_fine_scores_or_candidates(self):
        points = np.random.default_rng(12).uniform(-12, 12, (1200, 3))
        cfg = self.cfg()
        original = find_fine_queries(points, np.zeros(3), cfg)
        changed = replace(cfg, centroid_radius_mm=80., pca_radius_factor=3., min_neighbors=30,
            bilateral_radius_mm=100., bilateral_sigma_spatial_mm=50., bilateral_sigma_ced=.9,
            pca_ratio_saturation_tau=None, fusion_threshold_mode="fixed", fusion_score_threshold=.99)
        fine = find_fine_queries(points, np.zeros(3), changed)
        for key in ("ced_score", "bilateral_ced_score", "pca_ratio_raw", "pca_ratio_saturated", "fusion_score"):
            np.testing.assert_array_equal(fine["scores"][key], original["scores"][key])
        np.testing.assert_array_equal(fine["candidate_fine_indices"], original["candidate_fine_indices"])
        self.assertEqual(fine["fusion_score_threshold"], original["fusion_score_threshold"])
        self.assertGreater(len(original["candidate_fine_indices"]), 0)
        rejected = find_fine_queries(points, np.zeros(3), replace(cfg, validator=replace(
            cfg.validator, fine_fusion_threshold_mode="fixed", fine_fusion_score_threshold=1.)))
        self.assertEqual(len(rejected["candidate_fine_indices"]), 0)

    def test_fine_nms_method_selects_existing_greedy_or_local_maximum_path(self):
        points = np.array([[0., 0, 0], [2., 0, 0], [4., 0, 0]])
        cfg = self.cfg()
        cfg.validator = replace(cfg.validator, fine_fusion_threshold_mode="fixed", fine_fusion_score_threshold=0.)
        values = np.array([1., .8, .7])
        scores = {key: values.copy() for key in ("ced_score", "bilateral_ced_score", "pca_ratio_raw",
                                                "pca_ratio_saturated", "fusion_score")}
        scores.update(pca_valid_mask=np.ones(3, dtype=bool), fusion_valid_mask=np.ones(3, dtype=bool),
                      invalid_support_mask=np.zeros(3, dtype=bool), query_position_mm=points.copy())
        with patch("corner_detector.validator_fine.dense_ced_weighted_pca_scores", return_value=scores):
            greedy = find_fine_queries(points, np.zeros(3), cfg)
            local = find_fine_queries(points, np.zeros(3), replace(cfg, validator=replace(
                cfg.validator, fine_nms_method="local_maximum")))
        np.testing.assert_array_equal(greedy["candidate_fine_indices"], [0, 2])
        np.testing.assert_array_equal(local["candidate_fine_indices"], [0])

    def test_fine_pca_saturation_can_be_disabled(self):
        cfg = self.cfg()
        cfg.validator = replace(cfg.validator, fine_pca_ratio_saturation_tau=None)
        points = np.random.default_rng(12).uniform(-12, 12, (500, 3))
        fine = find_fine_queries(points, np.zeros(3), cfg)
        np.testing.assert_array_equal(fine["scores"]["pca_ratio_raw"], fine["scores"]["pca_ratio_saturated"])

    def test_fine_peak_relative_ratio_controls_threshold_before_nms(self):
        points = np.column_stack((np.arange(4.), np.zeros((4, 2))))
        values = np.array([.8, .4, .3, .09])
        scores = {key: values.copy() for key in ("ced_score", "bilateral_ced_score", "pca_ratio_raw",
                                                "pca_ratio_saturated", "fusion_score")}
        scores.update(pca_valid_mask=np.ones(4, dtype=bool), fusion_valid_mask=np.ones(4, dtype=bool),
                      invalid_support_mask=np.zeros(4, dtype=bool))
        cfg = self.cfg()
        cfg.validator = replace(cfg.validator, fine_fusion_threshold_mode="peak_relative",
                                fine_fusion_peak_relative_ratio=.5, fine_nms_radius_mm=.1)
        with patch("corner_detector.validator_fine.dense_ced_weighted_pca_scores", return_value=scores):
            fine = find_fine_queries(points, np.zeros(3), cfg)
            lower = find_fine_queries(points, np.zeros(3), replace(cfg, validator=replace(
                cfg.validator, fine_fusion_peak_relative_ratio=.25)))
        self.assertEqual(fine["fusion_score_threshold"], .4)
        self.assertEqual(fine["fusion_peak_relative_ratio"], .5)
        np.testing.assert_array_equal(fine["candidate_fine_indices"], [0])
        self.assertEqual(lower["fusion_score_threshold"], .2)
        np.testing.assert_array_equal(lower["candidate_fine_indices"], [0, 1, 2])
        np.testing.assert_array_equal(lower["scores"]["fusion_score"], fine["scores"]["fusion_score"])

    def test_display_disk_is_projected_onto_plane_and_stays_inside_roi(self):
        result = {"query_position_mm": np.zeros(3), "radius_mm": 5.,
                  "roi_points_mm": np.array([[0., 0, 4.]]),
                  "planes": [{"normal": np.array([0., 0, 1.]), "offset_mm": -4.,
                              "inlier_mask": np.array([True])}]}
        disk = view3d.guided_validator_primitives(result)["disk_vertices_mm"][0]
        np.testing.assert_allclose(disk[:, 2], 4.)
        np.testing.assert_allclose(np.linalg.norm(disk, axis=1), 5.)
        np.testing.assert_allclose(np.mean(disk[:-1], axis=0), [0., 0, 4.], atol=1e-12)

    def test_pipeline_selects_nearest_fine_query_and_reuses_scores_on_fine_reselection(self):
        raw = np.random.default_rng(12).uniform(-12, 12, (500, 3))
        scene = np.vstack(([[0., 0, 0], [100., 100., 100.]], raw))
        cfg = self.cfg()
        result = run_validator_pipeline(scene, 0, cfg, raw_points_mm=raw)
        fine = result["fine_result"]
        self.assertGreater(len(fine["candidate_fine_indices"]), 1)
        candidates = fine["candidate_fine_indices"]
        expected = candidates[np.lexsort((candidates, np.linalg.norm(scene[candidates], axis=1)))[0]]
        self.assertEqual(result["fine_query_index"], expected)
        self.assertEqual(result["query_index"], 0)
        np.testing.assert_array_equal(result["query_position_mm"], scene[expected])
        self.assertFalse(result["fine_fallback"])
        other = int(candidates[candidates != expected][0])
        with patch("corner_detector.validator_fine.find_fine_queries", side_effect=AssertionError("recomputed")):
            selected = run_validator_pipeline(scene, 0, cfg, raw_points_mm=raw,
                fine_result=fine, fine_index=other)
        self.assertEqual(selected["fine_query_index"], other)
        self.assertIs(selected["fine_result"], fine)
        np.testing.assert_array_equal(selected["query_position_mm"], scene[other])
        with self.assertRaises(ValueError):
            run_validator_pipeline(scene, 1, cfg, raw_points_mm=raw, fine_result=fine)

    def test_empty_fine_candidates_fall_back_to_coarse_query(self):
        raw = np.zeros((2, 3))
        scene = np.array([[1., 0, 0]])
        result = run_validator_pipeline(scene, 0, self.cfg(), raw_points_mm=raw)
        self.assertTrue(result["fine_fallback"])
        self.assertIsNone(result["fine_query_index"])
        np.testing.assert_array_equal(result["query_position_mm"], scene[0])
        self.assertEqual(result["planes"], [])

    def test_interactive_coarse_selection_stops_at_fine_and_fine_click_runs_ransac(self):
        raw = np.random.default_rng(12).uniform(-12, 12, (500, 3))
        cfg = self.cfg()
        frozen = {"points_raw_mm": raw, "points_voxel_mm": np.vstack((np.zeros((1, 3)), raw)),
                  "validator_query_indices": np.array([0])}
        with patch("corner_detector.guided_ransac.extract_guided_planes", side_effect=AssertionError("early RANSAC")), \
                redirect_stdout(StringIO()) as output:
            first = validator_query.inspect_validator_query(0, frozen, cfg)
        self.assertFalse(first["planes_computed"])
        self.assertIn("Fine heatmap/NMS ready", output.getvalue())
        self.assertGreater(len(first["fine_result"]["candidate_fine_indices"]), 0)
        chosen = int(first["fine_result"]["candidate_fine_indices"][0])
        with patch("corner_detector.validator_fine.find_fine_queries", side_effect=AssertionError("recomputed Fine")), \
                redirect_stdout(StringIO()):
            fitted = validator_query.inspect_validator_query(0, frozen, cfg, fine_index=chosen)
        self.assertTrue(fitted["planes_computed"])
        self.assertEqual(fitted["fine_query_index"], chosen)
        self.assertIs(first["fine_result"], fitted["fine_result"])

    def test_empty_fine_fallback_still_fits_when_interactive_stop_is_requested(self):
        result = run_validator_pipeline(np.zeros((1, 3)), 0, self.cfg(),
                                        raw_points_mm=np.zeros((2, 3)), stop_after_fine=True)
        self.assertTrue(result["fine_fallback"])
        self.assertTrue(result["planes_computed"])

    def test_pipeline_uses_stage_voxels_and_keeps_scene_provenance(self):
        source = np.array([[500., 500., 500.]])
        scene = np.array([[0., 0, 0], [100., 100., 100.]])
        fine_points = np.vstack((np.zeros((1, 3)), np.random.default_rng(12).uniform(-12, 12, (500, 3))))
        ransac_points = plane_cloud(3)
        cfg = replace(self.cfg(), validator=replace(self.cfg().validator,
            fine_voxel_mm=2., ransac_voxel_mm=4., inlier_threshold_mm=.01))
        with patch("corner_detector.preprocessing.voxel_downsample_3d", side_effect=[fine_points, ransac_points]):
            context = prepare_validator_voxel_context(scene, source, cfg)
        result = run_validator_pipeline(scene, 0, cfg, voxel_context=context)
        self.assertEqual(result["query_index"], 0)
        np.testing.assert_array_equal(result["query_position_mm"], fine_points[result["fine_query_index"]])
        self.assertEqual(result["fine_query_scene_index"], 0)
        np.testing.assert_array_equal(result["roi_points_mm"], ransac_points[result["roi_ransac_indices"]])
        self.assertEqual(len(result["planes"]), 3)
        for plane in result["planes"]:
            ids = plane["inlier_ransac_indices"]
            np.testing.assert_array_equal(ids, result["roi_ransac_indices"][plane["inlier_mask"]])
            self.assertGreater(ids.max(), len(scene))
            np.testing.assert_array_equal(plane["inlier_point_indices"], context["ransac"]["scene_indices"][ids])
        np.testing.assert_array_equal(result["fine_result"]["candidate_scene_indices"],
            context["fine"]["scene_indices"][result["fine_result"]["candidate_fine_indices"]])


if __name__ == "__main__":
    unittest.main()
