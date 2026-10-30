"""Direction membership, original point IDs, and the nested Ball point picker."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.special import expit
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from corner_detector.extremity import (
    ExtremityConfig, inspect_point_directions, run_extremity_refinement,
)
from corner_detector.visualization import manager, query_ui, view3d


def config(**overrides):
    return ExtremityConfig(**{**dict(radii_mm=(30.,), padding_mm=10., directions=256,
        object_clustering=False, voting_mode="soft_partition", voting_dtype="float64",
        soft_tau_mm=0.25, nms_radius_mm=5., min_vote_fraction=0.04,
        nms_method="local_maximum"), **overrides})


class DirectionDiagnosticTest(unittest.TestCase):
    def test_each_direction_matches_literal_normalized_vote_definition(self):
        points = np.random.default_rng(45).normal(0, 3, (23, 3))
        point = 3
        for dtype in ("float32", "float64"):
            for mode in ("hard", "soft_dense", "soft_partition", "soft_sparse"):
                with self.subTest(dtype=dtype, mode=mode):
                    cfg = config(voting_dtype=dtype, voting_mode=mode)
                    result = run_extremity_refinement(points, 0, cfg)
                    diagnostic = inspect_point_directions(result, point)
                    delta = (points - points[0]).astype(dtype)
                    directions = diagnostic["directions"].astype(dtype)
                    if mode == "hard":
                        expected = np.argmax(delta @ directions.T, axis=0) == point
                    else:
                        projection = directions @ delta.T
                        threshold = np.quantile(projection, 0.99, axis=1).astype(dtype)
                        weights = expit((projection - threshold[:, None]) / cfg.soft_tau_mm)
                        if mode == "soft_sparse":
                            weights[projection < threshold[:, None] - cfg.soft_alpha * cfg.soft_tau_mm] = 0
                        expected = weights[:, point] / weights.sum(axis=1)
                    np.testing.assert_allclose(diagnostic["direction_votes"], expected, rtol=2e-6, atol=2e-7)

    def test_membership_matches_literal_quantile_for_both_precisions_and_all_modes(self):
        points = np.random.default_rng(39).uniform(-10, 10, (129, 3))
        for dtype in ("float32", "float64"):
            for mode in ("hard", "soft_dense", "soft_partition", "soft_sparse"):
                with self.subTest(dtype=dtype, mode=mode):
                    result = run_extremity_refinement(points, 0, config(voting_dtype=dtype, voting_mode=mode))
                    diagnostic = inspect_point_directions(result, 7)
                    directions = diagnostic["directions"].astype(dtype)
                    delta = (result["roi_points_mm"] - points[0]).astype(dtype)
                    projection = directions @ delta.T
                    threshold = np.quantile(projection, 0.99, axis=1).astype(dtype)
                    np.testing.assert_allclose(diagnostic["threshold"], threshold, rtol=2e-6, atol=2e-6)
                    np.testing.assert_array_equal(diagnostic["top1_mask"], projection[:, 7] >= threshold)
                    self.assertEqual(diagnostic["raw_point_index"], result["roi_raw_indices"][7])

    def test_padding_competes_but_outer_point_has_no_output_vote(self):
        points = np.array([[0., 0, 0], [29., 0, 0], [35., 0, 0]])
        padded = run_extremity_refinement(points, 0, config())
        unpadded = run_extremity_refinement(points, 0, config(padding_mm=0))
        self.assertEqual(inspect_point_directions(padded, 1)["top1_count"], 0)
        self.assertGreater(inspect_point_directions(unpadded, 1)["top1_count"], 0)
        outer = inspect_point_directions(padded, 2)
        self.assertFalse(outer["inside_core"])
        self.assertTrue(outer["inside_search"])
        self.assertGreater(outer["top1_count"], 0)
        self.assertEqual(outer["raw_votes"], 0)
        np.testing.assert_array_equal(outer["direction_votes"], 0)

    def test_directional_mass_reconstructs_every_scale_and_total_score_in_all_modes(self):
        points = np.array([[0., 0, 0], [3., 1, 0], [3., 1, 0], [-4., 2, 1], [20., 0, 0], [35., 0, 0]])
        for dtype in ("float32", "float64"):
            for mode in ("hard", "soft_dense", "soft_partition", "soft_sparse"):
                with self.subTest(dtype=dtype, mode=mode):
                    result = run_extremity_refinement(points, 0,
                        config(radii_mm=(10., 30.), voting_dtype=dtype, voting_mode=mode))
                    for point in range(len(points)):
                        score = 0.
                        for scale in range(2):
                            diagnostic = inspect_point_directions(result, point, scale_index=scale)
                            np.testing.assert_allclose(diagnostic["direction_vote_sum"],
                                result["votes_per_scale"][scale, point], rtol=2e-6, atol=2e-6)
                            self.assertTrue(np.all(diagnostic["direction_votes"] >= 0))
                            self.assertTrue(np.all(diagnostic["direction_votes"] <= 1))
                            score += diagnostic["direction_scores"].sum()
                        np.testing.assert_allclose(score, result["normalized_score"][point], rtol=2e-6, atol=2e-8)

    def test_directions_conserve_one_vote_across_points_and_hard_ties_choose_first(self):
        points = np.zeros((4, 3))
        for mode in ("hard", "soft_dense", "soft_partition", "soft_sparse"):
            result = run_extremity_refinement(points, 0, config(voting_mode=mode))
            mass = np.stack([inspect_point_directions(result, i)["direction_votes"] for i in range(4)])
            np.testing.assert_allclose(mass.sum(axis=0), 1)
            if mode == "hard":
                np.testing.assert_array_equal(mass[0], 1)
                np.testing.assert_array_equal(mass[1:], 0)
            else:
                np.testing.assert_allclose(mass, 0.25)

    def test_soft_contribution_can_be_positive_without_any_top_one_percent_directions(self):
        points = np.array([[0., 0, 0], [29., 0, 0], [29.2, 0, 0]])
        result = run_extremity_refinement(points, 0, config())
        diagnostic = inspect_point_directions(result, 1)
        self.assertEqual(diagnostic["top1_count"], 0)
        self.assertGreater(diagnostic["direction_vote_sum"], 0)
        np.testing.assert_allclose(diagnostic["direction_vote_sum"], diagnostic["raw_votes"], rtol=1e-13)

    def test_membership_uses_the_selected_component_not_the_query_component(self):
        points = np.array([[-10., 0, 0], [0., 0, 0], [10., 0, 0], [11., 0, 0]])
        off = run_extremity_refinement(points, 1, config())
        on = run_extremity_refinement(points, 1, config(object_clustering=True),
                                     oracle_labels=np.array([0, 0, 0, 1]))
        self.assertEqual(inspect_point_directions(off, 2)["top1_count"], 0)
        own = inspect_point_directions(on, 2)
        np.testing.assert_array_equal(own["top1_mask"], own["directions"][:, 0] > 0)
        separate = inspect_point_directions(on, 3)
        self.assertEqual(separate["support_count"], 1)
        self.assertEqual(separate["top1_count"], 256)

    def test_ties_are_inclusive_and_small_scale_outside_search_is_explicit(self):
        tied = run_extremity_refinement(np.zeros((4, 3)), 0, config())
        self.assertEqual(inspect_point_directions(tied, 2)["top1_count"], 256)
        points = np.array([[0., 0, 0], [12., 0, 0]])
        result = run_extremity_refinement(points, 0, config(radii_mm=(5., 30.), padding_mm=2))
        small = inspect_point_directions(result, 1, scale_index=0)
        self.assertFalse(small["inside_search"])
        self.assertIsNone(small["threshold"])
        self.assertEqual(small["top1_count"], 0)
        self.assertGreater(inspect_point_directions(result, 1, scale_index=1)["top1_count"], 0)

    def test_diagnostics_do_not_mutate_or_rerun_detection_and_validate_indices(self):
        points = np.array([[0., 0, 0], [3., 0, 0], [0., 3, 0]])
        result = run_extremity_refinement(points, 0, config())
        before = {k: v.copy() for k, v in result.items() if isinstance(v, np.ndarray)}
        with patch("corner_detector.extremity._components", side_effect=AssertionError("recluster")), \
                patch("corner_detector.extremity.run_extremity_refinement", side_effect=AssertionError("rerun")):
            inspect_point_directions(result, 0)
            inspect_point_directions(result, 2)
        for key, value in before.items():
            np.testing.assert_array_equal(result[key], value)
        for point, scale in ((-1, 0), (3, 0), (0.5, 0), (0, -1), (0, 1)):
            with self.assertRaises(IndexError):
                inspect_point_directions(result, point, scale_index=scale)


class DirectionPickerTest(unittest.TestCase):
    def test_inspector_adds_no_empty_3d_line_collections(self):
        add_collection3d = Axes3D.add_collection3d

        def check_segments(ax, collection, *args, **kwargs):
            if isinstance(collection, Line3DCollection):
                self.assertGreater(len(collection._segments3d), 0)
                self.assertEqual(np.asarray(collection._segments3d).shape[-2:], (2, 3))
            return add_collection3d(ax, collection, *args, **kwargs)

        with patch.object(Axes3D, "add_collection3d", check_segments):
            _, display = self.make_display()
        self.assertEqual(len(display["rays"]._segments3d), 0)

    def make_display(self, **overrides):
        raw = np.array([[0., 0, 0], [12., 2, 1], [-3., 4, 2], [1., -5, 3]])
        # One voxel query is intentionally not interchangeable with four raw IDs.
        result = run_extremity_refinement(np.array([[0.5, 0.5, 0.]]), 0,
                                         config(**overrides), raw_points_mm=raw)
        display = manager.prepare_extremity_direction_inspector(result)
        FigureCanvasAgg(display["figure"]).draw()
        self.addCleanup(display["figure"].clear)
        return result, display

    def pixel(self, display, local_index):
        ax = display["roi_axes"]
        x, y, _ = proj3d.proj_transform(*display["roi_xyz"][local_index], ax.get_proj())
        return ax.transData.transform((x, y))

    def test_picking_after_rotation_preserves_original_ids_and_camera(self):
        result, display = self.make_display()
        for elev, azim in ((30, -60), (15, 35)):
            display["roi_axes"].view_init(elev=elev, azim=azim)
            display["figure"].canvas.draw()
            x, y = self.pixel(display, 1)
            diagnostic = manager.inspect_extremity_point(display, result, x, y, scale_index=0)
            self.assertEqual(diagnostic["raw_point_index"], result["roi_raw_indices"][1])
            np.testing.assert_array_equal(diagnostic["position_mm"], result["roi_points_mm"][1])
            self.assertEqual((display["roi_axes"].elev, display["roi_axes"].azim), (elev, azim))
            self.assertEqual(diagnostic["directions"].shape, (256, 3))
            display["figure"].canvas.draw()
        self.assertIsNone(manager.inspect_extremity_point(display, result, -100, -100, scale_index=0))

    def test_scale_switch_keeps_selection_and_marks_outside_small_search(self):
        result, display = self.make_display(radii_mm=(5., 30.), padding_mm=2)
        self.assertIsNone(manager.set_extremity_inspector_scale(display, result, 0))
        np.testing.assert_allclose(np.linalg.norm(display["core_wire"]._segments3d, axis=2), 5)
        x, y = self.pixel(display, 1)
        big = manager.inspect_extremity_point(display, result, x, y, scale_index=1)
        small = manager.set_extremity_inspector_scale(display, result, 0)
        self.assertEqual(big["raw_point_index"], small["raw_point_index"])
        self.assertFalse(small["inside_search"])
        self.assertEqual(small["top1_count"], 0)
        self.assertFalse(display["colorbar"].ax.get_visible())
        display["figure"].canvas.draw()

    def test_sphere_heatmap_uses_actual_linear_mass_and_mode_switch_does_not_recompute(self):
        result, display = self.make_display()
        x, y = self.pixel(display, 1)
        diagnostic = manager.inspect_extremity_point(display, result, x, y, scale_index=0)
        self.assertEqual(display["sphere_mode"], "contribution")
        self.assertEqual(display["direction_display_range"], (0, diagnostic["direction_votes"].max()))
        self.assertTrue(display["colorbar"].ax.get_visible())
        self.assertFalse(display["direction_background"].get_visible())
        self.assertEqual(len(display["direction_markers"]._offsets3d[0]), 256)
        before = diagnostic["direction_votes"].copy()
        with patch("corner_detector.extremity.inspect_point_directions", side_effect=AssertionError("recomputed")):
            manager.set_extremity_inspector_mode(display, result, "top1")
            self.assertTrue(display["direction_background"].get_visible())
            self.assertFalse(display["colorbar"].ax.get_visible())
            display["figure"].canvas.draw()
            manager.set_extremity_inspector_mode(display, result, "contribution")
        np.testing.assert_array_equal(display["diagnostic"]["direction_votes"], before)
        display["figure"].canvas.draw()
        with self.assertRaises(ValueError):
            manager.set_extremity_inspector_mode(display, result, "other")

    def test_tk_point_click_works_and_drag_does_not_select(self):
        result, display = self.make_display()
        fake_tk, canvas = Mock(), Mock()
        canvas._idle_draw_id = "pending-draw"
        callbacks = {}
        canvas.mpl_connect.side_effect = lambda event, callback: callbacks.setdefault(event, callback)
        with patch.object(query_ui, "tk", fake_tk), \
                patch.object(query_ui, "prepare_extremity_direction_inspector", return_value=display), \
                patch("matplotlib.backends.backend_tkagg.FigureCanvasTkAgg", return_value=canvas):
            close = query_ui.open_ball_point_inspector(object(), result)
        x, y = self.pixel(display, 1)
        press = SimpleNamespace(x=x, y=y, button=1, inaxes=display["roi_axes"])
        callbacks["button_press_event"](press)
        callbacks["button_release_event"](SimpleNamespace(x=x + 20, y=y, button=1, inaxes=display["roi_axes"]))
        self.assertIsNone(display["diagnostic"])
        callbacks["button_press_event"](press)
        callbacks["button_release_event"](press)
        self.assertEqual(display["diagnostic"]["raw_point_index"], result["roi_raw_indices"][1])
        mode_callback = fake_tk.OptionMenu.call_args.kwargs["command"]
        mode_callback("Top 1% directions")
        self.assertEqual(display["sphere_mode"], "top1")
        mode_callback("Contribution heatmap")
        self.assertEqual(display["sphere_mode"], "contribution")
        close()
        close()
        canvas.get_tk_widget.return_value.after_cancel.assert_called_once_with("pending-draw")
        fake_tk.Toplevel.return_value.destroy.assert_called_once()

    def test_changing_query_and_quitting_close_each_point_inspector(self):
        fake_tk = Mock()
        root = fake_tk.Tk.return_value
        frozen = {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
        cfg = SimpleNamespace(mode="ball", show_3d=True)
        result = object()
        first_close, second_close = Mock(), Mock()
        tick = 0

        def update():
            nonlocal tick
            tick += 1
            if tick == 2:
                frozen["pending_query"] = (3, 4)
            if tick == 3:
                root.bind_all.call_args.args[1](SimpleNamespace(keysym="q"))

        root.update.side_effect = update
        with patch.object(query_ui, "tk", fake_tk), \
                patch.object(query_ui, "TkImagePanel"), patch.object(query_ui, "set_image_window_size"), \
                patch.object(query_ui, "close_query_views") as close_views, \
                patch.object(query_ui, "poll_view_events"), patch.object(query_ui, "show_plane_edge_graph_view"), \
                patch.object(query_ui, "inspect_ball_query", return_value=result) as inspect, \
                patch.object(query_ui, "open_ball_point_inspector", side_effect=[first_close, second_close]):
            self.assertEqual(query_ui.run_frozen_click_stage(frozen, cfg, initial_pixel=(1, 2)), "quit")
        self.assertEqual(inspect.call_count, 2)
        first_close.assert_called_once()
        second_close.assert_called_once()
        self.assertTrue(close_views.call_args_list[0].kwargs["keep_extremity_overview"])


if __name__ == "__main__":
    unittest.main()
