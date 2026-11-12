"""Read-only baseline comparison and shared-camera regressions; no display needed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from corner_dataset_tool.run_baselines import main as run_main
from corner_dataset_tool.view_baselines import load_comparisons, show_run
from corner_detector.visualization.view3d import open_baseline_comparison_view, synchronize_baseline_cameras


class BaselineVisualizationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = self.root / "dataset/scene_0001"
        self.scene.mkdir(parents=True)
        self.xyz = np.array([[0., 0., 500.], [10., 0., 500.], [100., 0., 900.]], np.float32)
        np.savez(self.scene / "points.npz", xyz_mm=self.xyz, rgb=np.full((3, 3), 80, np.uint8), pixel_index=np.arange(3))
        self.run = self.root / "saved_run"
        directory = self.run / "scene_0001/roi"
        directory.mkdir(parents=True)
        np.savez(directory / "input_selection.npz", raw_point_indices=np.array([0, 1]), gt_xyz_mm=self.xyz[[1]])
        results = []
        for method, status in (("harris3d", "ok"), ("orthogonal", "ok"), ("ahmed", "failed")):
            (directory / method).mkdir()
            if status == "ok":
                np.savez(directory / method / "predictions.npz",
                         xyz_mm=np.array([[9., 0., 502.]]) if method == "harris3d" else np.empty((0, 3)))
            results.append({"scene": str(self.scene), "method": method, "status": status, "error": "backend failed"})
        (self.run / "manifest.json").write_text(json.dumps({"unit": "mm", "mode": "roi", "results": results}))

    def test_saved_panel_order_exact_coordinates_and_zero_vs_failure(self):
        display = next(load_comparisons(self.run, methods=["ahmed", "orthogonal", "harris3d"]))
        self.assertEqual([x["method"] for x in display["results"]], ["ahmed", "orthogonal", "harris3d"])
        self.assertEqual([x["status"] for x in display["results"]], ["failed", "ok", "ok"])
        np.testing.assert_array_equal(display["points_mm"], self.xyz[:2])
        np.testing.assert_array_equal(display["gt_mm"], self.xyz[[1]])
        np.testing.assert_array_equal(display["results"][2]["xyz_mm"], [[9, 0, 502]])
        self.assertEqual(display["results"][1]["xyz_mm"].shape, (0, 3))

    def test_cached_gt_ignores_edited_live_annotations_and_reading_changes_no_files(self):
        (self.scene / "annotations.json").write_text("edited or invalid labels after the run")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        display = next(load_comparisons(self.run))
        np.testing.assert_array_equal(display["gt_mm"], self.xyz[[1]])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_missing_method_scene_and_invalid_index_are_reported(self):
        for options in ({"methods": ["unknown"]}, {"scenes": ["scene_0099"]}):
            with self.assertRaises(ValueError):
                list(load_comparisons(self.run, **options))
        selection = self.run / "scene_0001/roi/input_selection.npz"
        np.savez(selection, raw_point_indices=np.array([999]), gt_xyz_mm=np.empty((0, 3)))
        with self.assertRaisesRegex(ValueError, "indices"):
            list(load_comparisons(self.run))

    def test_saved_view_dispatches_once_with_all_methods_through_manager(self):
        with patch("corner_detector.visualization.manager.show_baseline_comparison") as show:
            show_run(self.run, methods=["harris3d", "orthogonal"])
        show.assert_called_once()
        self.assertEqual([r["method"] for r in show.call_args.kwargs["results"]], ["harris3d", "orthogonal", "ahmed"])

    def test_default_order_and_unsaved_method_are_explicit_without_recomputation(self):
        manifest_file = self.run / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["results"] = [r for r in manifest["results"] if r["method"] != "harris3d"]
        manifest_file.write_text(json.dumps(manifest))
        with patch("corner_dataset_tool.run_baselines.run_method", side_effect=AssertionError("Read only")):
            display = next(load_comparisons(self.run))
        self.assertEqual([r["method"] for r in display["results"]], ["orthogonal", "ahmed", "harris3d"])
        missing = display["results"][-1]
        self.assertEqual(missing["status"], "not_saved")
        self.assertEqual(missing["xyz_mm"].shape, (0, 3))
        self.assertIn("rerun", missing["error"])

    def test_visualize_without_execution_is_rejected_before_writing(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_main([str(self.scene), "--method", "harris3d", "--prepare-only", "--visualize"])

    def test_runner_visualizes_all_results_only_after_manifest_is_saved(self):
        calls = []

        def fake_method(args, method, directory, input_file, points, indices):
            return {"method": method, "status": "ok", "prediction_count": 0}

        def show(directory, *, methods, scenes):
            manifest = json.loads((directory / "manifest.json").read_text())
            calls.append((methods, scenes, [r["method"] for r in manifest["results"]]))

        with patch("corner_dataset_tool.run_baselines.run_method", side_effect=fake_method), \
                patch("corner_dataset_tool.view_baselines.show_run", side_effect=show), \
                patch("corner_dataset_tool.run_baselines.ROOT", self.root / "binaries"), \
                patch.object(Path, "is_file", return_value=True), \
                contextlib.redirect_stdout(io.StringIO()):
            code = run_main([str(self.scene), "--method", "harris3d", "orthogonal", "--mode", "scene",
                             "--visualize", "--output", str(self.root / "new_run")])
        self.assertEqual(code, 0)
        self.assertEqual(calls, [(["harris3d", "orthogonal", "ahmed"], ["scene_0001"],
                                 ["harris3d", "orthogonal", "ahmed"])])

    def test_default_and_partial_visualization_both_compute_all_three(self):
        for arguments in ([], ["--method", "ahmed", "--visualize"]):
            with self.subTest(arguments=arguments):
                methods = []

                def fake_method(args, method, *unused):
                    methods.append(method)
                    return {"method": method, "status": "ok", "prediction_count": 0}

                with patch("corner_dataset_tool.run_baselines.run_method", side_effect=fake_method), \
                        patch("corner_dataset_tool.view_baselines.show_run"), \
                        patch.object(Path, "is_file", return_value=True), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = run_main([str(self.scene), "--mode", "scene", "--output",
                                     str(self.root / f"all-{len(arguments)}"), *arguments])
                self.assertEqual(code, 0)
                self.assertEqual(methods, ["ahmed", "orthogonal", "harris3d"] if arguments else
                                 ["orthogonal", "ahmed", "harris3d"])

    def test_camera_changes_in_any_panel_propagate_rotation_pan_and_projection(self):
        class Camera:
            def __init__(self):
                self.view, self.projection = np.eye(4), np.eye(4)

            def get_view_matrix(self):
                return self.view

            def get_projection_matrix(self):
                return self.projection

            def copy_from(self, other):
                self.view, self.projection = other.view.copy(), other.projection.copy()

        widgets = [SimpleNamespace(scene=SimpleNamespace(camera=Camera()), force_redraw=lambda: None) for _ in range(4)]
        state, changed = synchronize_baseline_cameras(widgets)
        self.assertTrue(changed)
        state, changed = synchronize_baseline_cameras(widgets, state)
        self.assertFalse(changed)
        for index in (2, 0, 3, 1):
            widgets[index].scene.camera.view[0, 3] += index + 1
            widgets[index].scene.camera.projection[0, 0] += .5
            expected = widgets[index].scene.camera
            state, changed = synchronize_baseline_cameras(widgets, state)
            self.assertTrue(changed)
            for widget in widgets:
                np.testing.assert_array_equal(widget.scene.camera.view, expected.view)
                np.testing.assert_array_equal(widget.scene.camera.projection, expected.projection)

    def test_panels_share_cloud_keep_corners_separate_and_stop_camera_polling_on_close(self):
        fake = MagicMock()
        bounds = MagicMock()
        bounds.get_center.return_value = np.array([5., 0., 500.])
        bounds.get_extent.return_value = np.array([10., 1., 1.])

        def cloud():
            value = MagicMock()
            value.get_axis_aligned_bounding_box.return_value = bounds
            return value

        def scene(_):
            value = MagicMock()
            value.camera.get_view_matrix.return_value = np.eye(4)
            value.camera.get_projection_matrix.return_value = np.eye(4)
            return value

        gui = fake.visualization.gui
        fake.geometry.PointCloud.side_effect = cloud
        fake.utility.Vector3dVector.side_effect = lambda x: np.asarray(x).copy()
        gui.SceneWidget.side_effect = MagicMock
        gui.Label.side_effect = lambda text: SimpleNamespace(text=text)
        fake.visualization.rendering.Open3DScene.side_effect = scene
        payload = next(load_comparisons(self.run))
        payload["results"][1]["implementation"] = "python_port"
        before = payload["points_mm"].copy()
        with patch("corner_detector.visualization.view3d.o3d", fake):
            display = open_baseline_comparison_view(**payload)
        self.assertEqual([label.text for label in display["labels"]],
                         ["GT | 1 corners", "Sommer | 0 corners", "Ahmed (Python port) | FAILED", "PCL Harris3D | 1 corners"])
        inputs = [w.scene.add_geometry.call_args_list[0].args[1] for w in display["widgets"]]
        self.assertTrue(all(cloud is inputs[0] for cloud in inputs))
        self.assertEqual([w.scene.add_geometry.call_count for w in display["widgets"]], [2, 1, 1, 2])
        np.testing.assert_array_equal(payload["points_mm"], before)
        window = display["window"]
        self.assertTrue(window.set_on_close.call_args.args[0]())
        for widget in display["widgets"]:
            widget.scene.camera.get_view_matrix.side_effect = AssertionError("Camera is destroyed")
        self.assertFalse(window.set_on_tick_event.call_args.args[0]())


if __name__ == "__main__":
    unittest.main()
