"""Dataset selection and external-baseline I/O contracts (no GUI/camera)."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from corner_dataset_tool.dataset_io import SceneData
from corner_dataset_tool.baselines.setup import ROOT, adapt_ahmed, adapt_orthogonal, replace_once
from corner_dataset_tool.run_baselines import (
    load_csv, main, matlab_string, save_predictions, select_input, write_oriented_ply,
)


class DatasetBaselineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene_dir = self.root / "dataset/scene_0001"
        self.scene_dir.mkdir(parents=True)
        self.xyz = np.array([[0, 0, 500], [10, 0, 500], [20, 0, 500], [100, 0, 900]], dtype=np.float32)
        np.savez(self.scene_dir / "points.npz", xyz_mm=self.xyz,
                 rgb=np.zeros((4, 3), np.uint8), pixel_index=np.arange(4))
        self.scene = SceneData.load(self.scene_dir)
        self.scene.set_roi([0, -1, 499], [20, 1, 501])
        self.scene.add_corner(0)
        self.scene.add_corner(3)
        self.scene.save_annotations()

    def test_roi_includes_bounds_and_preserves_raw_index_space(self):
        xyz, raw = select_input(self.scene, "roi")
        np.testing.assert_array_equal(raw, [0, 1, 2])
        np.testing.assert_array_equal(xyz, self.xyz[:3])
        np.testing.assert_array_equal(select_input(self.scene, "scene")[1], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            select_input(self.scene, "typo")

    def test_roi_requires_saved_bounds_and_rejects_empty_crop(self):
        (self.scene_dir / "annotations.json").unlink()
        with self.assertRaisesRegex(ValueError, "annotations"):
            select_input(self.scene, "roi")
        self.assertEqual(len(select_input(self.scene, "scene")[0]), 4)
        self.scene.set_roi([1000, 1000, 1000], [1100, 1100, 1100])
        self.scene.save_annotations()
        with self.assertRaisesRegex(ValueError, "Empty"):
            select_input(self.scene, "roi")

    def invoke(self, *arguments):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main([str(self.scene_dir), *arguments])

    def test_prepare_modes_convert_only_units_keep_gt_and_never_edit_scene(self):
        before = {p.name: p.read_bytes() for p in self.scene_dir.iterdir()}
        for mode, expected in (("roi", [0]), ("scene", [0, 3])):
            output = self.root / f"results-{mode}"
            self.assertEqual(self.invoke("--method", "harris3d", "orthogonal", "ahmed", "--mode", mode,
                                         "--prepare-only", "--output", str(output)), 0)
            directory = output / "scene_0001" / mode
            values = np.loadtxt(directory / "input_xyz_m.txt")
            np.testing.assert_allclose(values[0], [0, 0, .5], rtol=0, atol=1e-15)
            with np.load(directory / "input_selection.npz") as data:
                np.testing.assert_array_equal(data["gt_raw_point_indices"], expected)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual([r["status"] for r in manifest["results"]], ["prepared"] * 3)
            self.assertIn("1a05583", manifest["results"][1]["source"]["commit"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.scene_dir.iterdir()})

    def test_output_refuses_existing_and_input_tree(self):
        for output in (self.root, self.scene_dir / "results"):
            with self.assertRaises(SystemExit) as exc:
                self.invoke("--method", "ahmed", "--prepare-only", "--output", str(output))
            self.assertEqual(exc.exception.code, 2)
        self.assertFalse((self.scene_dir / "results").exists())

    def test_invalid_parameters_fail_before_any_write(self):
        for option, value in (("--sampling", "1"), ("--normal-radius-mm", "nan"), ("--seed", "-1"),
                              ("--timeout", "0"), ("--dmin-mm", "2000")):
            with self.assertRaises(SystemExit):
                self.invoke("--method", "ahmed", "--prepare-only", option, value)

    def test_virtual_prediction_preserves_coordinate_and_only_reports_nearest_id(self):
        output = self.root / "predictions"
        output.mkdir()
        points, raw = select_input(self.scene, "roi")
        self.assertEqual(save_predictions(output, np.array([[.009, 0, .502]]), points, raw), 1)
        with np.load(output / "predictions.npz") as data:
            np.testing.assert_allclose(data["xyz_mm"], [[9, 0, 502]], atol=1e-12)
            np.testing.assert_array_equal(data["nearest_raw_point_index"], [1])
            np.testing.assert_allclose(data["nearest_raw_distance_mm"], [np.sqrt(5)])
            self.assertTrue(np.isnan(data["response"][0]))
        # A cropped local index must be mapped through the supplied raw IDs.
        save_predictions(output, np.array([[.1, 0, .9]]), self.xyz[[1, 3]], np.array([1, 3]))
        with np.load(output / "predictions.npz") as data:
            np.testing.assert_array_equal(data["nearest_raw_point_index"], [3])

    def test_empty_prediction_csv_is_success_but_missing_or_malformed_is_not(self):
        file = self.root / "corners.csv"
        file.write_text("x_m,y_m,z_m\n")
        data = load_csv(file, 3)
        self.assertEqual(data.shape, (0, 3))
        self.assertEqual(save_predictions(self.root, data, self.xyz, np.arange(4)), 0)
        for text in ("x,y,z\n1,2\n", "x,y,z\nnan,0,0\n"):
            file.write_text(text)
            with self.assertRaises(ValueError):
                load_csv(file, 3)
        file.write_text("")
        with self.assertRaises(StopIteration):
            load_csv(file, 3)

    def test_binary_ply_has_float32_normal_and_xyz_payload_in_metres(self):
        file = self.root / "oriented.ply"
        values = np.array([[.01, .02, .5, 0, 0, -1, 7]])
        write_oriented_ply(file, values)
        header, payload = file.read_bytes().split(b"end_header\n")
        self.assertIn(b"element vertex 1", header)
        np.testing.assert_array_equal(np.frombuffer(payload, dtype="<f4"), values[0, :6].astype(np.float32))

    def test_ahmed_adapter_changes_interface_and_plot_tail_only(self):
        repo, target = self.root / "repo", self.root / "adapted"
        repo.mkdir()
        edge_source = "function edge = edgedetection(C)\nedge = C;\nend\n"
        (repo / "edgedetection.m").write_text(edge_source)
        numerical_body = "for L=1:1:10\nidx = kmeans(curve_p,N);\nE = edge(E_ind,:);\n"
        (repo / "corner_detection.m").write_text(
            "function corner_detection(cloud)\ncount = 1;num=1;\n" + numerical_body
            + "\nfigure\nCloud = pointCloud(cloud);\nplot3(E);\nend\nend")
        adapt_ahmed(repo, target)
        self.assertEqual((target / "edgedetection.m").read_text(), edge_source)
        result = (target / "corner_detection.m").read_text()
        self.assertIn(numerical_body, result)
        self.assertIn("predictions{L} = E;", result)
        self.assertNotIn("plot3", result)
        self.assertTrue(result.startswith("function predictions = corner_detection("))
        with self.assertRaises(ValueError):
            replace_once("a a", "a", "b")
        self.assertEqual(matlab_string("it's a path"), "'it''s a path'")

    @unittest.skipUnless((ROOT / "build/harris3d/pcl_baseline").is_file(), "Build PCL adapter for native integration")
    def test_native_pcl_harris_and_normals_have_known_planar_behavior(self):
        binary = ROOT / "build/harris3d/pcl_baseline"
        x, y = np.meshgrid(np.arange(-.05, .051, .005), np.arange(-.05, .051, .005))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, .5)))
        source, result = self.root / "plane.txt", self.root / "harris.csv"
        np.savetxt(source, points)
        subprocess.run([str(binary), "harris", str(source), str(result), ".02", "1e-6", "0"], check=True)
        self.assertEqual(load_csv(result, 4).shape, (0, 4))
        subprocess.run([str(binary), "normals", str(source), str(result), ".02"], check=True)
        normals = load_csv(result, 7)
        np.testing.assert_array_equal(normals[:, 6], np.arange(len(points)))
        np.testing.assert_allclose(normals[:, 3:6], np.tile([0, 0, -1], (len(points), 1)), atol=1e-6)

    @unittest.skipUnless((ROOT / "vendor/orthogonal/ply_corners/src/main.cpp").is_file(), "Fetch author code for parity check")
    def test_sommer_adapter_keeps_entire_numerical_stage_sequence_unchanged(self):
        repo = ROOT / "vendor/orthogonal"
        target = self.root / "sommer-adapted"
        adapt_orthogonal(repo, target)
        original = (repo / "ply_corners/src/main.cpp").read_text()
        adapted = (target / "orthogonal_main.cpp").read_text()
        start, end = "    // pairing", "    pcshow_corners(points, corners);"
        expected = original[original.index(start):original.index(end)]
        self.assertIn(expected, adapted)
        self.assertNotIn("pcshow(", adapted)

    @unittest.skipUnless((ROOT / "build/orthogonal/orthogonal_baseline").is_file(), "Build Sommer adapter for native integration")
    def test_native_sommer_recovers_known_three_plane_corner(self):
        r = np.linspace(.002, .4, 45)
        a, b = np.meshgrid(r, r)
        a, b = a.ravel(), b.ravel()
        zero = np.zeros_like(a)
        corner = np.array([.1, .1, 1.])
        points = np.concatenate([np.column_stack((zero, a, b)), np.column_stack((a, zero, b)),
                                 np.column_stack((a, b, zero))]) + corner
        normals = np.repeat(-np.eye(3), len(a), axis=0)
        source, output = self.root / "trihedral.ply", self.root / "sommer.csv"
        write_oriented_ply(source, np.column_stack((points, normals)))
        with (self.root / "sommer.log").open("w") as log:
            subprocess.run([str(ROOT / "build/orthogonal/orthogonal_baseline"), "--img", str(source),
                            "--output", str(output), "--dmin", ".02", "--dmax", ".4"],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
        result = load_csv(output, 3)
        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_allclose(result[0], corner, rtol=0, atol=1e-5)

    @unittest.skipUnless((ROOT / "build/harris3d/pcl_baseline").is_file(), "Build PCL adapter for failure integration")
    def test_backend_failure_is_not_an_empty_success_and_later_scenes_continue(self):
        next_scene = self.scene_dir.parent / "scene_0002"
        next_scene.mkdir()
        (next_scene / "points.npz").write_bytes((self.scene_dir / "points.npz").read_bytes())
        output = self.root / "failed-results"
        with patch("corner_dataset_tool.run_baselines.run_command", side_effect=subprocess.TimeoutExpired("test", 1)):
            with contextlib.redirect_stdout(io.StringIO()):
                code = main([str(self.scene_dir.parent), "--method", "harris3d", "--mode", "scene", "--output", str(output)])
        self.assertEqual(code, 1)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual([r["status"] for r in manifest["results"]], ["failed", "failed"])
        self.assertFalse(list(output.rglob("predictions.npz")))


if __name__ == "__main__":
    unittest.main()
