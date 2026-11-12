"""Tuned preset routing, author reproducibility, metric settings and raw IDs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from corner_dataset_tool.baselines.config import AhmedConfig, AUTHOR_AHMED, DATASET_AHMED, SCENE2_AHMED
from corner_dataset_tool.baselines.ahmed import run_ahmed
from corner_dataset_tool.run_baselines import main, voxel_representatives


class BaselinePresetTest(unittest.TestCase):
    def test_numeric_overrides_change_only_supplied_knn_and_gate(self):
        points = np.column_stack((np.arange(400.), np.zeros(400), np.ones(400)))
        cfg = AhmedConfig(10., 40, 3., 10, 10)
        result = run_ahmed(points, 1, cfg)
        np.testing.assert_array_equal(result["edge_initial_indices"], np.r_[np.arange(17), np.arange(383, 400)])
        self.assertFalse(result["level_masks"].any())
        self.assertEqual(AUTHOR_AHMED.edge_neighbors, 400)
        self.assertEqual(DATASET_AHMED.edge_neighbors, 100)

    def test_invalid_parameter_values_are_rejected(self):
        for values in ({"voxel_mm": 0}, {"voxel_mm": float("nan")}, {"edge_threshold": -1},
                       {"edge_neighbors": 1}, {"thinning_neighbors": 3.5}, {"corner_neighbors": 0},
                       {"pca_neighbors": 2}, {"pca_neighbors": 3.5}, {"span_mm": 0},
                       {"angle_min_deg": float("nan")}, {"angle_max_deg": 181},
                       {"angle_min_deg": 150}, {"angle_min_deg": -1}):
            with self.assertRaises(ValueError):
                AhmedConfig(**values)

    def test_voxel_sampling_selects_real_points_and_resolves_centroid_ties_by_raw_order(self):
        points = np.array([[0., 0, 0], [1., 0, 0], [2., 0, 0], [20., 0, 0], [22., 0, 0]])
        np.testing.assert_array_equal(voxel_representatives(points, 10.), [1, 3])
        np.testing.assert_array_equal(voxel_representatives(points, 0.), np.arange(5))
        np.testing.assert_array_equal(points[voxel_representatives(points, 10.)], [[1, 0, 0], [20, 0, 0]])

    def test_presets_overrides_and_prepared_sampling_preserve_input_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = root / "scene_0001"
            scene.mkdir()
            points = np.array([[0., 0, 500], [1., 0, 500], [2., 0, 500], [20., 0, 500], [22., 0, 500]])
            np.savez(scene / "points.npz", xyz_mm=points, rgb=np.zeros((5, 3), np.uint8), pixel_index=np.arange(5))
            for preset in ("author", "dataset", "scene2"):
                output = root / preset
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([str(scene), "--mode", "scene", "--method", "orthogonal", "ahmed",
                                           "--preset", preset, "--prepare-only", "--output", str(output)]), 0)
                info = json.loads((output / "manifest.json").read_text())
                parameters = info["results"][0]["parameters"]
                self.assertEqual(parameters["dmax_mm"], 1000. if preset == "author" else 100.)
                self.assertEqual(parameters["ahmed_edge_neighbors"], 400 if preset == "author" else 100)
                ahmed = info["results"][1]
                self.assertEqual(ahmed["parameters"]["ahmed_level"], 1 if preset == "scene2" else 10)
                cfg = {"author": AUTHOR_AHMED, "dataset": DATASET_AHMED, "scene2": SCENE2_AHMED}[preset]
                for field in ("pca_neighbors", "span_mm", "angle_min_deg", "angle_max_deg"):
                    value = getattr(cfg, field)
                    self.assertEqual(ahmed["parameters"]["ahmed_" + field], value)
                    command = ahmed["commands"][0]
                    self.assertEqual(command[command.index("--" + field.replace("_", "-")) + 1], str(value))
            output = root / "overrides"
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(scene), "--mode", "scene", "--method", "orthogonal", "ahmed", "--prepare-only",
                      "--sommer-voxel-mm", "10", "--dmin-mm", "7", "--ahmed-thinning-neighbors", "12",
                      "--preset", "scene2", "--ahmed-level", "3", "--ahmed-pca-neighbors", "6",
                      "--ahmed-span-mm", "9", "--ahmed-angle-min-deg", "55", "--ahmed-angle-max-deg", "120",
                      "--output", str(output)])
            directory = output / "scene_0001/scene/orthogonal"
            np.testing.assert_array_equal(np.load(directory / "sampled_raw_indices.npy"), [1, 3])
            np.testing.assert_allclose(np.loadtxt(directory / "sampled_xyz_m.txt"), points[[1, 3]] / 1000)
            info = json.loads((output / "manifest.json").read_text())
            self.assertEqual(info["results"][0]["parameters"]["dmin_mm"], 7.)
            self.assertEqual(info["results"][1]["parameters"]["ahmed_thinning_neighbors"], 12)
            self.assertEqual(info["results"][1]["parameters"]["ahmed_level"], 3)
            self.assertEqual(info["results"][1]["parameters"]["ahmed_pca_neighbors"], 6)
            self.assertEqual(info["results"][1]["parameters"]["ahmed_span_mm"], 9)
            self.assertEqual(info["results"][1]["parameters"]["ahmed_angle_min_deg"], 55)
            self.assertEqual(info["results"][1]["parameters"]["ahmed_angle_max_deg"], 120)

    def test_scene2_rejects_original_matlab_backend(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main(["unused", "--method", "ahmed", "--preset", "scene2", "--ahmed-backend", "matlab"])
        self.assertEqual(exc.exception.code, 2)

    def test_native_normal_ids_are_mapped_through_sommer_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene = root / "scene_0001"
            scene.mkdir()
            points = np.array([[0., 0, 500], [1., 0, 500], [2., 0, 500], [20., 0, 500], [22., 0, 500]])
            np.savez(scene / "points.npz", xyz_mm=points, rgb=np.zeros((5, 3), np.uint8), pixel_index=np.arange(5))

            def native(command, log, timeout):
                if command[1] == "normals":
                    Path(command[3]).write_text("x_m,y_m,z_m,nx,ny,nz,input_index\n.001,0,.5,0,0,-1,0\n.02,0,.5,0,0,-1,1\n")
                else:
                    Path(command[command.index("--output") + 1]).write_text("x_m,y_m,z_m\n")

            with patch("corner_dataset_tool.run_baselines.run_command", side_effect=native), \
                    patch("corner_dataset_tool.run_baselines.subprocess.check_output", return_value="test"), \
                    patch.object(Path, "is_file", return_value=True), contextlib.redirect_stdout(io.StringIO()):
                status = main([str(scene), "--mode", "scene", "--method", "orthogonal", "--sommer-voxel-mm", "10",
                               "--output", str(root / "run")])
            self.assertEqual(status, 0)
            np.testing.assert_array_equal(np.load(root / "run/scene_0001/scene/orthogonal/normal_raw_indices.npy"), [1, 3])


if __name__ == "__main__":
    unittest.main()
