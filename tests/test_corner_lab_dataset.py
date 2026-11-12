"""Standalone lab scene-number input, raw-index GT import and camera CLI compatibility."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from pointcloud_inputs.visualization import require_open3d

require_open3d()
import corner_lab_open3d_cone as lab


class CornerLabDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scene = self.root/'dataset'/'scene_0002'
        self.scene.mkdir(parents=True)
        self.xyz = np.array([[1., 2., 500.], [2., 3., 510.], [3., 4., 520.], [4., 5., 530.]], dtype=np.float32)
        self.rgb = np.arange(12, dtype=np.uint8).reshape(4, 3)
        np.savez(self.scene/'points.npz', xyz_mm=self.xyz, rgb=self.rgb, pixel_index=np.arange(4)+10)
        self.annotation = {'unit': 'mm', 'roi': {'min_mm': [0., 0., 495.], 'max_mm': [2., 3., 505.]},
                           'corners': [{'id': 5, 'raw_point_index': 3, 'xyz_mm': self.xyz[3].tolist()}]}
        (self.scene/'annotations.json').write_text(json.dumps(self.annotation))

    def test_scene_number_resolves_dataset_or_tool_root_by_id(self):
        expected = self.scene/'points.npz'
        for root in (self.root, self.root/'dataset'):
            self.assertEqual(lab.resolve_dataset_scene(root, 2), expected)
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                lab.resolve_dataset_scene(self.root, value)
        with self.assertRaisesRegex(FileNotFoundError, 'Scene 1 not found'):
            lab.resolve_dataset_scene(self.root, 1)

    def test_points_and_gt_keep_full_raw_order_even_outside_saved_roi_without_writes(self):
        before = {p.name: p.read_bytes() for p in self.scene.iterdir()}
        for path in (self.scene, self.scene/'points.npz'):
            folder, xyz, rgb = lab.load_cloud(path)
            self.assertEqual(folder, self.scene)
            np.testing.assert_array_equal(xyz, self.xyz)
            np.testing.assert_array_equal(rgb, self.rgb)
            self.assertEqual(lab.load_corners(folder, len(xyz), dataset=True), [{'id': 'gt_005', 'point_index': 3}])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.scene.iterdir()})

    def test_existing_lab_sidecar_takes_precedence_without_changing_dataset_gt(self):
        before = (self.scene/'annotations.json').read_bytes()
        records = [{'id': 'corner_001', 'point_index': 1, 'refined_xyz_mm': [2.1, 3.1, 510.1]}]
        lab.save_corners(self.scene, records)
        self.assertEqual(lab.load_corners(self.scene, 4, dataset=True), records)
        self.assertEqual(before, (self.scene/'annotations.json').read_bytes())

    def test_missing_annotations_is_allowed_and_invalid_gt_is_rejected(self):
        (self.scene/'annotations.json').unlink()
        self.assertEqual(lab.load_corners(self.scene, 4, dataset=True), [])
        self.annotation['corners'][0]['raw_point_index'] = 4
        (self.scene/'annotations.json').write_text(json.dumps(self.annotation))
        with self.assertRaisesRegex(ValueError, 'invalid/duplicate'):
            lab.load_corners(self.scene, 4, dataset=True)

    def test_existing_cloud_format_and_folder_priority_are_preserved(self):
        legacy_xyz = self.xyz + 10
        np.savez(self.scene/'cloud.npz', points_mm=legacy_xyz, colors_rgb=self.rgb)
        _, xyz, _ = lab.load_cloud(self.scene)
        np.testing.assert_array_equal(xyz, legacy_xyz)
        _, xyz, _ = lab.load_cloud(self.scene/'points.npz')
        np.testing.assert_array_equal(xyz, self.xyz)
        self.assertEqual(lab.load_corners(self.scene, 4), [])

    def test_invalid_arrays_and_unknown_archive_schema_fail_explicitly(self):
        file = self.root/'bad.npz'
        for xyz in (np.empty((0, 3)), np.full((4, 3), np.nan), np.zeros((4, 2))):
            np.savez(file, xyz_mm=xyz, rgb=np.zeros_like(xyz, dtype=np.uint8))
            with self.assertRaises(ValueError):
                lab.load_cloud(file)
        np.savez(file, unrelated=np.ones(3))
        with self.assertRaisesRegex(ValueError, 'expected saved XYZ'):
            lab.load_cloud(file)

    def test_main_resolves_scene_before_gui_startup_and_camera_flag_is_unchanged(self):
        for flags, startup, camera in ((['--scene', '2', '--dataset', str(self.root)], str(self.scene/'points.npz'), False),
                                       (['--camera'], None, True)):
            app = Mock()
            with (patch.object(lab.gui, 'Application', SimpleNamespace(instance=app)),
                  patch.object(lab, 'CornerLab') as window, redirect_stdout(StringIO())):
                lab.main(flags)
            window.assert_called_once_with(startup, False, camera, roi_only=False)
            app.initialize.assert_called_once()
            app.run.assert_called_once()

    def test_bad_scene_or_conflicting_input_never_initializes_gui(self):
        for flags in (['--scene', '1', '--dataset', str(self.root)], ['--scene', '0'],
                      ['--scene', '2', '--demo'], ['--scene', '2', '--open', 'cloud.npz']):
            app = Mock()
            with (patch.object(lab.gui, 'Application', SimpleNamespace(instance=app)),
                  redirect_stderr(StringIO()), self.assertRaises(SystemExit)):
                lab.main(flags)
            app.initialize.assert_not_called()

    def test_failed_dataset_label_load_does_not_replace_the_current_cloud(self):
        window = lab.CornerLab.__new__(lab.CornerLab)
        window.folder, window.xyz, window.rgb, window.corners = 'old', self.xyz, self.rgb, []
        window.refine_job_id = 0
        window._error = Mock()
        with patch.object(lab, 'load_corners', side_effect=ValueError('invalid GT')):
            window._open_dataset(self.scene/'points.npz')
        self.assertEqual(window.folder, 'old')
        self.assertIs(window.xyz, self.xyz)
        self.assertIs(window.rgb, self.rgb)
        window._error.assert_called_once()

    def test_constructor_initializes_refinement_and_plane_state_before_loading_scene(self):
        gui = MagicMock()
        for name in ('SceneWidget', 'ImageWidget', 'Slider', 'Checkbox', 'Combobox', 'Button', 'Horiz'):
            getattr(gui, name).side_effect = lambda *args, **kwargs: MagicMock()
        gui.Label.side_effect = lambda text: SimpleNamespace(text=text)
        gui.Application.instance.create_window.return_value.theme.font_size = 14
        gui.Application.instance.create_window.return_value.scaling = 1.
        with (patch.object(lab, 'gui', gui), patch.object(lab, 'rendering', MagicMock()),
              redirect_stdout(StringIO())):
            window = lab.CornerLab(str(self.scene/'points.npz'), False, False)
            self.assertGreaterEqual(window.refine_job_id, 1)
            self.assertFalse(window.refine_running)
            self.assertTrue(window.spherical_planes_enabled)
            self.assertEqual(window.spherical_plane_k, 3)
            self.assertEqual(window.spherical_plane_iterations, 300)
            self.assertEqual(window.spherical_plane_tolerance_deg, window.sphere_plane_tolerance_slider.int_value)
            self.assertEqual(window.spherical_plane_histogram_mode, window.sphere_plane_hist_combo.selected_index)
            self.assertEqual(window.corners, [{'id': 'gt_005', 'point_index': 3}])
            np.testing.assert_array_equal(window.xyz, self.xyz)
            token = window.refine_job_id
            window._radius_changed(31)
            window._corner_selection_changed('gt_005', 0)
            self.assertEqual(window.refine_job_id, token+2)
            window.win.show_message_box.assert_not_called()
