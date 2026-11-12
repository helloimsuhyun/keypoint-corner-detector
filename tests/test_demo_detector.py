"""The lab's exact demo points enter the existing detector without camera input."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

from pointcloud_inputs.demo import make_demo_cloud
from corner_detector import runner
from corner_detector.settings import DEFAULTS, configure, parse_args


class DemoDetectorTest(unittest.TestCase):
    def test_demo_preserves_original_lab_geometry_colors_and_corner_id(self):
        points, colors, corner = make_demo_cloud()
        self.assertEqual(points.shape, (12601, 3))
        self.assertEqual(points.dtype, np.float32)
        self.assertEqual(colors.dtype, np.uint8)
        self.assertEqual(corner, 12600)
        np.testing.assert_array_equal(points[corner], [0., 0., 700.])
        np.testing.assert_array_equal(colors[corner], [255, 255, 255])
        self.assertEqual(hashlib.sha256(points.tobytes()).hexdigest(),
                         '4d18efda499b3f8a01db39e32692fc9b6f1b027e2a5e2a8771a9c1e699ab911c')
        self.assertEqual(hashlib.sha256(colors.tobytes()).hexdigest(),
                         '7e127f1e9841a4d577264a7306a9964e8df187e75fe56858b6d289fa3b0a2bb1')

    def test_demo_dispatch_passes_raw_rgb_and_config_to_shared_detector(self):
        raw, colors, _ = make_demo_cloud()
        voxel_points = raw[::100].astype(np.float64)
        with (patch.object(runner, 'voxel_downsample_3d', return_value=voxel_points) as voxelize,
              patch.object(runner, 'run_dense_heatmap_mode') as detect,
              patch.object(runner, 'capture_stl_point_cloud', side_effect=AssertionError('STL capture')),
              patch.object(runner, 'load_kinect_point_cloud', side_effect=AssertionError('Kinect input')),
              patch.object(runner, 'capture_realsense_point_cloud', side_effect=AssertionError('camera')),
              patch.object(runner, 'prepare_frozen_from_frame', side_effect=AssertionError('fake depth')),
              redirect_stdout(StringIO())):
            result = runner.main(['--source', 'demo', '--mode', 'detector', '--only', '--no-3d-view'])
        self.assertIs(result, detect.return_value)
        detect.assert_called_once()
        points, cfg = detect.call_args.args
        self.assertIs(points, voxel_points)
        np.testing.assert_array_equal(voxelize.call_args.args[0], raw)
        self.assertEqual(voxelize.call_args.args[1], DEFAULTS.voxel_3d_mm)
        np.testing.assert_array_equal(detect.call_args.kwargs['raw_points_mm'], raw)
        np.testing.assert_array_equal(detect.call_args.kwargs['rgb_points_mm'], raw)
        np.testing.assert_array_equal(detect.call_args.kwargs['rgb_colors_rgb'], colors)
        self.assertFalse(cfg.show_3d)
        self.assertTrue(cfg.only)
        self.assertEqual(cfg.bilateral_mode, DEFAULTS.bilateral_mode)
        self.assertEqual(cfg.centroid_radius_mm, DEFAULTS.centroid_radius_mm)
        self.assertEqual(cfg.q10_validator is not None, DEFAULTS.q10_validator_enabled)

    def test_demo_rejects_unsupported_modes_files_noise_and_insufficient_voxels(self):
        for flags in (['--source', 'demo', '--mode', 'fine'],
                      ['--source', 'demo', '--mode', 'detector', '--input', 'unused.npz']):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)
        args = parse_args(['--source', 'demo', '--mode', 'detector', '--noise-mm', '1'])
        with self.assertRaisesRegex(ValueError, 'organized frame'):
            configure(args)
        with (patch.object(runner, 'voxel_downsample_3d', return_value=np.empty((0, 3))),
              patch.object(runner, 'run_dense_heatmap_mode') as detect,
              redirect_stdout(StringIO()), self.assertRaisesRegex(SystemExit, 'Too few demo points')):
            runner.main(['--source', 'demo', '--mode', 'detector', '--no-3d-view'])
        detect.assert_not_called()
