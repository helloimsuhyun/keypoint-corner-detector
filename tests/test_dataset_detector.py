"""Saved scene adapter: ROI/RGB alignment, natural order and no input writes."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from corner_detector import dataset_input, runner
from corner_detector.settings import configure, parse_args


class DatasetDetectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.points = np.column_stack((np.arange(6.), np.zeros(6), np.full(6, 500.)))
        self.colors = np.arange(18, dtype=np.uint8).reshape(6, 3)
        self.cfg = configure(parse_args(['--source', 'dataset', '--mode', 'detector', '--input', str(self.root),
                                         '--min-neighbors', '3', '--no-3d-view']))[0]

    def scene(self, name, *, annotation=True):
        path = self.root/name
        path.mkdir(parents=True)
        np.savez(path/'points.npz', xyz_mm=self.points, rgb=self.colors, pixel_index=np.arange(6))
        if annotation:
            (path/'annotations.json').write_text(json.dumps({'unit': 'mm',
                'roi': {'min_mm': [1., -1., 499.], 'max_mm': [3., 1., 501.]}, 'corners': []}))
        return path

    def run_dataset(self, source, *, region='roi'):
        with (patch.object(dataset_input, 'voxel_downsample_3d', side_effect=lambda raw, voxel: raw.copy()),
              patch.object(dataset_input, 'run_dense_heatmap_mode') as run, redirect_stdout(StringIO())):
            dataset_input.run_dataset_detector(source, self.cfg, region=region)
        return run

    def test_roi_passes_original_selected_xyz_and_rgb_to_shared_detector_without_writing(self):
        scene = self.scene('scene_0001')
        saved = {p.name: p.read_bytes() for p in scene.iterdir()}
        run = self.run_dataset(scene)
        run.assert_called_once()
        np.testing.assert_array_equal(run.call_args.args[0], self.points[1:4])
        np.testing.assert_array_equal(run.call_args.kwargs['raw_points_mm'], self.points[1:4])
        np.testing.assert_array_equal(run.call_args.kwargs['rgb_points_mm'], self.points[1:4])
        np.testing.assert_array_equal(run.call_args.kwargs['rgb_colors_rgb'], self.colors[1:4])
        self.assertIs(run.call_args.args[1], self.cfg)
        self.assertEqual(saved, {p.name: p.read_bytes() for p in scene.iterdir()})

    def test_full_scene_works_without_annotations(self):
        run = self.run_dataset(self.scene('scene_0001', annotation=False), region='scene')
        np.testing.assert_array_equal(run.call_args.kwargs['raw_points_mm'], self.points)
        np.testing.assert_array_equal(run.call_args.kwargs['rgb_colors_rgb'], self.colors)

    def test_dataset_and_tool_root_process_numbered_scenes_in_natural_order(self):
        self.scene('dataset/scene10')
        self.scene('dataset/scene2')
        with (patch.object(dataset_input, 'voxel_downsample_3d', side_effect=lambda raw, voxel: raw.copy()),
              patch.object(dataset_input, 'run_dense_heatmap_mode') as run, redirect_stdout(StringIO()) as out):
            dataset_input.run_dataset_detector(self.root, self.cfg)
        self.assertEqual(run.call_count, 2)
        self.assertLess(out.getvalue().index('scene2'), out.getvalue().index('scene10'))

    def test_missing_roi_empty_roi_and_missing_input_fail_before_detector(self):
        path = self.scene('scene1', annotation=False)
        with self.assertRaisesRegex(ValueError, 'annotations.json'):
            self.run_dataset(path)
        (path/'annotations.json').write_text(json.dumps({'unit': 'mm',
            'roi': {'min_mm': [100., 100., 100.], 'max_mm': [101., 101., 101.]}, 'corners': []}))
        with self.assertRaisesRegex(ValueError, 'Empty'):
            self.run_dataset(path)
        with self.assertRaises(FileNotFoundError):
            self.run_dataset(self.root/'missing')
        with self.assertRaisesRegex(ValueError, 'No scenes'):
            empty=self.root/'empty'
            empty.mkdir()
            self.run_dataset(empty)

    def test_dataset_dispatch_avoids_capture_and_organized_depth_preprocessing(self):
        with (patch.object(dataset_input, 'run_dataset_detector') as run,
              patch.object(runner, 'capture_realsense_point_cloud', side_effect=AssertionError('camera')),
              patch.object(runner, 'prepare_frozen_from_frame', side_effect=AssertionError('fake depth frame')),
              redirect_stdout(StringIO())):
            runner.main(['--source', 'dataset', '--mode', 'detector', '--input', str(self.root),
                         '--dataset-region', 'scene', '--no-3d-view'])
        self.assertEqual(run.call_args.args[0], str(self.root))
        self.assertEqual(run.call_args.kwargs['region'], 'scene')

    def test_cli_requires_dataset_input_detector_mode_and_no_synthetic_depth_noise(self):
        for flags in (['--source', 'dataset', '--mode', 'detector'],
                      ['--source', 'dataset', '--mode', 'fine', '--input', str(self.root)]):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)
        args = parse_args(['--source', 'dataset', '--mode', 'detector', '--input', str(self.root), '--noise-mm', '1'])
        with self.assertRaisesRegex(ValueError, 'organized frame'):
            configure(args)
