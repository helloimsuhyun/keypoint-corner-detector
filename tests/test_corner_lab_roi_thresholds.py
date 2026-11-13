"""Saved ROI raw-ID safety and inclusive per-field NMS score gates."""
import copy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_corner_lab_ui import lab, make_window, cached_comparison


def comparison():
    values = vars(cached_comparison()).copy()
    values.update(matches=(), center_mm=np.zeros(3), view_radius_mm=40.,
                  tau=.5, ced_cutoff=.1)
    return lab.FusionPcaExtremaComparison(**values)


class CornerLabRoiThresholdTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.xyz = np.array([[20., 0., 700.], [1., 0., 700.],
                             [30., 0., 700.], [3., 0., 700.]], dtype=np.float32)
        self.rgb = np.arange(12, dtype=np.uint8).reshape(4, 3)
        np.savez(self.root/'points.npz', xyz_mm=self.xyz, rgb=self.rgb, pixel_index=np.arange(4))
        self.annotation = {'unit': 'mm', 'roi': {'min_mm': [1., -1., 699.], 'max_mm': [3., 1., 701.]},
                           'corners': [{'id': 1, 'raw_point_index': 0, 'xyz_mm': self.xyz[0].tolist()},
                                       {'id': 2, 'raw_point_index': 3, 'xyz_mm': self.xyz[3].tolist()}]}
        (self.root/'annotations.json').write_text(json.dumps(self.annotation))

    def test_roi_is_inclusive_and_limits_display_analysis_and_picking_without_renumbering(self):
        window = make_window()
        window.roi_only = True
        window._refresh_analysis = Mock()
        saved = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with (patch.object(lab, 'voxel_downsample_analysis', side_effect=lambda xyz, rgb, size: (xyz.copy(), rgb.copy())) as down,
              redirect_stdout(StringIO())):
            self.assertTrue(window._open_dataset(self.root/'points.npz'))
        np.testing.assert_array_equal(window.input_raw_indices, [1, 3])
        np.testing.assert_array_equal(window.xyz, self.xyz)
        np.testing.assert_array_equal(down.call_args.args[0], self.xyz[[1, 3]])
        np.testing.assert_array_equal(down.call_args.args[1], self.rgb[[1, 3]])
        cloud = window.scene_widget.scene.add_geometry.call_args.args[1]
        np.testing.assert_array_equal(np.asarray(cloud.points), self.xyz[[1, 3]])
        self.assertEqual(window.corners, [{'id': 'gt_002', 'point_index': 3}])
        self.assertEqual(window.hidden_corners, [{'id': 'gt_001', 'point_index': 0}])
        self.assertEqual(saved, {p.name: p.read_bytes() for p in self.root.iterdir()})
        with redirect_stdout(StringIO()):
            window._picked_world_point(self.xyz[1])
        self.assertEqual(window.corners[-1]['point_index'], 1)
        labels = json.loads((self.root/'corners.json').read_text())['corners']
        self.assertEqual({c['point_index'] for c in labels}, {0, 1, 3})
        self.assertEqual(saved['annotations.json'], (self.root/'annotations.json').read_bytes())
        self.assertEqual(saved['points.npz'], (self.root/'points.npz').read_bytes())

    def test_missing_or_empty_roi_preserves_current_scene_and_toggle_rolls_back(self):
        window = make_window()
        window._refresh_analysis = Mock()
        window._error = Mock()
        with redirect_stdout(StringIO()):
            self.assertTrue(window._open_dataset(self.root/'points.npz'))
        original = window.xyz
        (self.root/'annotations.json').unlink()
        window.roi_only_check.checked = True
        window._roi_only_changed(True)
        self.assertFalse(window.roi_only)
        self.assertFalse(window.roi_only_check.checked)
        self.assertIs(window.xyz, original)
        self.annotation['roi'] = {'min_mm': [99., 99., 99.], 'max_mm': [100., 100., 100.]}
        (self.root/'annotations.json').write_text(json.dumps(self.annotation))
        with self.assertRaisesRegex(ValueError, 'contains no points'):
            lab.load_dataset_roi_indices(self.root, 4)

    def test_roi_cli_passes_flag_without_changing_scene_resolution(self):
        scene = self.root/'scene_0002'
        scene.mkdir()
        (scene/'points.npz').write_bytes((self.root/'points.npz').read_bytes())
        app = Mock()
        with (patch.object(lab.gui, 'Application', SimpleNamespace(instance=app)),
              patch.object(lab, 'CornerLab') as window, redirect_stdout(StringIO())):
            lab.main(['--scene', '2', '--dataset', str(self.root), '--dataset-region', 'roi'])
        window.assert_called_once_with(str(scene/'points.npz'), False, False, roi_only=True)

    def test_each_threshold_is_inclusive_in_its_own_max_or_min_direction(self):
        result = comparison()
        saved = copy.deepcopy(vars(result))
        self.assertIs(lab.threshold_extrema_comparison(result, {}), result)
        for _, key, ids_key, extreme, _ in lab.SCORE_VIEWS:
            if ids_key is None:
                continue
            ids = getattr(result, ids_key)
            boundary = float(getattr(result, key)[ids[0]])
            same = lab.threshold_extrema_comparison(result, {key: boundary})
            np.testing.assert_array_equal(getattr(same, ids_key), ids)
            cutoff = boundary + .01 if extreme == 'MAX' else max(0., boundary - .01)
            if cutoff != boundary:
                filtered = lab.threshold_extrema_comparison(result, {key: cutoff})
                self.assertEqual(len(getattr(filtered, ids_key)), 0)
        np.testing.assert_equal(vars(result), saved)
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                lab.threshold_extrema_comparison(result, {'fusion_scores': value})

    def test_thresholded_matching_reassigns_pairs_from_filtered_extrema(self):
        from dataclasses import replace
        result = comparison()
        result = replace(result, ced_maxima=np.array([1, 2]), weighted_lambda1_minima=np.array([1]),
                         ced_scores=np.array([0., .1, .8, 0., 0., 0., 0.]))
        result = lab.rematch_extrema_comparison(result, 8.)
        self.assertEqual(result.match_ced_weighted_l1, ((1, 1, 0.),))
        filtered = lab.threshold_extrema_comparison(result, {'ced_scores': .5})
        self.assertEqual(filtered.match_ced_weighted_l1, ((2, 1, 1.),))
        self.assertEqual(result.match_ced_weighted_l1, ((1, 1, 0.),))

    def test_apply_thresholds_redraws_cached_result_and_rejects_bad_input_atomically(self):
        window = make_window()
        window.compare_result = comparison()
        window.compare_show = True
        window._draw_extrema_compare_overlay = Mock()
        window._error = Mock()
        token = window.compare_job_id
        window.compare_threshold_edits['fusion_scores'].text_value = '.2'
        window.compare_threshold_edits['normal_lambda1_mm2'].text_value = '150'
        window._compare_thresholds_changed()
        self.assertEqual(window.compare_thresholds, {'fusion_scores': .2, 'normal_lambda1_mm2': 150.})
        self.assertEqual(window.compare_job_id, token)
        window._draw_extrema_compare_overlay.assert_called_once()
        previous = window.compare_thresholds
        window.compare_threshold_edits['fusion_scores'].text_value = 'nan'
        window._compare_thresholds_changed()
        self.assertIs(window.compare_thresholds, previous)
        window._error.assert_called_once()
