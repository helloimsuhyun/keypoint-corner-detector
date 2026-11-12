"""Native-view isolation and camera guards; no sensor or shape simulation."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from corner_detector.visualization import manager, query_ui, view3d


class ConeViewLifecycleTest(unittest.TestCase):
    def worker(self):
        context, parent, child, process = Mock(), Mock(), Mock(), Mock()
        context.Pipe.return_value = (parent, child)
        context.Process.return_value = process
        parent.poll.return_value = False
        process.is_alive.return_value = True
        process.exitcode = None
        with patch('multiprocessing.get_context', return_value=context) as spawn:
            controller = manager._start_q10_debug_process({'kind': 'display-only'})
            spawn.assert_called_once_with('spawn')
        child.close.assert_called_once()
        process.start.assert_called_once()
        return controller, parent, process

    def test_ready_and_native_crash_are_contained_without_raising(self):
        controller, connection, process = self.worker()
        connection.poll.side_effect = [True, False]
        connection.recv.return_value = ('ready', None)
        self.assertTrue(controller['poll']())
        self.assertTrue(controller['ready'])
        connection.poll.side_effect = None
        connection.poll.return_value = False
        process.is_alive.return_value = False
        process.exitcode = -11
        self.assertFalse(controller['poll']())
        self.assertFalse(controller['poll']())
        controller['close']()
        process.join.assert_called_once()
        process.terminate.assert_not_called()
        connection.close.assert_called_once()

    def test_close_has_bounded_termination_and_is_idempotent(self):
        controller, connection, process = self.worker()
        process.is_alive.side_effect = [True, True]
        controller['close']()
        controller['close']()
        connection.send.assert_called_once_with(('close', None))
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.join.call_count, 3)
        connection.close.assert_called_once()

    def test_worker_python_error_keeps_parent_alive(self):
        controller, connection, process = self.worker()
        connection.poll.return_value = True
        connection.recv.return_value = ('error', 'graphics unavailable')
        process.is_alive.return_value = False
        self.assertFalse(controller['poll']())
        connection.close.assert_called_once()

    def test_worker_loop_owns_one_view_and_honors_close_message(self):
        connection = Mock()
        connection.poll.side_effect = [False, True]
        connection.recv.return_value = ('close', None)
        view = {'poll': Mock(return_value=True), 'close': Mock()}
        with (patch.object(view3d, 'open_q10_debug_view', return_value=view),
              patch.object(query_ui.time, 'sleep')):
            query_ui.run_q10_debug_view_process(connection, {})
        connection.send.assert_called_once_with(('ready', None))
        self.assertEqual(view['poll'].call_count, 2)
        view['close'].assert_called_once()
        connection.close.assert_called_once()

    def test_worker_setup_error_reports_and_closes_pipe(self):
        connection = Mock()
        with patch.object(view3d, 'open_q10_debug_view', side_effect=RuntimeError('no GL')):
            query_ui.run_q10_debug_view_process(connection, {})
        connection.send.assert_called_once_with(('error', 'no GL'))
        connection.close.assert_called_once()

    def test_camera_clipboard_paste_is_intercepted(self):
        vis = Mock()
        query_ui.bind_safe_camera_keys(vis)
        key, callback = vis.register_key_action_callback.call_args.args
        self.assertEqual(key, ord('V'))
        self.assertFalse(callback(vis, 1, 2))
        vis.set_view_status.assert_not_called()

    def test_roll_uses_numeric_matrices_and_skips_invalid_camera(self):
        for matrix, valid in ((np.eye(4), True), (np.full((4, 4), np.nan), False)):
            vis = Mock()
            control = vis.get_view_control.return_value
            control.convert_to_pinhole_camera_parameters.return_value.extrinsic = matrix
            query_ui.bind_lite_validator_click(vis, [], Mock())
            move = vis.register_mouse_move_callback.call_args.args[0]
            button = vis.register_mouse_button_callback.call_args.args[0]
            move(vis, 10., 20.)
            button(vis, 0, 1, 1)
            move(vis, 15., 20.)
            self.assertEqual(control.set_up.call_count, int(valid))
            vis.get_view_status.assert_not_called()
            vis.set_view_status.assert_not_called()

    def test_invalid_camera_matrices_are_never_propagated(self):
        good, bad = Mock(), Mock()
        camera = SimpleNamespace(extrinsic=np.eye(4), intrinsic=SimpleNamespace(intrinsic_matrix=np.eye(3)))
        invalid = SimpleNamespace(extrinsic=np.full((4, 4), np.nan), intrinsic=camera.intrinsic)
        good.get_view_control.return_value.convert_to_pinhole_camera_parameters.return_value = camera
        bad.get_view_control.return_value.convert_to_pinhole_camera_parameters.return_value = invalid
        previous, _ = view3d.synchronize_dense_camera_views([good, bad], [True, True], None)
        self.assertIsNone(previous[1])
        camera.extrinsic[0, 3] = 4.
        signatures, master = view3d.synchronize_dense_camera_views([good, bad], [True, True], previous)
        self.assertEqual(master, 0)
        self.assertIsNone(signatures[1])
        bad.get_view_control.return_value.convert_from_pinhole_camera_parameters.assert_not_called()


if __name__ == '__main__':
    unittest.main()
