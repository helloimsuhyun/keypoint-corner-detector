"""Open3D offline 3D ROI / multi-corner annotation GUI.

Run: python corner_annotator.py dataset/scene_0001

Ctrl+left click: label an actual raw point; optional pick mode allows plain click
Shift+left click: center the ROI on the nearest raw point (size unchanged)
Left/Right arrow: previous/next existing numbered scene (autosave current labels).
ROI + grey context mode retains the full point cloud in view.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from dataset_io import SceneData, make_o3d_cloud
from point_picker import pick_projected_point
from scene_navigation import adjacent_scene, list_scene_folders


def main():
    parser = argparse.ArgumentParser(description="3D ROI and multi-corner label editor")
    parser.add_argument("scene", type=Path, help="Folder with points.npz (saved by realsense_capture.py)")
    args = parser.parse_args()
    try:
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
    except ImportError as exc:
        raise SystemExit(f"Missing annotation dependency: {exc}. Install requirements.txt") from exc
    model = SceneData.load(args.scene)

    class Annotator:
        def __init__(self, data: SceneData):
            self.data = data
            self.scene_folders = list_scene_folders(data.root)
            self.undo_stack = []
            self.dirty = False
            self._ignore_edit_callbacks = False
            self._display_indices = np.empty((0,), dtype=np.int64)
            self._pick_indices = np.empty((0,), dtype=np.int64)
            self._label_handles = []
            self._list_corner_ids = []
            self._last_picked_index = None

            app = gui.Application.instance
            self.window = app.create_window(f"Corner Dataset | {data.root.name}", 1360, 830)
            self.window.set_on_layout(self._layout)
            self.window.set_on_key(self._on_window_key)
            self.window.set_on_close(self._on_close)
            self.widget3d = gui.SceneWidget()
            self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
            self.widget3d.scene.set_background([0.09, 0.105, 0.125, 1.0])
            self.widget3d.set_on_mouse(self._on_mouse)
            self.window.add_child(self.widget3d)
            self._make_panel()
            self._refresh_scene(reframe=True)
            self._refresh_list()
            self._status("Ready. Ctrl+click: add corner | Shift+click: move ROI center")

        def _make_panel(self):
            em = self.window.theme.font_size
            self.panel = gui.ScrollableVert(0, gui.Margins(em * 0.7, em * 0.5, em * 0.7, em * 0.6))
            self.window.add_child(self.panel)
            self.panel.add_child(gui.Label("CORNER DATASET ANNOTATOR"))
            self.scene_label = gui.Label("")
            self.panel.add_child(self.scene_label)
            nav = gui.Horiz(6)
            prev_button = gui.Button("< Previous")
            prev_button.set_on_clicked(lambda: self._navigate_scene(-1))
            nav.add_child(prev_button)
            next_button = gui.Button("Next >")
            next_button.set_on_clicked(lambda: self._navigate_scene(1))
            nav.add_child(next_button)
            self.panel.add_child(nav)
            self.panel.add_child(gui.Label("Left / Right arrow: switch scene (autosave)"))
            self._update_scene_label()
            self.count_label = gui.Label("")
            self.panel.add_child(self.count_label)

            self.panel.add_child(gui.Label("VIEW MODE"))
            self.display_mode = gui.Combobox()
            self.display_mode.add_item("Full cloud + ROI box")
            self.display_mode.add_item("ROI points only")
            self.display_mode.add_item("ROI + grey background")
            self.display_mode.selected_index = 2
            self.display_mode.set_on_selection_changed(lambda text, index: self._refresh_scene())
            self.panel.add_child(self.display_mode)
            self.panel.add_child(gui.Label("ROI CENTER (mm)"))
            self.center_edits = self._vector_controls((self.data.roi_min + self.data.roi_max) / 2)
            self.panel.add_child(gui.Label("ROI SIZE (mm)"))
            self.size_edits = self._vector_controls(np.maximum(self.data.roi_max - self.data.roi_min, 0.01))

            row = gui.Horiz(6)
            b = gui.Button("Apply ROI")
            b.set_on_clicked(self._apply_roi)
            row.add_child(b)
            b = gui.Button("Full cloud")
            b.set_on_clicked(self._reset_roi)
            row.add_child(b)
            self.panel.add_child(row)

            row2 = gui.Horiz(6)
            b = gui.Button("150mm box")
            b.set_on_clicked(self._preset_roi)
            row2.add_child(b)
            b = gui.Button("Frame ROI")
            b.set_on_clicked(self._frame_roi)
            row2.add_child(b)
            self.panel.add_child(row2)

            self.panel.add_child(gui.Label("CORNER SELECTION"))
            self.pick_mode = gui.Checkbox("Pick mode: plain left click adds GT")
            self.pick_mode.checked = False
            self.pick_mode.set_on_checked(self._pick_mode_changed)
            self.panel.add_child(self.pick_mode)
            self.panel.add_child(gui.Label("Ctrl+left click always adds GT"))
            self.panel.add_child(gui.Label("Turn Pick mode OFF to rotate by dragging"))
            pick_row = gui.Horiz(6)
            pick_row.add_child(gui.Label("Pick radius (px)"))
            self.pick_radius = gui.NumberEdit(gui.NumberEdit.INT)
            self.pick_radius.set_limits(2, 40)
            self.pick_radius.int_value = 12
            pick_row.add_child(self.pick_radius)
            self.panel.add_child(pick_row)
            self.corner_list = gui.ListView()
            self.corner_list.set_max_visible_items(8)
            self.corner_list.set_on_selection_changed(self._on_list_selected)
            self.panel.add_child(self.corner_list)

            row3 = gui.Horiz(6)
            b = gui.Button("Delete selected")
            b.set_on_clicked(self._delete_selected)
            row3.add_child(b)
            b = gui.Button("Undo")
            b.set_on_clicked(self._undo)
            row3.add_child(b)
            self.panel.add_child(row3)

            row4 = gui.Horiz(6)
            b = gui.Button("Save labels")
            b.set_on_clicked(self._save)
            row4.add_child(b)
            b = gui.Button("Export ROI.ply")
            b.set_on_clicked(self._export)
            row4.add_child(b)
            self.panel.add_child(row4)

            b = gui.Button("Reload saved labels")
            b.set_on_clicked(self._reload)
            self.panel.add_child(b)
            self.panel.add_child(gui.Label("Blue outline: ROI   Red dots: GT"))
            self.panel.add_child(gui.Label("Raw cloud is NEVER modified."))
            self.panel.add_child(gui.Label("Shift+click: recenter ROI on point"))
            self.status_label = gui.Label("")
            self.panel.add_child(self.status_label)

        def _vector_controls(self, values):
            fields = []
            horiz = gui.Horiz(4)
            for label, value in zip("XYZ", values):
                horiz.add_child(gui.Label(label))
                field = gui.NumberEdit(gui.NumberEdit.DOUBLE)
                field.decimal_precision = 2
                field.set_limits(-1000000, 1000000)
                field.set_preferred_width(82)
                field.set_value(float(value))
                horiz.add_child(field)
                fields.append(field)
            self.panel.add_child(horiz)
            return fields

        def _layout(self, _context):
            rect = self.window.content_rect
            sidebar_width = min(int(390 * self.window.scaling), max(290, int(rect.width * 0.36)))
            self.widget3d.frame = gui.Rect(rect.x, rect.y, rect.width - sidebar_width, rect.height)
            self.panel.frame = gui.Rect(rect.get_right() - sidebar_width, rect.y, sidebar_width, rect.height)

        def _status(self, msg: str):
            self.status_label.text = msg
            self.window.post_redraw()

        def _update_scene_label(self):
            current_ix = self.scene_folders.index(self.data.root)
            self.scene_label.text = (
                f"SCENE {current_ix + 1}/{len(self.scene_folders)}: {self.data.root.name}"
            )
            self.window.title = f"Corner Dataset | {self.data.root.name}"

        def _navigate_scene(self, direction: int):
            """Load a sibling capture, saving any ROI/GT edits before leaving.

            Load the target first: if its data is malformed, keep the current
            scene and unsaved edits intact. Abort transition if saving fails.
            """
            # Refresh to see any new scenes captured while the editor is open.
            try:
                siblings = list_scene_folders(self.data.root)
                target = adjacent_scene(siblings, self.data.root, direction)
                if target is None:
                    self._status("Already at the first scene." if direction < 0
                                 else "Already at the last scene.")
                    return False
                next_data = SceneData.load(target)
                if self.dirty:
                    self.data.save_annotations()
            except Exception as exc:
                self._status(f"Scene switch failed: {exc}")
                self.window.show_message_box("Scene switch failed", str(exc))
                return False

            self.data = next_data
            self.scene_folders = siblings
            self.undo_stack.clear()   # Never undo changes across different scenes.
            self.dirty = False
            self._last_picked_index = None
            self._set_roi_fields()
            self._update_scene_label()
            self._refresh_scene(reframe=True)
            self._refresh_list()
            self._status(f"Loaded {target.name}. Previous scene's changes were saved.")
            return True

        def _on_window_key(self, event):
            """Window-level shortcut works even if the sidebar has focus."""
            import open3d.visualization.gui as gui
            if event.type != gui.KeyEvent.Type.DOWN or event.is_repeat:
                return False
            if event.key == gui.KeyName.LEFT:
                self._navigate_scene(-1)
                return True
            if event.key == gui.KeyName.RIGHT:
                self._navigate_scene(1)
                return True
            return False

        def _on_close(self):
            """Save pending annotation changes on normal GUI window close."""
            if self.dirty:
                try:
                    self.data.save_annotations()
                    self.dirty = False
                except Exception as exc:
                    self._status(f"Save failed: {exc}")
                    self.window.show_message_box("Save failed", str(exc))
                    return False
            return True

        def _set_roi_fields(self):
            center = 0.5 * (self.data.roi_min + self.data.roi_max)
            extent = self.data.roi_max - self.data.roi_min
            for ed, v in zip(self.center_edits, center):
                ed.set_value(float(v))
            for ed, v in zip(self.size_edits, extent):
                ed.set_value(float(v))

        def _push_undo(self):
            snapshot = (self.data.roi_min.copy(), self.data.roi_max.copy(),
                        deepcopy(self.data.corners), self.data.next_id)
            self.undo_stack.append(snapshot)
            if len(self.undo_stack) > 50:
                self.undo_stack.pop(0)

        def _apply_roi(self):
            center = np.array([e.double_value for e in self.center_edits])
            size = np.array([e.double_value for e in self.size_edits])
            if np.any(size <= 0):
                self._status("ROI sizes must all be positive.")
                return
            low, high = center - size / 2, center + size / 2
            prospective = np.logical_and(self.data.xyz_mm >= low, self.data.xyz_mm <= high).all(axis=1)
            if not np.any(prospective):
                self._status("ROI is empty. Enlarge the box or change its center.")
                return
            self._push_undo()
            self.data.set_roi(low, high)
            self.dirty = True
            self._refresh_scene()
            self._refresh_list()
            self._status("ROI applied. Raw point cloud kept unchanged.")

        def _reset_roi(self):
            self._push_undo()
            self.data.set_roi(self.data.full_min - 0.01, self.data.full_max + 0.01)
            self._set_roi_fields()
            self.dirty = True
            self._refresh_scene()
            self._refresh_list()
            self._status("ROI reset to all original points.")

        def _preset_roi(self):
            for field in self.size_edits:
                field.set_value(150.0)
            self._apply_roi()

        def _frame_roi(self):
            import open3d as o3d
            box = o3d.geometry.AxisAlignedBoundingBox(self.data.roi_min, self.data.roi_max)
            center = box.get_center()
            self.widget3d.setup_camera(60, box, center)
            self._status("Camera framed to ROI.")

        def _material(self, pixels=3):
            import open3d.visualization.rendering as rendering
            material = rendering.MaterialRecord()
            material.shader = "defaultUnlit"
            material.point_size = float(pixels * self.window.scaling)
            return material

        def _add_cloud(self, name, indices, colors, pixels=3):
            if len(indices):
                points = self.data.xyz_mm[indices]
                self.widget3d.scene.add_geometry(name, make_o3d_cloud(points, colors), self._material(pixels))

        def _refresh_scene(self, reframe=False):
            import open3d as o3d
            import open3d.visualization.rendering as rendering
            scene = self.widget3d.scene
            for label in self._label_handles:
                self.widget3d.remove_3d_label(label)
            self._label_handles.clear()
            scene.clear_geometry()
            inside = self.data.roi_indices()
            all_idx = np.arange(len(self.data.xyz_mm), dtype=np.int64)
            mode = self.display_mode.selected_index
            if mode == 0:
                self._add_cloud("all_raw_points", all_idx, self.data.rgb, 3)
                self._display_indices = all_idx
                self._pick_indices = all_idx
            elif mode == 1:
                self._add_cloud("roi_points", inside, self.data.rgb[inside], 3)
                self._display_indices = inside
                self._pick_indices = inside
            else:
                outside = np.flatnonzero(~self.data.roi_mask())
                # Greyed geometry is only a rendering copy, not a change to original data.
                grays = np.tile(np.array([65, 70, 78], dtype=np.uint8), (len(outside), 1))
                self._add_cloud("background_ghost", outside, grays, 2)
                self._add_cloud("roi_points", inside, self.data.rgb[inside], 4)
                self._display_indices = all_idx
                self._pick_indices = inside  # grey background is not labelable in this mode

            # ROI box is an overlay: source points are unaffected.
            box = o3d.geometry.AxisAlignedBoundingBox(self.data.roi_min, self.data.roi_max)
            box.color = (0.22, 0.52, 1.0)
            edges = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
            line_mat = rendering.MaterialRecord()
            line_mat.shader = "unlitLine"
            line_mat.line_width = 1.75 * self.window.scaling
            scene.add_geometry("roi_box", edges, line_mat)
            shown_corners = [c for c in self.data.corners
                             if mode != 1 or np.all((self.data.xyz_mm[c["raw_point_index"]] >= self.data.roi_min)
                                                     & (self.data.xyz_mm[c["raw_point_index"]] <= self.data.roi_max))]
            if shown_corners:
                pts = np.array([self.data.xyz_mm[c["raw_point_index"]] for c in shown_corners])
                reds = np.tile(np.array([255, 55, 55], dtype=np.uint8), (len(pts), 1))
                scene.add_geometry("corner_markers", make_o3d_cloud(pts, reds), self._material(11))
                for c, pt in zip(shown_corners, pts):
                    label = self.widget3d.add_3d_label(pt.astype(np.float32), str(c["id"]))
                    self._label_handles.append(label)
            self.count_label.text = f"All: {len(all_idx):,} | ROI: {len(inside):,} | GT: {len(self.data.corners)}"
            if reframe:
                full_box = o3d.geometry.AxisAlignedBoundingBox(self.data.full_min - 0.01, self.data.full_max + 0.01)
                self.widget3d.setup_camera(60, full_box, full_box.get_center())
            self.widget3d.force_redraw()

        def _refresh_list(self):
            rows = []
            self._list_corner_ids = []
            for corner in self.data.corners:
                p = self.data.xyz_mm[corner["raw_point_index"]]
                in_roi = bool(np.all((p >= self.data.roi_min) & (p <= self.data.roi_max)))
                suffix = "" if in_roi else " [OUTSIDE ROI]"
                rows.append(f"#{corner['id']} | raw:{corner['raw_point_index']} | {p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}{suffix}")
                self._list_corner_ids.append(corner["id"])
            self.corner_list.set_items(rows)

        def _pick_mode_changed(self, checked):
            if checked:
                self._status("Pick mode ON: left click adds GT; disable to rotate with left drag.")
            else:
                self._status("Pick mode OFF: Ctrl+click adds GT; left drag rotates.")

        def _on_list_selected(self, value, is_double_click):
            if not value:
                return
            self._status(f"Selected {value}")

        def _delete_selected(self):
            ix = self.corner_list.selected_index
            if not (0 <= ix < len(self._list_corner_ids)):
                self._status("Select a corner from the list first.")
                return
            corner_id = self._list_corner_ids[ix]
            self._push_undo()
            self.data.remove_corner(corner_id)
            self.dirty = True
            self._refresh_scene()
            self._refresh_list()
            self._status(f"Deleted corner #{corner_id}. Undo is available.")

        def _undo(self):
            if not self.undo_stack:
                self._status("Nothing to undo.")
                return
            low, high, corners, next_id = self.undo_stack.pop()
            self.data.set_roi(low, high)
            self.data.corners = corners
            self.data.next_id = next_id
            self._set_roi_fields()
            self.dirty = True
            self._refresh_scene()
            self._refresh_list()
            self._status("Undid last annotation/ROI edit.")

        def _save(self):
            try:
                self.data.save_annotations()
                self.dirty = False
                self._status(f"Saved {len(self.data.corners)} corners and ROI to annotations.json")
            except Exception as exc:
                self.window.show_message_box("Save error", str(exc))

        def _export(self):
            try:
                count = self.data.export_roi()
                self.dirty = False
                self._status(f"Exported roi.ply and roi_indices.npy ({count:,} points); GT saved")
            except Exception as exc:
                self.window.show_message_box("Export error", str(exc))

        def _reload(self):
            try:
                self.data = SceneData.load(self.data.root)
                self.undo_stack.clear()
                self.dirty = False
                self._set_roi_fields()
                self._refresh_scene()
                self._refresh_list()
                self._status("Reloaded annotations.json from disk.")
            except Exception as exc:
                self.window.show_message_box("Reload error", str(exc))

        def _on_mouse(self, event):
            import open3d.visualization.gui as gui
            if event.type != gui.MouseEvent.Type.BUTTON_DOWN:
                return gui.Widget.EventCallbackResult.IGNORED
            if not event.is_button_down(gui.MouseButton.LEFT):
                return gui.Widget.EventCallbackResult.IGNORED

            if event.is_modifier_down(gui.KeyModifier.CTRL):
                action = "corner"
            elif event.is_modifier_down(gui.KeyModifier.SHIFT):
                action = "roi_center"
            elif self.pick_mode.checked:
                action = "corner"
            else:
                return gui.Widget.EventCallbackResult.IGNORED

            # This path requires no async render_to_depth_image() and cannot
            # mistake the blue ROI wireframe for a point in the source cloud.
            x = float(event.x - self.widget3d.frame.x)
            y = float(event.y - self.widget3d.frame.y)
            width = int(self.widget3d.frame.width)
            height = int(self.widget3d.frame.height)
            if not (0 <= x < width and 0 <= y < height):
                return gui.Widget.EventCallbackResult.CONSUMED
            if not len(self._pick_indices):
                self._status("No selectable points in ROI; change the ROI or view mode.")
                return gui.Widget.EventCallbackResult.CONSUMED

            try:
                camera = self.widget3d.scene.camera
                picked = pick_projected_point(
                    self.data.xyz_mm[self._pick_indices],
                    camera.get_view_matrix(), camera.get_projection_matrix(),
                    x, y, width, height,
                    tolerance_px=float(self.pick_radius.int_value),
                )
                if picked is None:
                    self._status("No point near cursor. Zoom in or increase Pick radius (px).")
                else:
                    raw_ix = int(self._pick_indices[picked.index])
                    self._apply_pick(raw_ix, action, picked.distance_px)
            except Exception as exc:
                self._status(f"Pick failed: {type(exc).__name__}: {exc}")
                print("Pick failed:", repr(exc), flush=True)
            return gui.Widget.EventCallbackResult.CONSUMED

        def _apply_pick(self, raw_ix, action, pixel_distance):
            xyz = self.data.xyz_mm[raw_ix]
            self._last_picked_index = raw_ix
            if action == "roi_center":
                size = np.array([e.double_value for e in self.size_edits])
                if np.any(size <= 0):
                    self._status("ROI size must be positive.")
                    return
                for field, v in zip(self.center_edits, xyz):
                    field.set_value(float(v))
                self._apply_roi()
                return
            if any(c["raw_point_index"] == raw_ix for c in self.data.corners):
                self._status(f"Raw point #{raw_ix} is already a corner.")
                return
            self._push_undo()
            corner_id = self.data.add_corner(raw_ix)
            self.dirty = True
            self._refresh_scene()
            self._refresh_list()
            self._status(
                f"Added GT #{corner_id}: raw index {raw_ix}, "
                f"screen offset {pixel_distance:.1f}px | {xyz.round(2).tolist()}mm"
            )

    gui.Application.instance.initialize()
    _window = Annotator(model)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
