"""Frozen-frame Tk query GUI for plane/edge diagnostics."""

from __future__ import annotations

import json
import time

import numpy as np
from PIL import Image, ImageDraw
from pointcloud_inputs import PointCloudFrame

from ..constants import BUILD, WINDOW_FROZEN
from ..query_debug import inspect_query
from ..ball_query import inspect_ball_query
from ..validator_query import inspect_validator_pixel, inspect_validator_query
from .manager import (
    prepare_guided_validator_view,
    start_guided_validator_view,
    prepare_guided_validator_result,
    save_guided_validator_roi_views,
    pick_guided_validator_query,
    close_query_views,
    poll_view_events,
    prepare_extremity_overview,
    prepare_extremity_direction_inspector,
    inspect_extremity_point,
    set_extremity_inspector_scale,
    set_extremity_inspector_mode,
    show_plane_edge_graph_view,
)

try:
    import tkinter as tk
    from PIL import ImageTk
except ImportError:
    tk = None
    ImageTk = None


def run_frozen_click_stage(
    frozen,
    cfg,
    *,
    initial_pixel=None,
    allow_reselection=True,
):
    if tk is None:
        raise RuntimeError("Tk is unavailable. Install Tk and Pillow in the active environment.")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("Cannot open the Tk display: " + str(exc)) from exc

    action = None
    ball_mode = getattr(cfg, "mode", None) in ("ball", "extremity256")
    guided_mode = getattr(cfg, "mode", None) in ("validator", "new-validator")
    edge_debug_window = None
    edge_debug_view = None
    edge_debug_plane_index = 0
    edge_debug_peak_index = 0
    close_point_inspector = None
    root.title(f"{WINDOW_FROZEN} | {'BALL | ' if ball_mode else ''}{BUILD}")
    query_instruction = ("Click: nearest CED query | " if ball_mode or guided_mode else "Click: query | ") if allow_reselection else ""
    projection_only = bool(getattr(getattr(cfg, "validator", None), "projection_only", False))
    workflow_keys = ("3D: 1 Seeds -> 2 Voxel -> 3 Planes -> 4 2D projections | " if projection_only else
                     "3D: 1 Coarse -> 2 Voxel -> 3 Planes | 4 ROI-Hull-TLS (S: PNG) / 5 Pair / 6 Corner + raw | " if guided_mode else
                     "3D: all queries + local views + point/directions | " if ball_mode else
                     "P/N: plane | J/K: Hough peak | ")
    instructions = query_instruction + workflow_keys + "R: recapture | Q/Esc: quit"
    status = tk.StringVar(master=root, value=instructions)
    tk.Label(root, textvariable=status, anchor="w").pack(fill="x")
    frame_view = TkImagePanel(root, draw_frozen(frozen))
    set_image_window_size(root, frozen["rgb"], extra_height=35)

    def request_action(value):
        nonlocal action
        action = value

    if guided_mode and cfg.show_3d:
        start_guided_validator_view(frozen)

    def step_edge_debug_plane(delta):
        nonlocal edge_debug_plane_index, edge_debug_peak_index
        data = frozen.get('plane_edge_debug_clustering')
        if not data or not data.get('global_planes'):
            return
        edge_debug_plane_index = (edge_debug_plane_index + delta) % len(data['global_planes'])
        edge_debug_peak_index = 0
        refresh_edge_debug_window()

    def step_edge_debug_peak(delta):
        nonlocal edge_debug_peak_index
        graph = frozen.get("plane_edge_graph_result")
        raw = frozen.get("plane_edge_debug_clustering")
        count = 0
        if graph and raw and raw.get("global_planes"):
            raw_index = int(np.clip(edge_debug_plane_index, 0, len(raw["global_planes"]) - 1))
            plane_id = int(raw["global_planes"][raw_index]["plane_id"])
            fitted = next((p for p in graph["planes"] if int(p["plane_id"]) == plane_id), None)
            if fitted is not None:
                count = len(fitted.get("line_vote", {}).get("hough_peaks_after_nms", ()))
        edge_debug_peak_index = int(np.clip(edge_debug_peak_index + delta, 0, max(0, count - 1)))
        refresh_edge_debug_window()

    def key_cb(event):
        key = event.keysym.lower()
        if key in ("q", "escape", "r"):
            request_action("recapture" if key == "r" else "quit")
        elif key == "p":
            step_edge_debug_plane(-1)
        elif key == "n":
            step_edge_debug_plane(1)
        elif key == "j":
            step_edge_debug_peak(-1)
        elif key == "k":
            step_edge_debug_peak(1)

    def mouse_cb(event):
        if not allow_reselection:
            return
        point = frame_view.image_pixel(event.x, event.y)
        if point is not None:
            frozen["pending_query"] = point

    def close_edge_debug_window():
        nonlocal edge_debug_window, edge_debug_view
        if edge_debug_window is not None:
            edge_debug_window.destroy()
        edge_debug_window = None
        edge_debug_view = None

    def refresh_edge_debug_window():
        nonlocal edge_debug_window, edge_debug_view
        graph=frozen.get('plane_edge_graph_result')
        clustering=frozen.get('plane_edge_debug_clustering')
        edge_config=frozen.get('plane_edge_debug_config')
        if graph is None or clustering is None or edge_config is None:
            return
        from .plots import make_plane_edge_debug_figure, figure_to_rgb
        fig=make_plane_edge_debug_figure(
            graph, clustering, frozen['points_voxel_mm'], edge_config,
            edge_debug_plane_index, edge_debug_peak_index)
        try:
            rgb=figure_to_rgb(fig)
        finally:
            fig.clear()
        if edge_debug_window is None:
            edge_debug_window=tk.Toplevel(root)
            edge_debug_window.title(
                '3. Per-plane 2D projection | weighted Hough NMS | '
                'continuous line refinement | final segment NMS'
            )
            edge_debug_window.protocol('WM_DELETE_WINDOW', close_edge_debug_window)
            bar=tk.Frame(edge_debug_window)
            bar.pack(side='top',fill='x',pady=2)
            for name,callback in [('◀ Plane',lambda:step_edge_debug_plane(-1)),
                                  ('Plane ▶',lambda:step_edge_debug_plane(1)),
                                  ('◀ Peak',lambda:step_edge_debug_peak(-1)),
                                  ('Peak ▶',lambda:step_edge_debug_peak(1))]:
                tk.Button(bar,text=name,command=callback).pack(side='left',padx=3)
            tk.Label(bar,text='P/N: plane, J/K: peak | gray halo supplies neighborhoods only').pack(side='left',padx=10)
            edge_debug_view=TkImagePanel(edge_debug_window,rgb)
            set_image_window_size(edge_debug_window,rgb,extra_height=45)
        else:
            edge_debug_view.set_rgb(rgb)
        edge_debug_window.deiconify()

    root.protocol("WM_DELETE_WINDOW", lambda: request_action("quit"))
    root.bind_all("<KeyPress>", key_cb)
    frame_view.canvas.bind("<Button-1>", mouse_cb)
    if initial_pixel is not None:
        frozen["pending_query"] = (int(initial_pixel[0]), int(initial_pixel[1]))
    print(f"Frozen frame ({BUILD}): {instructions}", flush=True)
    try:
        while action is None:
            root.update_idletasks()
            root.update()
            view_action = frozen.pop("validator_view_action", None)
            if view_action is not None:
                request_action(view_action)
            if action is not None:
                break
            query_index = frozen.pop("pending_validator_query", None)
            if query_index is not None:
                inspect_validator_query(query_index, frozen, cfg)
            pending = frozen.pop("pending_query", None)
            if pending is not None:
                status.set("Computing query-local voxels and RANSAC..." if guided_mode else
                           "Opening stored local 3-D extremity votes..." if ball_mode else
                           "Computing planes, padded 2-D boundaries and 3-D graph...")
                root.update_idletasks()
                close_edge_debug_window()
                if close_point_inspector is not None:
                    close_point_inspector()
                    close_point_inspector = None
                close_query_views(frozen, keep_extremity_overview=ball_mode,
                                  keep_guided_validator=guided_mode)
                edge_debug_plane_index = 0
                edge_debug_peak_index = 0
                inspector = (inspect_validator_pixel if guided_mode else
                             inspect_ball_query if ball_mode else inspect_query)
                result = inspector(pending[0], pending[1], frozen, cfg)
                if ball_mode and result is not None and cfg.show_3d:
                    close_point_inspector = open_ball_point_inspector(root, result)
                frame_view.set_rgb(draw_frozen(frozen))
                if not ball_mode and frozen.get("plane_edge_debug_config") is not None:
                    refresh_edge_debug_window()
                    root.update_idletasks()
                show_plane_edge_graph_view(frozen, show_3d=cfg.show_3d)
                status.set(instructions)
            poll_view_events(frozen)
            time.sleep(0.01)
    finally:
        close_edge_debug_window()
        if close_point_inspector is not None:
            close_point_inspector()
        close_query_views(frozen)
        root.destroy()
    return action or "quit"


def run_total_view_loop(windows):
    """Keep both total-mode windows responsive and share recapture/quit controls."""
    action, stopping = "quit", False

    def finish(value):
        def pressed(_):
            nonlocal action, stopping
            action, stopping = value, True
            return False
        return pressed

    try:
        for window in windows:
            window.register_key_callback(ord("R"), finish("recapture"))
            window.register_key_callback(ord("Q"), finish("quit"))
            window.register_key_callback(256, finish("quit"))
        active = [True] * len(windows)
        while not stopping and any(active):
            for index, window in enumerate(windows):
                if active[index]:
                    active[index] = bool(window.poll_events())
                    if active[index]:
                        window.update_renderer()
            time.sleep(.01)
    finally:
        for window in windows:
            window.destroy_window()
    return action


def run_guided_validator_view_process(connection, scene_points_mm, query_indices, scene_colors, fusion_vmax=1.0,
                                    raw_points_mm=None, query_positions_mm=None, projection_only=False):
    """Keep native GUI/Filament shutdown separate from detector and Tk lifetime."""
    frozen = {"points_voxel_mm": scene_points_mm,
              "points_raw_mm": raw_points_mm,
              "validator_query_indices": query_indices, "validator_scene_colors": scene_colors,
              "validator_fusion_vmax": fusion_vmax}
    if projection_only:
        frozen["validator_projection_only"] = True
    if query_positions_mm is not None:
        frozen["validator_query_positions_mm"] = query_positions_mm
    try:
        open_guided_validator_window(frozen, lambda action: connection.send(("action", action)))
        while frozen.get("guided_validator_display") is not None:
            while connection.poll():
                kind, result = connection.recv()
                if kind == "close":
                    return
                if kind == "result":
                    prepare_guided_validator_result(frozen, result)
            poll_view_events(frozen)
            index = frozen.pop("pending_validator_query", None)
            if index is not None:
                connection.send(("query", index))
            time.sleep(0.01)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()
        close_query_views(frozen)


def open_guided_validator_window(frozen, request_action):
    """Route native Open3D events into the existing Tk query loop."""
    display = prepare_guided_validator_view(frozen)
    gui = display["gui"]
    pressed = None

    def mouse(event):
        nonlocal pressed
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            pressed = ((event.x, event.y) if event.is_button_down(gui.MouseButton.LEFT) else None)
        elif event.type == gui.MouseEvent.Type.DRAG:
            if pressed is not None and np.linalg.norm(np.asarray(pressed) - [event.x, event.y]) > 4:
                pressed = None
        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            start, pressed = pressed, None
            if start is not None and np.linalg.norm(np.asarray(start) - [event.x, event.y]) <= 4:
                index = pick_guided_validator_query(display, event.x, event.y)
                if index is not None:
                    frozen["pending_validator_query"] = index
        return gui.Widget.EventCallbackResult.IGNORED

    def key(event):
        if event.type == gui.KeyEvent.DOWN:
            if event.key == gui.KeyName.S:
                save_guided_validator_roi_views(frozen)
                return gui.Widget.EventCallbackResult.HANDLED
            if event.key in (gui.KeyName.Q, gui.KeyName.ESCAPE, gui.KeyName.R):
                request_action("recapture" if event.key == gui.KeyName.R else "quit")
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    display["overview"].set_on_mouse(mouse)
    widgets = [display["overview"], display["roi_view"], display["local"]]
    widgets.extend(panel["widget"] for stage in display["geometry_windows"] for panel in stage["panels"])
    for widget in widgets:
        widget.set_on_key(key)
    for stage in display["geometry_windows"]:
        comparison = [panel for panel in stage["panels"]
                      if panel.get("roi_debug_view") in ("D-before", "D-after")]
        for panel in comparison:
            def sync_camera(event, panel=panel, comparison=comparison, stage=stage):
                def copy_camera():
                    if not stage["closed"]:
                        for other in comparison:
                            if other is not panel:
                                other["widget"].scene.camera.copy_from(panel["widget"].scene.camera)
                                other["widget"].force_redraw()
                display["app"].post_to_main_thread(stage["window"], copy_camera)
                return gui.Widget.EventCallbackResult.IGNORED
            panel["widget"].set_on_mouse(sync_camera)


def open_ball_point_inspector(root, result):
    """Embed the ROI picker and direction sphere in the existing Tk event loop."""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    display = prepare_extremity_direction_inspector(result)
    window = tk.Toplevel(root)
    window.title(f"BALL point directions | CED query {result['query_index']}")
    tk.Label(window, text="Left: click a point / drag to rotate / right-drag to zoom | "
             "Sphere: directional vote heatmap / Top 1% directions").pack(fill="x")
    info = tk.StringVar(master=window, value="Select any raw ROI point, including rejected/low-vote points.")
    tk.Label(window, textvariable=info, anchor="w").pack(fill="x")
    controls = tk.Frame(window)
    controls.pack(fill="x")
    canvas = FigureCanvasTkAgg(display["figure"], master=window)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    scale_index = int(np.argmax(result["config"].radii_mm))
    pressed = None
    closed = False

    def show_details(diagnostic):
        if diagnostic is not None:
            info.set(f"raw={diagnostic['raw_point_index']} xyz-mm={diagnostic['position_mm'].round(3).tolist()} "
                     f"component={diagnostic['cluster_id']} | top1%={diagnostic['top1_count']}/"
                     f"{len(diagnostic['directions'])} | support={diagnostic['support_count']}\n"
                     f"scale sum={diagnostic['direction_vote_sum']:.4g} "
                     f"stored scale V={diagnostic['scale_raw_votes']:.4g} | total V={diagnostic['raw_votes']:.4g} "
                     f"S={diagnostic['normalized_score']:.4g}")
        canvas.draw_idle()

    def press(event):
        nonlocal pressed
        pressed = ((event.x, event.y) if event.button == 1 and
                   event.inaxes is display["roi_axes"] else None)

    def release(event):
        nonlocal pressed
        start, pressed = pressed, None
        if (closed or start is None or event.button != 1 or event.inaxes is not display["roi_axes"]
                or np.linalg.norm(np.asarray(start) - [event.x, event.y]) > 4):
            return
        diagnostic = inspect_extremity_point(display, result, event.x, event.y, scale_index=scale_index)
        show_details(diagnostic)

    modes = {"Contribution heatmap": "contribution", "Top 1% directions": "top1"}
    mode_choice = tk.StringVar(master=window, value="Contribution heatmap")

    def change_mode(label):
        show_details(set_extremity_inspector_mode(display, result, modes[label]))

    tk.OptionMenu(controls, mode_choice, *modes, command=change_mode).pack(side="left")

    if len(result["config"].radii_mm) > 1:
        options = {f"Scale {i + 1}: r={radius:g} mm": i
                   for i, radius in enumerate(result["config"].radii_mm)}
        choice = tk.StringVar(master=window, value=list(options)[scale_index])

        def change_scale(label):
            nonlocal scale_index
            scale_index = options[label]
            show_details(set_extremity_inspector_scale(display, result, scale_index))

        tk.OptionMenu(controls, choice, *options, command=change_scale).pack(side="left")

    def close():
        nonlocal closed
        if not closed:
            closed = True
            if canvas._idle_draw_id is not None:
                canvas.get_tk_widget().after_cancel(canvas._idle_draw_id)
                canvas._idle_draw_id = None
            display["figure"].clear()
            window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    canvas.mpl_connect("button_press_event", press)
    canvas.mpl_connect("button_release_event", release)
    canvas.draw_idle()
    return close


def bind_lite_validator_click(vis, records, on_pick, *, position_key="position"):
    """Left click selects an NMS marker; dragging retains the existing camera controls."""
    from .view3d import pick_lite_validator_candidate
    bind_safe_camera_keys(vis)

    cursor, pressed = np.zeros(2), None
    buttons, modifiers, dragged = set(), 0, False

    def move(callback_vis, x, y):
        nonlocal dragged
        delta = np.array([x, y]) - cursor
        control = callback_vis.get_view_control()
        # Open3D replaces its default handlers when custom mouse callbacks are registered.
        if 2 in buttons or (0 in buttons and modifiers & 2):
            control.translate(*delta, *cursor)
        elif 0 in buttons:
            if modifiers & 1:
                try:
                    matrix = np.asarray(control.convert_to_pinhole_camera_parameters().extrinsic, dtype=float)
                    if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                        up, front = -matrix[1, :3], -matrix[2, :3]
                        if min(np.linalg.norm(up), np.linalg.norm(front)) > 1e-12:
                            up, front = up / np.linalg.norm(up), front / np.linalg.norm(front)
                            angle = .003 * delta[0]
                            control.set_up(up * np.cos(angle) + np.cross(front, up) * np.sin(angle)
                                           + front * np.dot(front, up) * (1 - np.cos(angle)))
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    pass
            elif modifiers & 4:
                control.camera_local_rotate(*delta, *cursor)
            else:
                control.rotate(*delta, *cursor)
        cursor[:] = x, y
        if pressed is not None and np.linalg.norm(cursor - pressed) > 4:
            dragged = True
        return False

    def button(callback_vis, button, action, mods):
        nonlocal pressed, modifiers, dragged
        if action == 1:
            buttons.add(button)
            modifiers = mods
            if button == 0:
                pressed, dragged = cursor.copy(), False
                callback_vis.get_view_control().reset_camera_local_rotate()
        elif action == 0:
            buttons.discard(button)
            if button != 0:
                return False
            start, pressed = pressed, None
            if start is not None and not dragged and modifiers == 0 and mods == 0:
                options = {} if position_key == "position" else {"position_key": position_key}
                record = pick_lite_validator_candidate(callback_vis, records, *cursor, **options)
                if record is not None:
                    print(f"[LITE DEBUG] Selected NMS candidate {record['candidate_id']}", flush=True)
                    on_pick(record)
                else:
                    print("[LITE DEBUG] No NMS marker near click; click a candidate marker.", flush=True)
        return False

    vis.register_mouse_move_callback(move)
    vis.register_mouse_button_callback(button)


def bind_safe_camera_keys(vis):
    # Native V/Ctrl-V interprets arbitrary clipboard text as camera JSON.
    vis.register_key_action_callback(ord("V"), lambda _vis, _action, _mods: False)


def run_baseline_comparison_process(connection, points_mm, rgb, gt_mm, results, title):
    """Keep the comparison GUI lifecycle separate from detector execution."""
    from .view3d import open_baseline_comparison_view
    try:
        display = open_baseline_comparison_view(points_mm, rgb, gt_mm, results, title=title)
        display["app"].run()
        connection.send(None)
    except Exception as exc:
        connection.send(str(exc))
    finally:
        connection.close()


def run_q10_debug_view_process(connection, record):
    """Own exactly one Open3D window, with no detector/Tk/GLFW context inherited."""
    from .view3d import open_q10_debug_view
    view = None
    try:
        view = open_q10_debug_view(record)
        connection.send(("ready", None))
        while view["poll"]():
            if connection.poll():
                kind, value = connection.recv()
                if kind == "close":
                    break
                if kind == "step" and "set_step" in view:
                    view["set_step"](value)
            time.sleep(.01)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send(("error", str(exc)))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if view is not None:
            view["close"]()
        connection.close()


def open_lite_validator_debug_window(display, *, spatial_view=None):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    if tk is None:
        raise RuntimeError("Lite debug requires Tk in the active environment")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("Cannot open the Lite debug display: " + str(exc)) from exc
    record = display["record"]
    root.title(f"Q10 + spherical MaxGap | refine {'ON' if record.get('refine_enabled', True) else 'OFF'} "
               f"| NMS candidate {record['candidate_id']}")
    root.geometry(f"{min(1760, root.winfo_screenwidth()-80)}x{min(1020, root.winfo_screenheight()-100)}")
    info = tk.StringVar(master=root)
    tk.Label(root, textvariable=info, anchor="w").pack(fill="x")
    canvas = FigureCanvasTkAgg(display["figure"], master=root)
    closed = False

    info.set("Markers: RED original | ORANGE current | WHITE path | GREEN pass / RED reject / AMBER unknown\n"
             "Voxels: P1 ORANGE / P2 GREEN / P3 PURPLE | WHITE overlapping planes | DIM discarded by all member planes\n"
             "BLUE unscored | GRAY no fitted plane | Plane colors match arcs, projected dots and rose plots. CYAN arrow = Q10 direction.")
    canvas.get_tk_widget().pack(fill="both", expand=True)
    def select_step(value):
        step = int(float(value))
        display["set_step"](step)
        canvas.draw_idle()
        if spatial_view is not None and "set_step" in spatial_view:
            spatial_view["set_step"](step)
    slider = tk.Scale(root, from_=0, to=len(record["trace"]["positions_mm"]) - 1,
                      orient="horizontal", label="Q10 refinement step (final RANSAC / plane support stays fixed)",
                      command=select_step, resolution=1)
    slider.set(len(record["trace"]["positions_mm"]) - 1)
    slider.pack(fill="x")

    def close():
        nonlocal closed, spatial_view
        if not closed:
            closed = True
            if spatial_view is not None:
                spatial_view["close"]()
                spatial_view = None
            if canvas._idle_draw_id is not None:
                canvas.get_tk_widget().after_cancel(canvas._idle_draw_id)
                canvas._idle_draw_id = None
            display["figure"].clear()
            root.destroy()

    def poll():
        nonlocal spatial_view
        if not closed:
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                close()
            if spatial_view is not None and not spatial_view["poll"]():
                spatial_view["close"]()
                spatial_view = None
        return not closed

    root.protocol("WM_DELETE_WINDOW", close)
    root.update_idletasks()
    root.lift()
    return {"poll": poll, "close": close}


class TkImagePanel:
    def __init__(self, parent, rgb):
        self.canvas = tk.Canvas(parent, highlightthickness=0, background="#20242a")
        self.canvas.pack(fill="both", expand=True)
        self.item = self.canvas.create_image(0, 0, anchor="nw")
        self.photo = None
        self.layout = None
        self.source = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        self.canvas.bind("<Configure>", self.redraw)

    def set_rgb(self, rgb):
        self.source = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        self.redraw()

    def redraw(self, event=None):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if min(cw, ch) <= 1:
            return
        iw, ih = self.source.size
        scale = min(cw / iw, ch / ih)
        dw, dh = max(1, round(iw * scale)), max(1, round(ih * scale))
        left, top = (cw - dw) // 2, (ch - dh) // 2
        resized = self.source.resize((dw, dh), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(resized, master=self.canvas)
        self.canvas.coords(self.item, left, top)
        self.canvas.itemconfigure(self.item, image=self.photo)
        self.layout = (left, top, dw, dh)

    def image_pixel(self, x, y):
        if self.layout is None:
            return None
        left, top, dw, dh = self.layout
        if not (left <= x < left + dw and top <= y < top + dh):
            return None
        iw, ih = self.source.size
        return (
            min(iw - 1, int((x - left) * iw / dw)),
            min(ih - 1, int((y - top) * ih / dh)),
        )


def set_image_window_size(window, rgb, extra_height=0):
    height, width = rgb.shape[:2]
    scale = min(
        1.0,
        0.9 * window.winfo_screenwidth() / width,
        max(1, 0.8 * window.winfo_screenheight() - extra_height) / height,
    )
    window.geometry(
        f"{max(1, round(width * scale))}x{max(1, round(height * scale)) + extra_height}"
    )


def draw_frozen(frozen: dict) -> np.ndarray:
    image = Image.fromarray(frozen["rgb"])
    point = frozen.get("selected_pixel")
    if point is not None:
        x, y = point
        draw = ImageDraw.Draw(image)
        draw.line((x - 12, y, x + 12, y), fill=(255, 0, 0), width=3)
        draw.line((x, y - 12, x, y + 12), fill=(255, 0, 0), width=3)
    return np.asarray(image).copy()


def run_validator_mode(
    frame: PointCloudFrame,
    frozen: dict,
    cfg,
) -> str:
    # Keep ``frame`` in the public signature used by the runner. The frozen
    # mapping contains the aligned RGB/depth data consumed by the Tk stage.
    _ = frame
    frozen.pop("selected_pixel", None)
    frozen.pop("pending_query", None)
    if getattr(cfg, "mode", None) in ("ball", "extremity256"):
        prepare_extremity_overview(frozen, show_3d=cfg.show_3d)
        print("[BALL] click a valid RGB/depth pixel to inspect the nearest existing CED NMS query", flush=True)
    elif getattr(cfg, "mode", None) == "validator":
        print("[VALIDATOR] Select one orange Coarse NMS query. Window 6 shows edges -> overlap removal -> extension / corner LS for that query.", flush=True)
    else:
        print("[QUERY] click a valid RGB/depth pixel in the frozen-frame window", flush=True)
    return run_frozen_click_stage(
        frozen,
        cfg,
        allow_reselection=True,
    )


# Library compatibility only. The CLI mode itself is named ``fine``.
run_query_mode = run_validator_mode
