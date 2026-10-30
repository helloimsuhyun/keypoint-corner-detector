"""Interactive virtual-depth capture from an STL mesh."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

import numpy as np

from .common import CameraIntrinsics, PointCloudFrame, frame_from_rgbd


def capture_stl_point_cloud(
    stl_path: str | Path,
    *,
    stl_unit_mm: float = 1.0,
    width: int = 640,
    height: int = 480,
    fov_deg: float = 60.0,
    min_depth_m: float = 0.01,
    max_depth_m: float = 5.0,
    output_ply: str | Path | None = None,
) -> PointCloudFrame:
    """Open an STL camera window and capture the visible surface with ``P``.

    Orbit, pan, and zoom in the Open3D window, press ``P`` to capture, or press
    ``Q`` to cancel. STL coordinates are interpreted as millimeters by default;
    use ``stl_unit_mm=1000`` for an STL authored in meters.
    """

    if stl_unit_mm <= 0 or not np.isfinite(stl_unit_mm):
        raise ValueError("stl_unit_mm must be finite and > 0")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0 < fov_deg < 180:
        raise ValueError("fov_deg must be in (0, 180)")
    if not 0 <= min_depth_m < max_depth_m:
        raise ValueError("Require 0 <= min_depth_m < max_depth_m")

    source_path = Path(stl_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    o3d = _load_open3d()
    mesh = o3d.io.read_triangle_mesh(str(source_path))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"Failed to load STL mesh: {source_path}")
    if abs(stl_unit_mm - 1.0) > 1e-12:
        mesh.scale(float(stl_unit_mm), center=(0.0, 0.0, 0.0))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.72, 0.74, 0.78])

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    capture = _select_and_capture(
        o3d,
        mesh,
        scene,
        width=width,
        height=height,
        fov_deg=fov_deg,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    if capture is None:
        raise RuntimeError("STL capture was cancelled")
    frame = frame_from_rgbd(
        capture["depth_m"],
        capture["intrinsics"],
        source="stl-virtual-camera",
        rgb=capture["rgb"],
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        metadata={
            "stl_path": str(source_path),
            "stl_unit_mm": float(stl_unit_mm),
            "camera_extrinsic_world_to_camera": capture["extrinsic"].tolist(),
            "fov_deg_requested": float(fov_deg),
        },
    )
    if output_ply is not None:
        frame.save_ply(output_ply)
    return frame


def _load_open3d():
    # Open3D imports its optional ML package eagerly. Geometry-only setups can
    # otherwise fail because of an unrelated sklearn/NumPy version mismatch.
    if "open3d" not in sys.modules and "open3d.ml" not in sys.modules:
        sys.modules["open3d.ml"] = types.ModuleType("open3d.ml")
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Open3D geometry/visualization could not be imported; check the "
            "installed Open3D and NumPy versions"
        ) from exc
    return o3d


def _select_and_capture(
    o3d: Any,
    mesh: Any,
    scene: Any,
    *,
    width: int,
    height: int,
    fov_deg: float,
    min_depth_m: float,
    max_depth_m: float,
) -> dict[str, Any] | None:
    state: dict[str, Any] = {"capture": None, "cancelled": False}
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    created = visualizer.create_window(
        window_name="STL virtual camera | mouse: orbit/pan/zoom | P: capture | Q: cancel",
        width=int(width),
        height=int(height),
    )
    if not created:
        raise RuntimeError("Failed to create the Open3D STL capture window")
    visualizer.add_geometry(mesh)
    render = visualizer.get_render_option()
    if render is not None:
        render.background_color = np.array([0.025, 0.025, 0.025])
        render.mesh_show_back_face = True
    _initialize_view(visualizer, mesh, fov_deg)

    def capture_callback(active_visualizer: Any) -> bool:
        parameters = (
            active_visualizer.get_view_control().convert_to_pinhole_camera_parameters()
        )
        intrinsics = _intrinsics_from_parameters(parameters)
        rgb = _capture_rgb(active_visualizer, intrinsics.width, intrinsics.height)
        depth_m = _raycast_depth_m(
            o3d,
            scene,
            parameters,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        state["capture"] = {
            "rgb": rgb,
            "depth_m": depth_m,
            "intrinsics": intrinsics,
            "extrinsic": np.asarray(parameters.extrinsic, dtype=np.float64),
        }
        active_visualizer.close()
        return False

    def cancel_callback(active_visualizer: Any) -> bool:
        state["cancelled"] = True
        active_visualizer.close()
        return False

    visualizer.register_key_callback(ord("P"), capture_callback)
    visualizer.register_key_callback(ord("Q"), cancel_callback)
    try:
        visualizer.run()
    finally:
        visualizer.destroy_window()
    return None if state["cancelled"] else state["capture"]


def _initialize_view(visualizer: Any, mesh: Any, fov_deg: float) -> None:
    control = visualizer.get_view_control()
    control.set_lookat(mesh.get_axis_aligned_bounding_box().get_center())
    control.set_front([0.6, -0.7, -0.35])
    control.set_up([0.0, 0.0, 1.0])
    control.set_zoom(0.72)
    try:
        control.change_field_of_view(float(fov_deg) - float(control.get_field_of_view()))
    except RuntimeError:
        pass


def _intrinsics_from_parameters(parameters: Any) -> CameraIntrinsics:
    intrinsic = parameters.intrinsic
    matrix = np.asarray(intrinsic.intrinsic_matrix, dtype=np.float64)
    return CameraIntrinsics(
        width=int(intrinsic.width),
        height=int(intrinsic.height),
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
    )


def _camera_rays(parameters: Any) -> tuple[np.ndarray, CameraIntrinsics]:
    intrinsics = _intrinsics_from_parameters(parameters)
    world_to_camera = np.asarray(parameters.extrinsic, dtype=np.float64)
    camera_to_world = np.linalg.inv(world_to_camera)
    vv, uu = np.mgrid[0 : intrinsics.height, 0 : intrinsics.width]
    x = (uu.astype(np.float64) - intrinsics.cx) / intrinsics.fx
    y = (vv.astype(np.float64) - intrinsics.cy) / intrinsics.fy
    directions_camera = np.stack((x, y, np.ones_like(x)), axis=-1).reshape(-1, 3)
    directions_world = directions_camera @ camera_to_world[:3, :3].T
    origins_world = np.repeat(
        camera_to_world[None, :3, 3], len(directions_world), axis=0
    )
    rays = np.concatenate((origins_world, directions_world), axis=1).astype(np.float32)
    return rays, intrinsics


def _raycast_depth_m(
    o3d: Any,
    scene: Any,
    parameters: Any,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    rays, intrinsics = _camera_rays(parameters)
    result = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    # Mesh and ray origins use millimeters. With camera ray z=1, t_hit is the
    # camera-space Z depth rather than Euclidean range.
    depth_m = (
        result["t_hit"].numpy().reshape(intrinsics.height, intrinsics.width).astype(np.float32)
        / 1000.0
    )
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    depth_m[~valid] = np.nan
    return depth_m


def _capture_rgb(visualizer: Any, width: int, height: int) -> np.ndarray:
    screen = np.asarray(visualizer.capture_screen_float_buffer(do_render=True))
    if screen.ndim != 3 or screen.shape[2] < 3:
        return np.full((height, width, 3), 32, dtype=np.uint8)
    rgb = np.clip(screen[..., :3] * 255.0, 0, 255).astype(np.uint8)
    if rgb.shape[:2] != (height, width):
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Pillow is required to resize the STL capture") from exc
        rgb = np.asarray(Image.fromarray(rgb).resize((width, height)))
    return np.ascontiguousarray(rgb)
