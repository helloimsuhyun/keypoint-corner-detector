"""Live Intel RealSense RGB-D point-cloud input."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import CameraIntrinsics, PointCloudFrame, depth_preview_rgb, frame_from_rgbd


def capture_realsense_point_cloud(
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    serial_number: str | None = None,
    warmup_frames: int = 30,
    interactive: bool = True,
    timeout_ms: int = 5000,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    output_ply: str | Path | None = None,
) -> PointCloudFrame:
    """Capture one color-aligned RealSense RGB-D frame.

    In interactive mode a live RGB/depth preview is shown; press ``P`` or Space
    to capture and ``Q``/Escape to cancel. With ``interactive=False``, the first
    complete frame after ``warmup_frames`` is returned automatically.
    """

    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height, and fps must be positive")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be >= 0")
    if not 0 <= min_depth_m < max_depth_m:
        raise ValueError("Require 0 <= min_depth_m < max_depth_m")
    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - hardware environment dependent
        raise RuntimeError(
            "pyrealsense2 is required for live RealSense capture"
        ) from exc

    cv2 = None
    window_name = "RealSense input | P/Space: capture | Q/Esc: cancel"
    if interactive:
        try:
            import cv2 as cv2_module
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("opencv-python is required for interactive capture") from exc
        cv2 = cv2_module

    pipeline = rs.pipeline()
    config = rs.config()
    if serial_number:
        config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    profile = pipeline.start(config)
    align_to_color = rs.align(rs.stream.color)
    device = profile.get_device()
    depth_scale = float(device.first_depth_sensor().get_depth_scale())
    metadata = {
        "device": _device_metadata(rs, device),
        "stream": {"width": width, "height": height, "fps": fps},
        "depth_scale_m_per_unit": depth_scale,
        "depth_aligned_to": "color",
        "rgb_depth_registered": True,
    }
    complete_frames = 0

    try:
        while True:
            frames = pipeline.wait_for_frames(int(timeout_ms))
            aligned = align_to_color.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            complete_frames += 1
            depth_raw = np.asanyarray(depth_frame.get_data()).copy()
            rgb = np.asanyarray(color_frame.get_data()).copy()
            depth_m = depth_raw.astype(np.float32) * depth_scale
            intrinsics = _intrinsics_from_profile(color_frame.profile)
            ready = complete_frames > warmup_frames

            capture_now = ready and not interactive
            if interactive and cv2 is not None:
                preview = _preview_bgr(cv2, rgb, depth_m, ready=ready)
                cv2.imshow(window_name, preview)
                key = int(cv2.waitKey(1)) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    raise RuntimeError("RealSense capture was cancelled")
                if ready and key in (ord("p"), ord("P"), 13, 32):
                    capture_now = True
            if not capture_now:
                continue

            frame = frame_from_rgbd(
                depth_m,
                intrinsics,
                source="realsense-live",
                rgb=rgb,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
                metadata={
                    **metadata,
                    "intrinsics": intrinsics.as_dict(),
                    "warmup_frames": int(warmup_frames),
                },
            )
            if output_ply is not None:
                frame.save_ply(output_ply)
            return frame
    finally:
        pipeline.stop()
        if interactive and cv2 is not None:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass


def _intrinsics_from_profile(profile: Any) -> CameraIntrinsics:
    native = profile.as_video_stream_profile().get_intrinsics()
    return CameraIntrinsics(
        width=int(native.width),
        height=int(native.height),
        fx=float(native.fx),
        fy=float(native.fy),
        cx=float(native.ppx),
        cy=float(native.ppy),
    )


def _device_metadata(rs: Any, device: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for label, key in (
        ("name", rs.camera_info.name),
        ("serial_number", rs.camera_info.serial_number),
        ("firmware_version", rs.camera_info.firmware_version),
        ("product_line", rs.camera_info.product_line),
    ):
        try:
            if device.supports(key):
                result[label] = str(device.get_info(key))
        except RuntimeError:
            pass
    return result


def _preview_bgr(cv2: Any, rgb: np.ndarray, depth_m: np.ndarray, *, ready: bool) -> np.ndarray:
    color_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_bgr = cv2.cvtColor(depth_preview_rgb(depth_m), cv2.COLOR_RGB2BGR)
    preview = np.concatenate((color_bgr, depth_bgr), axis=1)
    text = "P/Space: capture | Q/Esc: cancel" if ready else "warming up..."
    cv2.putText(
        preview,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview
