"""RealSense RGB-D capture and immutable point-cloud scene creation."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from dataset_io import create_scene


def next_scene_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    for i in range(1, 1000000):
        path = base / f"scene_{i:04d}"
        if not path.exists():
            return path
    raise RuntimeError("Too many scenes")


def intrinsics_to_dict(intrinsics):
    return {
        "width": int(intrinsics.width), "height": int(intrinsics.height),
        "fx": float(intrinsics.fx), "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx), "ppy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "coeffs": [float(v) for v in intrinsics.coeffs],
    }


def main():
    parser = argparse.ArgumentParser(description="Capture RGB-D point-cloud scenes using a RealSense camera")
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", default="", help="Optional RealSense serial number")
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
        import cv2
    except ImportError as exc:
        raise SystemExit(f"Missing capture dependency: {exc}. Install requirements.txt") from exc

    pipeline = rs.pipeline()
    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    print("Starting camera... Space = save capture, Q/Esc = exit")
    try:
        profile = pipeline.start(cfg)
    except RuntimeError as exc:
        raise SystemExit(f"Cannot start RealSense: {exc}") from exc
    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()
    sensor = profile.get_device().first_depth_sensor()
    depth_scale_m = sensor.get_depth_scale()
    device = profile.get_device()
    serial = device.get_info(rs.camera_info.serial_number)

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth = np.asanyarray(depth_frame.get_data()).copy()  # uint16, aligned to RGB
            bgr = np.asanyarray(color_frame.get_data()).copy()
            # Depth preview only; the underlying 16-bit array is unchanged.
            dep_8 = cv2.convertScaleAbs(depth, alpha=0.03)
            dep_vis = cv2.applyColorMap(dep_8, cv2.COLORMAP_TURBO)
            preview = np.hstack((bgr, dep_vis))
            cv2.putText(preview, "SPACE: capture | Q: quit", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("RealSense Corner Dataset Capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key != 32:
                continue

            # Use SDK-calculated vertices: handles actual RealSense projection model.
            pc.map_to(color_frame)
            cloud = pc.calculate(depth_frame)
            xyz_metres = np.asanyarray(cloud.get_vertices()).view(np.float32).reshape(-1, 3)
            color_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pixel_ids = np.flatnonzero((depth.reshape(-1) > 0) & np.isfinite(xyz_metres).all(axis=1) & (xyz_metres[:, 2] > 0))
            if len(pixel_ids) == 0:
                print("No valid depth points; scene not saved.")
                continue
            xyz_mm = (xyz_metres[pixel_ids] * 1000.0).astype(np.float32)
            colors = color_rgb.reshape(-1, 3)[pixel_ids].copy()
            intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            scene = next_scene_dir(args.output)
            metadata = {
                "captured_at": timestamp,
                "device_serial": serial,
                "depth_scale_m_per_unit": float(depth_scale_m),
                "aligned_depth_intrinsics": intrinsics_to_dict(intrinsics),
                "depth_is_aligned_to_color": True,
                "depth_shape": list(depth.shape),
                "frame_number": int(depth_frame.get_frame_number()),
            }
            create_scene(scene, xyz_mm, colors, pixel_index=pixel_ids.astype(np.int32),
                         metadata=metadata, rgb_image=color_rgb, depth_z16=depth)
            print(f"Saved {scene} | {len(pixel_ids):,} points")
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
