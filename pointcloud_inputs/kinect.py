"""Microsoft Kinect 7-Scenes RGB-D input."""

from __future__ import annotations

import io
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

from .common import CameraIntrinsics, PointCloudFrame, frame_from_rgbd


KINECT_7SCENES_INTRINSICS = CameraIntrinsics(
    width=640,
    height=480,
    fx=585.0,
    fy=585.0,
    cx=320.0,
    cy=240.0,
)


def load_kinect_point_cloud(
    dataset_path: str | Path,
    frame_index: int,
    *,
    sequence: str | None = None,
    intrinsics: CameraIntrinsics = KINECT_7SCENES_INTRINSICS,
    min_depth_m: float = 0.3,
    max_depth_m: float = 5.0,
    cache_dir: str | Path | None = None,
    output_ply: str | Path | None = None,
) -> PointCloudFrame:
    """Load one Kinect 7-Scenes depth frame as a point cloud.

    ``dataset_path`` may be an extracted ``seq-XX`` directory, a sequence ZIP,
    a scene directory containing sequences, or an outer scene ZIP. For an outer
    ZIP, only the selected nested sequence ZIP is extracted into ``cache_dir``.

    The 7-Scenes color and depth streams are not registered. Therefore the raw
    color frame is exposed as ``frame.unregistered_rgb`` and is deliberately
    not attached to PLY vertices. Use ``frame.preview_rgb()`` for pixel picking.
    """

    source = resolve_kinect_sequence(
        dataset_path, sequence=sequence, cache_dir=cache_dir
    )
    depth_mm, color_rgb = read_kinect_frame(source, frame_index)
    height, width = depth_mm.shape
    scaled_intrinsics = intrinsics.scaled(width, height)
    invalid = (depth_mm == 0) | (depth_mm == 65535)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[invalid] = np.nan
    frame = frame_from_rgbd(
        depth_m,
        scaled_intrinsics,
        source="kinect-7scenes",
        unregistered_rgb=color_rgb,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        metadata={
            "dataset": "Microsoft 7-Scenes",
            "sequence_source": str(source),
            "frame_index": int(frame_index),
            "depth_unit": "millimeter",
            "depth_invalid_values": [0, 65535],
            "rgb_depth_registered": False,
            "intrinsics_assumed": scaled_intrinsics.as_dict(),
        },
    )
    if output_ply is not None:
        frame.save_ply(output_ply)
    return frame


def resolve_kinect_sequence(
    dataset_path: str | Path,
    *,
    sequence: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve a 7-Scenes source to one sequence directory or ZIP."""

    source = Path(dataset_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    requested = None if sequence is None else sequence.removesuffix(".zip")

    if source.is_file():
        if source.suffix.lower() != ".zip" or not zipfile.is_zipfile(source):
            raise ValueError(f"Expected a valid 7-Scenes ZIP or directory: {source}")
        with zipfile.ZipFile(source) as archive:
            files = [entry for entry in archive.infolist() if not entry.is_dir()]
            if any(entry.filename.endswith(".depth.png") for entry in files):
                return source
            nested = [
                entry
                for entry in files
                if re.search(r"(?:^|/)seq-\d+\.zip$", entry.filename, re.I)
            ]
            if requested is not None:
                nested = [entry for entry in nested if Path(entry.filename).stem == requested]
            if not nested:
                raise ValueError(f"No matching seq-XX.zip found inside {source}")
            nested.sort(key=lambda entry: (Path(entry.filename).stem != "seq-01", entry.filename))
            selected = nested[0]
            root = (
                Path(cache_dir).expanduser().resolve()
                if cache_dir is not None
                else source.parent / ".7scenes_sequence_cache" / source.stem
            )
            root.mkdir(parents=True, exist_ok=True)
            target = root / Path(selected.filename).name
            reusable = (
                target.is_file()
                and target.stat().st_size == selected.file_size
                and zipfile.is_zipfile(target)
            )
            if not reusable:
                temporary = target.with_suffix(target.suffix + ".partial")
                try:
                    with archive.open(selected) as src, temporary.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                    if not zipfile.is_zipfile(temporary):
                        raise ValueError("Extracted nested sequence ZIP is corrupt")
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            return target

    if any(source.glob("frame-*.depth.png")):
        return source
    candidates = [
        item
        for item in source.iterdir()
        if (
            item.is_dir()
            and item.name.startswith("seq-")
            and any(item.glob("frame-*.depth.png"))
        )
        or (
            item.is_file()
            and item.name.startswith("seq-")
            and item.suffix.lower() == ".zip"
            and zipfile.is_zipfile(item)
        )
    ]
    if requested is not None:
        candidates = [item for item in candidates if item.stem == requested]
    if not candidates:
        raise FileNotFoundError(f"No matching 7-Scenes sequence under {source}")
    return sorted(candidates, key=lambda item: (item.stem, item.is_file()))[0]


def read_kinect_frame(
    sequence_source: str | Path, frame_index: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read the original uint16 depth and optional RGB frame."""

    if frame_index < 0:
        raise ValueError("frame_index must be >= 0")
    source = Path(sequence_source).expanduser().resolve()
    depth_name = f"frame-{frame_index:06d}.depth.png"
    color_name = f"frame-{frame_index:06d}.color.png"

    if source.is_dir():
        depth_path = source / depth_name
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        depth = _read_png(depth_path.read_bytes(), rgb=False)
        color_path = source / color_name
        color = _read_png(color_path.read_bytes(), rgb=True) if color_path.is_file() else None
    else:
        if not zipfile.is_zipfile(source):
            raise ValueError(f"Not a valid sequence ZIP: {source}")
        with zipfile.ZipFile(source) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            depth_member = _find_member(names, depth_name)
            if depth_member is None:
                raise FileNotFoundError(f"Missing {depth_name} in {source}")
            color_member = _find_member(names, color_name)
            depth = _read_png(archive.read(depth_member), rgb=False)
            color = (
                None
                if color_member is None
                else _read_png(archive.read(color_member), rgb=True)
            )

    if depth.ndim != 2 or depth.dtype.kind not in "ui" or depth.dtype.itemsize < 2:
        raise ValueError(
            f"Expected an original 16-bit depth PNG, got {depth.shape} / {depth.dtype}"
        )
    if color is not None and color.shape[:2] != depth.shape:
        raise ValueError("Kinect color and depth dimensions differ")
    return depth, color


def _find_member(names: list[str], basename: str) -> str | None:
    matches = [name for name in names if name == basename or name.endswith("/" + basename)]
    return None if not matches else min(matches, key=len)


def _read_png(payload: bytes, *, rgb: bool) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pillow is required to read Kinect 7-Scenes PNG files") from exc
    with Image.open(io.BytesIO(payload)) as image:
        if rgb:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return np.asarray(image).copy()
