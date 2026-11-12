"""Run the shared detector on immutable saved scene points and optional ROI."""

from pathlib import Path

import numpy as np

from corner_dataset_tool.dataset_io import SceneData
from corner_dataset_tool.scene_navigation import scene_sort_key

from .dense_heatmap import run_dense_heatmap_mode
from .preprocessing import voxel_downsample_3d


def run_dataset_detector(input_path, cfg, *, region="roi"):
    if region not in ("roi", "scene"):
        raise ValueError("Dataset region must be roi or scene")
    source = Path(input_path).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset/scene directory not found: {source}")
    if not (source / "points.npz").is_file() and (source / "dataset").is_dir():
        source = source / "dataset"
    scenes = ([source] if (source / "points.npz").is_file() else
              sorted((p for p in source.iterdir() if p.is_dir() and (p / "points.npz").is_file()),
                     key=scene_sort_key))
    if not scenes:
        raise ValueError(f"No scenes with points.npz: {source}")
    for number, path in enumerate(scenes, 1):
        scene = SceneData.load(path)
        if region == "roi" and not (path / "annotations.json").is_file():
            raise ValueError(f"Saved ROI requires annotations.json: {path}; use --dataset-region scene")
        ids = scene.roi_indices() if region == "roi" else np.arange(len(scene.xyz_mm), dtype=np.int64)
        if not len(ids):
            raise ValueError(f"Empty dataset {region}: {path}")
        raw = np.asarray(scene.xyz_mm[ids], dtype=np.float64)
        points = voxel_downsample_3d(raw, cfg.voxel_3d_mm)
        if len(points) < cfg.min_neighbors:
            raise ValueError(f"Too few points after voxelization: {path}")
        print(f"[DATASET {number}/{len(scenes)}] {path.name} region={region} "
              f"raw={len(raw):,}/{len(scene.xyz_mm):,} voxel={len(points):,} | mm", flush=True)
        run_dense_heatmap_mode(points, cfg, raw_points_mm=raw,
                               rgb_points_mm=raw, rgb_colors_rgb=scene.rgb[ids], print_candidates=False)
