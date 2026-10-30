#!/home/choisuhyun/keypoint-corner-detector/.conda/kinect7scenes-minispinnet/bin/python
"""Register two Kinect 7-Scenes frames with FPS corners + BUFFER-X Mini-SpinNet.

This is a headless-friendly companion to
``kinect7scenes_fps_projection_length_inspector.py``.  It deliberately reuses
that file's corner detector, then computes 32-D descriptors with the pretrained
Mini-SpinNet descriptor from BUFFER-X.  The official CUDA-only PointNet2 query
operators are replaced by deterministic SciPy KD-tree queries; all learned
layers and their pretrained weights are unchanged.

cd /home/choisuhyun/keypoint-corner-detector

./kinect7scenes_minispinnet_registration.py \
  --source-frame 10 \
  --target-frame 50

Default demo (held-out RedKitchen test scene already present on this workstation):

    ./kinect7scenes_minispinnet_registration.py --no-gui

Choose any two frames from a 7-Scenes sequence:

    ./kinect7scenes_minispinnet_registration.py \
        --capture ~/datasets/7scenes/redkitchen/seq-03.zip \
        --source-frame 965 --target-frame 838

If the two selected frames do not overlap enough for direct matching, the
program automatically registers a short chain of intermediate sequence frames
and refines the composed endpoint transform.  Use ``--bridge-step 0`` to
disable that fallback.

The estimated 4x4 transform maps source-camera XYZ into target-camera XYZ.
Registration never uses the dataset poses; poses are loaded only afterwards to
report optional error metrics.

BUFFER-X: https://github.com/MIT-SPARK/BUFFER-X (MIT license)
Checkpoint: Hyungtae-Lim/BUFFER-X on Hugging Face
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import site
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
# Keep the dedicated environment isolated from user-site and ROS packages that
# may otherwise precede it through the workstation's global PYTHONPATH.
_user_site = str(site.getusersitepackages())
sys.path[:] = [
    entry
    for entry in sys.path
    if entry != _user_site and not entry.startswith("/opt/ros/")
]
CACHE_ROOT = SCRIPT_DIR / ".cache" / "kinect7scenes_minispinnet"
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_ROOT / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
for _directory in (Path(os.environ["NUMBA_CACHE_DIR"]), Path(os.environ["MPLCONFIGDIR"])):
    _directory.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

import kinect7scenes_fps_projection_length_inspector as detector


BUFFERX_COMMIT = "b56af186cdfef30b691a4866dbb5f5a55852f285"
CHECKPOINT_URL = (
    "https://huggingface.co/Hyungtae-Lim/BUFFER-X/resolve/"
    "f657b1f737fce0a8eab40c56204abaea27316b4a/snapshot/"
    "threedmatch/Desc/best.pth?download=true"
)
CHECKPOINT_SHA256 = "daf910b6e2a79c4346a6ef8eeb18b56ffc15b552cc82af091b8b6c0a18116eef"
DEFAULT_CHECKPOINT = SCRIPT_DIR / ".models" / "BUFFER-X" / "threedmatch" / "Desc" / "best.pth"
CACHE_FORMAT_VERSION = 2


@dataclass(frozen=True)
class DetectorRecipe:
    """The exact scale recipe shown in the source inspector's example."""

    voxel_3d_mm: float = 15.0
    voxel_2d_mm: float = 15.0
    normal_radius_x: float = 4.0
    normal_sigma_r_x: float = 0.75
    query_radius_x: float = 10.0
    boundary_radius_x: float = 4.0
    boundary_epsilon_x: float = 1.0
    tensor_radius_x: float = 4.0
    tensor_sigma_d_x: float = 1.5
    vote_line_sigma_x: float = 1.0
    corner_search_radius_x: float = 6.0
    surface_weight_min: float = 0.1
    ext_radius_x: float = 4.0
    ext_directions: int = 1024
    hough_min_score: float = 0.015
    ext_min_score: float = 0.08
    corner_nms_radius_x: float = 3.0
    min_depth_m: float = 0.3
    max_depth_m: float = 5.0


@dataclass
class FrameBundle:
    frame_index: int
    cloud_m: np.ndarray
    keypoints_m: np.ndarray
    scores: np.ndarray
    descriptors: np.ndarray


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and sha256_file(path) == CHECKPOINT_SHA256:
        return path
    if path.exists():
        raise RuntimeError(
            f"Checkpoint exists but SHA-256 is wrong: {path}. "
            "Remove it or pass --checkpoint with the official file."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    log(f"[MODEL] Downloading official BUFFER-X 3DMatch descriptor checkpoint -> {path}")
    try:
        urllib.request.urlretrieve(CHECKPOINT_URL, partial)
        actual = sha256_file(partial)
        if actual != CHECKPOINT_SHA256:
            raise RuntimeError(f"Downloaded checkpoint hash mismatch: {actual}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


# -----------------------------------------------------------------------------
# CPU/GPU-compatible Mini-SpinNet inference port.
# Learned layer names and shapes intentionally match BUFFER-X exactly.
# -----------------------------------------------------------------------------


def _pad_cylinder_2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = (kernel_size - 1) // 2
    if pad == 0:
        return x
    x = torch.cat((x[..., -pad:], x, x[..., :pad]), dim=3)
    zeros = x.new_zeros((x.shape[0], x.shape[1], pad, x.shape[3]))
    return torch.cat((zeros, x, zeros), dim=2)


def _pad_cylinder_3d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    # BUFFER-X pads elevation with zeros and azimuth circularly.  It does not
    # pad the radial dimension; the first 3x3x3 convolution collapses 3 -> 1.
    pad = (kernel_size - 1) // 2
    if pad == 0:
        return x
    x = torch.cat((x[..., -pad:], x, x[..., :pad]), dim=4)
    zeros = x.new_zeros((x.shape[0], x.shape[1], x.shape[2], pad, x.shape[4]))
    return torch.cat((zeros, x, zeros), dim=3)


class CylindricalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        ops: list[nn.Module] = []

        def conv3(in_channels: int, out_channels: int) -> None:
            ops.extend(
                (
                    nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3)),
                    nn.BatchNorm3d(out_channels, affine=False),
                    nn.ReLU(inplace=True),
                )
            )

        def conv2(in_channels: int, out_channels: int, *, final: bool = False) -> None:
            ops.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3)))
            if not final:
                ops.extend((nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(inplace=True)))

        conv3(16, 64)
        conv2(64, 64)
        conv2(64, 128)
        conv2(128, 128)
        conv2(128, 64)
        conv2(64, 64)
        conv2(64, 32)
        conv2(32, 32, final=True)
        self.ops = nn.ModuleList(ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for operation in self.ops:
            if isinstance(operation, nn.Conv3d):
                x = _pad_cylinder_3d(x, operation.kernel_size[1])
            elif isinstance(operation, nn.Conv2d):
                if x.ndim == 5:
                    x = x.squeeze(2)
                x = _pad_cylinder_2d(x, operation.kernel_size[0])
            x = operation(x)
        return x


class MiniSpinNet(nn.Module):
    """Only the learned portion of BUFFER-X ``models.patch_embedder.MiniSpinNet``."""

    rad_n = 3
    ele_n = 7
    azi_n = 20
    voxel_sample = 10

    def __init__(self) -> None:
        super().__init__()
        self.pnt_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.pool_layer = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )
        self.conv_net = CylindricalNet()

    def forward(self, invariant_voxels: torch.Tensor) -> torch.Tensor:
        # [B, 420, 10, 3] -> official Mini-SpinNet learned pipeline.
        x = invariant_voxels.permute(0, 3, 1, 2).contiguous()
        x = self.pnt_layer(x)
        x = F.max_pool2d(x, kernel_size=(1, x.shape[-1])).squeeze(3)
        x = x.reshape(x.shape[0], x.shape[1], self.rad_n, self.ele_n, self.azi_n)
        x = self.conv_net(x)
        weights = self.pool_layer(x)
        descriptor = F.avg_pool2d(x * weights, kernel_size=(x.shape[2], x.shape[3]))
        return F.normalize(descriptor.reshape(descriptor.shape[0], -1), p=2, dim=1)


def load_minispinnet(checkpoint: Path, device: torch.device) -> MiniSpinNet:
    model = MiniSpinNet()
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        state = torch.load(checkpoint, map_location="cpu")
    desc_state = {
        key.removeprefix("Desc."): value for key, value in state.items() if key.startswith("Desc.")
    }
    if not desc_state:
        raise RuntimeError(f"No Desc.* weights found in {checkpoint}")
    model.load_state_dict(desc_state, strict=True)
    model.eval().to(device)
    return model


def _voxel_centers() -> np.ndarray:
    beta = np.linspace(0.0, np.pi, MiniSpinNet.ele_n, endpoint=False)
    beta += np.pi / MiniSpinNet.ele_n / 2.0
    alpha = np.linspace(0.0, 2.0 * np.pi, MiniSpinNet.azi_n, endpoint=False)
    alpha += np.pi / MiniSpinNet.azi_n
    b_grid, a_grid = np.meshgrid(beta, alpha, indexing="ij")
    unit = np.stack(
        (
            np.sin(b_grid) * np.cos(a_grid),
            np.sin(b_grid) * np.sin(a_grid),
            np.cos(b_grid),
        ),
        axis=-1,
    )
    scales = np.arange(MiniSpinNet.rad_n, dtype=np.float64) / MiniSpinNet.rad_n
    scales += 1.0 / (2.0 * MiniSpinNet.rad_n)
    return (scales[:, None, None, None] * unit[None]).reshape(-1, 3)


VOXEL_CENTERS = _voxel_centers()


def _align_vector_to_z_row(normal: np.ndarray) -> np.ndarray:
    """Return a row-vector rotation such that ``normal @ R == [0,0,1]``."""
    a = np.asarray(normal, dtype=np.float64)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b = np.array([0.0, 0.0, 1.0])
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            column_rotation = np.eye(3)
        else:
            # Stable 180-degree rotation about x for an antiparallel normal.
            column_rotation = np.diag([1.0, -1.0, -1.0])
    else:
        vx, vy, vz = cross
        skew = np.array(((0.0, -vz, vy), (vz, 0.0, -vx), (-vy, vx, 0.0)))
        column_rotation = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
    return column_rotation.T


def _make_invariant_patch(
    cloud_tree: cKDTree,
    cloud_m: np.ndarray,
    keypoint_m: np.ndarray,
    radius_m: float,
    seed: int,
) -> np.ndarray:
    neighbor_ids = np.asarray(cloud_tree.query_ball_point(keypoint_m, radius_m), dtype=np.int64)
    if len(neighbor_ids) < 3:
        raise ValueError(f"Descriptor patch has only {len(neighbor_ids)} points")
    rng = np.random.default_rng(seed)
    if len(neighbor_ids) > 511:
        neighbor_ids = rng.choice(neighbor_ids, size=511, replace=False)
    else:
        neighbor_ids = rng.permutation(neighbor_ids)

    patch = np.repeat(np.asarray(keypoint_m, dtype=np.float64)[None], 512, axis=0)
    count = min(511, len(neighbor_ids))
    patch[:count] = cloud_m[neighbor_ids[:count]]
    patch[-1] = keypoint_m
    local = patch - keypoint_m

    covariance = local.T @ local
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    # BUFFER-X disambiguates the PCA normal toward the camera origin.
    if float(np.dot(normal, keypoint_m)) > 0.0:
        normal = -normal
    local = (local @ _align_vector_to_z_row(normal)) / radius_m

    # Pure KD-tree equivalent of BUFFER-X sphere_query.  Patch order is already
    # randomly sampled, so taking the first ten indices reproduces random local
    # voxel sampling without PointNet2's CUDA operator.
    local_tree = cKDTree(local)
    groups = local_tree.query_ball_point(VOXEL_CENTERS, 0.8 / MiniSpinNet.rad_n)
    sampled = np.zeros(
        (len(VOXEL_CENTERS), MiniSpinNet.voxel_sample, 3), dtype=np.float32
    )
    for voxel_index, ids in enumerate(groups):
        if ids:
            take = np.asarray(ids[: MiniSpinNet.voxel_sample], dtype=np.int64)
            sampled[voxel_index, : len(take)] = local[take]

    sampled = sampled.reshape(
        MiniSpinNet.rad_n,
        MiniSpinNet.ele_n,
        MiniSpinNet.azi_n,
        MiniSpinNet.voxel_sample,
        3,
    )
    for azi_index in range(MiniSpinNet.azi_n):
        angle = -azi_index * 2.0 * np.pi / MiniSpinNet.azi_n
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.array(
            ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
        )
        sampled[:, :, azi_index] = sampled[:, :, azi_index] @ rotation.T
    return sampled.reshape(-1, MiniSpinNet.voxel_sample, 3)


def describe_keypoints(
    model: MiniSpinNet,
    cloud_m: np.ndarray,
    keypoints_m: np.ndarray,
    radius_m: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(cloud_m)
    descriptors: list[np.ndarray] = []
    valid_ids: list[int] = []
    invariant_batch: list[np.ndarray] = []

    def flush() -> None:
        if not invariant_batch:
            return
        tensor = torch.from_numpy(np.stack(invariant_batch)).to(device)
        with torch.inference_mode():
            output = model(tensor).cpu().numpy()
        descriptors.extend(output)
        invariant_batch.clear()

    for index, keypoint in enumerate(keypoints_m):
        try:
            invariant_batch.append(
                _make_invariant_patch(tree, cloud_m, keypoint, radius_m, seed + index * 104729)
            )
            valid_ids.append(index)
            if len(invariant_batch) >= batch_size:
                flush()
        except (ValueError, np.linalg.LinAlgError) as exc:
            log(f"[DESC] skipping keypoint {index}: {exc}")
    flush()
    if not descriptors:
        raise RuntimeError("Mini-SpinNet could not describe any detector keypoint")
    result = np.asarray(descriptors, dtype=np.float32)
    norms = np.linalg.norm(result, axis=1)
    if not np.isfinite(result).all() or not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError("Mini-SpinNet produced invalid or non-normalized descriptors")
    return result, np.asarray(valid_ids, dtype=np.int64)


# -----------------------------------------------------------------------------
# Frame loading and the original FPS/Hough/projection-length corner detector.
# -----------------------------------------------------------------------------


def make_detector_config(recipe: DetectorRecipe) -> detector.BoundaryConfig:
    v3, v2 = recipe.voxel_3d_mm, recipe.voxel_2d_mm
    detector.MIN_DEPTH_M = recipe.min_depth_m
    detector.MAX_DEPTH_M = recipe.max_depth_m
    detector.VOXEL_3D_MM = v3
    detector.VOXEL_2D_MM = v2
    return detector.BoundaryConfig(
        voxel_3d_mm=v3,
        voxel_2d_mm=v2,
        normal_radius_mm=recipe.normal_radius_x * v3,
        sigma_r_mm=recipe.normal_sigma_r_x * v3,
        query_radius_mm=recipe.query_radius_x * v2,
        boundary_radius_mm=recipe.boundary_radius_x * v2,
        boundary_epsilon_mm=recipe.boundary_epsilon_x * v2,
        tensor_radius_mm=recipe.tensor_radius_x * v2,
        tensor_sigma_d_mm=recipe.tensor_sigma_d_x * v2,
        vote_line_sigma_mm=recipe.vote_line_sigma_x * v2,
        hough_min_score=recipe.hough_min_score,
        corner_search_radius_mm=recipe.corner_search_radius_x * v2,
        corner_step_max_mm=4.0 * v2,
        corner_refine_radius_mm=2.0 * v2,
        corner_converge_mm=1.0 * v2,
        ext_radius_mm=recipe.ext_radius_x * v3,
        ext_directions=recipe.ext_directions,
        ext_min_score=recipe.ext_min_score,
        ext_method="projection-length",
        ext_top_fraction=0.20,
        ext_trim_fraction=0.05,
        ext_min_span_mm=1.0,
        surface_weight_min=recipe.surface_weight_min,
        normal_iterations=5,
        normal_update_mode="synchronous",
        enable_iteration=False,
        enable_refine=False,
        show_3d=False,
    )


def load_frame_cloud(
    sequence: Path,
    frame_index: int,
    recipe: DetectorRecipe,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    depth_mm, _ = detector.read_kinect7scenes_frame(sequence, frame_index)
    height, width = depth_mm.shape
    valid = (depth_mm > 0) & (depth_mm != 65535)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m[~valid] = np.nan
    intrinsics = detector.SimIntrinsics(
        width,
        height,
        fx * width / 640.0,
        fy * height / 480.0,
        cx * width / 640.0,
        cy * height / 480.0,
    )
    raw_mm = detector.depth_to_xyz_mm(depth_m, intrinsics)
    cloud_mm = detector.voxel_downsample_3d(raw_mm, recipe.voxel_3d_mm)
    if len(cloud_mm) < 100:
        raise RuntimeError(f"Frame {frame_index} has only {len(cloud_mm)} valid voxel points")
    log(
        f"[FRAME {frame_index:06d}] raw={len(raw_mm):,}, "
        f"voxel({recipe.voxel_3d_mm:g} mm)={len(cloud_mm):,}"
    )
    return cloud_mm / 1000.0, (height, width)


def detector_cache_path(
    sequence: Path, frame_index: int, recipe: DetectorRecipe, fps_count: int
) -> Path:
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "source": str(sequence.resolve()),
        "frame": int(frame_index),
        "recipe": asdict(recipe),
        "fps_count": int(fps_count),
        "detector_build": detector.BUILD,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return CACHE_ROOT / "detections" / f"{sequence.stem}_{frame_index:06d}_{key}.npz"


def detect_keypoints(
    sequence: Path,
    frame_index: int,
    cloud_m: np.ndarray,
    cfg: detector.BoundaryConfig,
    recipe: DetectorRecipe,
    fps_count: int,
    normal_estimator: Any,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = detector_cache_path(sequence, frame_index, recipe, fps_count)
    if cache_path.is_file() and not force:
        with np.load(cache_path, allow_pickle=False) as saved:
            keypoints = np.asarray(saved["keypoints_m"], dtype=np.float64)
            scores = np.asarray(saved["scores"], dtype=np.float64)
        log(f"[DETECT {frame_index:06d}] cache hit: {len(keypoints)} corners ({cache_path})")
        return keypoints, scores

    cloud_mm = np.asarray(cloud_m, dtype=np.float64) * 1000.0
    scene_cache = detector.precompute_scene_cache(cloud_mm, cfg, normal_estimator)
    frozen = {"points_voxel_mm": cloud_mm, "scene_cache": scene_cache}
    started = time.perf_counter()
    result = detector.detect_corners_from_fps_scene(
        frozen,
        cfg,
        normal_estimator,
        count=fps_count,
        nms_radius_mm=recipe.corner_nms_radius_x * recipe.voxel_3d_mm,
    )
    corners = result["corners"]
    if not corners:
        raise RuntimeError(f"Detector found no keypoints in frame {frame_index}")
    keypoints = np.asarray([entry["position"] for entry in corners], dtype=np.float64) / 1000.0
    scores = np.asarray([entry["hough_score"] for entry in corners], dtype=np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, keypoints_m=keypoints, scores=scores)
    log(
        f"[DETECT {frame_index:06d}] {len(keypoints)} NMS corners in "
        f"{time.perf_counter() - started:.2f}s; cached at {cache_path}"
    )
    return keypoints, scores


def select_keypoints(
    keypoints: np.ndarray, scores: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    if maximum > 0:
        order = order[:maximum]
    return keypoints[order], scores[order]


def prepare_frame_bundle(
    sequence: Path,
    frame_index: int,
    recipe: DetectorRecipe,
    cfg: detector.BoundaryConfig,
    normal_estimator: Any,
    model: MiniSpinNet,
    device: torch.device,
    *,
    fps_count: int,
    max_keypoints: int,
    descriptor_radius_m: float,
    descriptor_batch_size: int,
    descriptor_seed: int,
    force_redetect: bool,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> FrameBundle:
    cloud_m, _ = load_frame_cloud(sequence, frame_index, recipe, fx, fy, cx, cy)
    keypoints_m, scores = detect_keypoints(
        sequence,
        frame_index,
        cloud_m,
        cfg,
        recipe,
        fps_count,
        normal_estimator,
        force_redetect,
    )
    keypoints_m, scores = select_keypoints(keypoints_m, scores, max_keypoints)
    descriptors, valid = describe_keypoints(
        model,
        cloud_m,
        keypoints_m,
        descriptor_radius_m,
        descriptor_batch_size,
        device,
        descriptor_seed,
    )
    return FrameBundle(
        frame_index=frame_index,
        cloud_m=cloud_m,
        keypoints_m=keypoints_m[valid],
        scores=scores[valid],
        descriptors=descriptors,
    )


# -----------------------------------------------------------------------------
# Descriptor matching, rigid registration, ICP, pose-only evaluation.
# -----------------------------------------------------------------------------


def mutual_descriptor_matches(
    source: np.ndarray,
    target: np.ndarray,
    ratio_threshold: float,
    max_distance: float,
    min_matches: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(source @ target.T, -1.0, 1.0), 0.0))
    if target.shape[0] < 2 or source.shape[0] < 2:
        raise RuntimeError("At least two descriptors per frame are required")
    source_order = np.argsort(distances, axis=1)
    target_order = np.argsort(distances, axis=0)
    source_best = source_order[:, 0]
    target_best = target_order[0, :]
    source_ratio = distances[np.arange(len(source)), source_best] / (
        distances[np.arange(len(source)), source_order[:, 1]] + 1e-12
    )
    target_ratio = distances[target_best, np.arange(len(target))] / (
        distances[target_order[1, :], np.arange(len(target))] + 1e-12
    )

    def filtered(ratio: float, distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src_ids = np.arange(len(source), dtype=np.int64)
        tgt_ids = source_best
        keep = target_best[tgt_ids] == src_ids
        keep &= source_ratio <= ratio
        keep &= target_ratio[tgt_ids] <= ratio
        keep &= distances[src_ids, tgt_ids] <= distance
        return src_ids[keep], tgt_ids[keep], distances[src_ids[keep], tgt_ids[keep]]

    src_ids, tgt_ids, values = filtered(ratio_threshold, max_distance)
    if len(src_ids) < min_matches:
        relaxed_ratio = max(ratio_threshold, 1.0)
        relaxed_distance = max(max_distance, 1.5)
        log(
            f"[MATCH] only {len(src_ids)} strict matches; relaxing mutual-NN filters "
            f"to ratio={relaxed_ratio:g}, distance={relaxed_distance:g}"
        )
        src_ids, tgt_ids, values = filtered(relaxed_ratio, relaxed_distance)
    if len(src_ids) < 3:
        raise RuntimeError(f"Only {len(src_ids)} mutual Mini-SpinNet matches; need at least 3")
    order = np.argsort(values)
    return src_ids[order], tgt_ids[order], values[order]


def make_point_cloud(points: np.ndarray, color: tuple[float, float, float] | None = None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    if color is not None:
        cloud.paint_uniform_color(color)
    return cloud


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares Kabsch fit mapping source row points onto target."""
    if len(source) < 3 or source.shape != target.shape:
        raise ValueError("Rigid fit needs at least three paired 3-D points")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def run_registration(
    source_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    source_match_ids: np.ndarray,
    target_match_ids: np.ndarray,
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    ransac_distance: float,
    ransac_iterations: int,
    icp_distance: float,
    icp_iterations: int,
    use_icp: bool,
    seed: int,
) -> dict[str, Any]:
    o3d.utility.random.seed(seed)
    source_kpcd = make_point_cloud(source_keypoints)
    target_kpcd = make_point_cloud(target_keypoints)
    correspondence = np.column_stack((source_match_ids, target_match_ids)).astype(np.int32)
    started = time.perf_counter()
    ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source_kpcd,
        target_kpcd,
        o3d.utility.Vector2iVector(correspondence),
        ransac_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_distance),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iterations, 0.999),
    )
    ransac_time = time.perf_counter() - started
    raw_ransac_transform = np.asarray(ransac.transformation, dtype=np.float64)
    matched_source = source_keypoints[source_match_ids]
    matched_target = target_keypoints[target_match_ids]
    ransac_residuals = np.linalg.norm(
        transform_points(matched_source, raw_ransac_transform) - matched_target, axis=1
    )
    ransac_inliers = ransac_residuals <= ransac_distance
    required_inliers = max(6, int(math.ceil(0.10 * len(correspondence))))
    if int(np.count_nonzero(ransac_inliers)) < required_inliers:
        raise RuntimeError(
            f"RANSAC found only {np.count_nonzero(ransac_inliers)} inlier descriptor matches; "
            f"need at least {required_inliers}"
        )

    # Open3D's RANSAC hypothesis is based on a minimal sample.  Refit all of
    # its candidate-pair inliers, then reselect/refit once for a less noisy
    # initialization before dense ICP.
    ransac_transform = fit_rigid_transform(
        matched_source[ransac_inliers], matched_target[ransac_inliers]
    )
    ransac_residuals = np.linalg.norm(
        transform_points(matched_source, ransac_transform) - matched_target, axis=1
    )
    ransac_inliers = ransac_residuals <= ransac_distance
    if int(np.count_nonzero(ransac_inliers)) >= 3:
        ransac_transform = fit_rigid_transform(
            matched_source[ransac_inliers], matched_target[ransac_inliers]
        )
        ransac_residuals = np.linalg.norm(
            transform_points(matched_source, ransac_transform) - matched_target, axis=1
        )
        ransac_inliers = ransac_residuals <= ransac_distance

    final_transform = ransac_transform.copy()
    icp = None
    icp_time = 0.0
    icp_rejected = False
    source_pcd = make_point_cloud(source_cloud)
    target_pcd = make_point_cloud(target_cloud)
    if use_icp:
        started = time.perf_counter()
        target_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.060, max_nn=50)
        )
        target_pcd.normalize_normals()
        thresholds = (icp_distance, icp_distance * 0.53, icp_distance * 0.27)
        iteration_counts = (
            max(10, int(round(icp_iterations * 0.50))),
            max(10, int(round(icp_iterations * 0.31))),
            max(10, icp_iterations - int(round(icp_iterations * 0.81))),
        )
        for threshold, iterations in zip(thresholds, iteration_counts):
            try:
                loss = o3d.pipelines.registration.TukeyLoss(k=threshold)
                estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
            except TypeError:
                estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            icp = o3d.pipelines.registration.registration_icp(
                source_pcd,
                target_pcd,
                threshold,
                final_transform,
                estimation,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
            )
            final_transform = np.asarray(icp.transformation, dtype=np.float64)
        icp_time = time.perf_counter() - started

        # Do not accept an ICP update that destroys the descriptor-supported
        # coarse solution (possible for repetitive planar office geometry).
        candidate_residuals = np.linalg.norm(
            transform_points(matched_source, final_transform) - matched_target, axis=1
        )
        candidate_inliers = candidate_residuals <= ransac_distance
        drift_m, drift_deg = transform_delta(ransac_transform, final_transform)
        if (
            int(np.count_nonzero(candidate_inliers)) < max(
                3, int(np.count_nonzero(ransac_inliers) // 2)
            )
            or drift_m > 0.20
            or drift_deg > 15.0
        ):
            log(
                "[ICP] rejected: descriptor support deteriorated or update drifted "
                f"({drift_m:.3f} m, {drift_deg:.2f} deg)"
            )
            final_transform = ransac_transform.copy()
            icp_rejected = True

    final_residuals = np.linalg.norm(
        transform_points(matched_source, final_transform) - matched_target, axis=1
    )
    final_inliers = final_residuals <= ransac_distance
    dense_distance = 0.030
    coarse_evaluation = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd, dense_distance, ransac_transform
    )
    final_evaluation = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd, dense_distance, final_transform
    )
    return {
        "ransac": ransac,
        "icp": icp,
        "raw_ransac_transform": raw_ransac_transform,
        "ransac_transform": ransac_transform,
        "transform": final_transform,
        "ransac_residuals": ransac_residuals,
        "ransac_inliers": ransac_inliers,
        "residuals": final_residuals,
        "inliers": final_inliers,
        "ransac_time_s": ransac_time,
        "icp_time_s": icp_time,
        "icp_rejected": icp_rejected,
        "dense_evaluation_distance_m": dense_distance,
        "coarse_dense_fitness": float(coarse_evaluation.fitness),
        "coarse_dense_rmse_m": float(coarse_evaluation.inlier_rmse),
        "final_dense_fitness": float(final_evaluation.fitness),
        "final_dense_rmse_m": float(final_evaluation.inlier_rmse),
    }


def read_pose(sequence: Path, frame_index: int) -> np.ndarray | None:
    name = f"frame-{frame_index:06d}.pose.txt"
    try:
        if sequence.is_dir():
            path = sequence / name
            if not path.is_file():
                return None
            pose = np.loadtxt(path)
        else:
            with zipfile.ZipFile(sequence) as archive:
                matches = [
                    item for item in archive.namelist() if item == name or item.endswith("/" + name)
                ]
                if not matches:
                    return None
                pose = np.loadtxt(io.StringIO(archive.read(min(matches, key=len)).decode()))
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    # The text poses are rounded, so their 3x3 blocks are close to, but not
    # exactly on, SO(3).  Project them before inversion/error evaluation to
    # avoid biased geodesic angles (including spurious 0.000 degree errors).
    pose = pose.copy()
    pose[:3, :3] = project_rotation_matrix(pose[:3, :3])
    pose[3] = (0.0, 0.0, 0.0, 1.0)
    return pose


def project_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Project a finite 3x3 matrix to the nearest proper rotation."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Rotation matrix must be a finite 3x3 array")
    left, _, right_t = np.linalg.svd(matrix)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def transform_error(estimate: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    translation_m = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
    estimate_rotation = project_rotation_matrix(estimate[:3, :3])
    truth_rotation = project_rotation_matrix(truth[:3, :3])
    relative_rotation = estimate_rotation @ truth_rotation.T
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    return translation_m, float(np.degrees(np.arccos(cosine)))


def transform_delta(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Translation/rotation delta between two source-to-target transforms."""
    first = np.asarray(first, dtype=np.float64).copy()
    second = np.asarray(second, dtype=np.float64).copy()
    first[:3, :3] = project_rotation_matrix(first[:3, :3])
    second[:3, :3] = project_rotation_matrix(second[:3, :3])
    relative = np.linalg.inv(first) @ second
    translation = float(np.linalg.norm(relative[:3, 3]))
    cosine = float(np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return translation, float(np.degrees(np.arccos(cosine)))


def bridge_frame_indices(
    source_frame: int, target_frame: int, requested_step: int, max_legs: int
) -> list[int]:
    gap = abs(target_frame - source_frame)
    if gap == 0:
        return [source_frame]
    step = max(1, int(requested_step))
    step = max(step, int(math.ceil(gap / max(1, max_legs))))
    direction = 1 if target_frame > source_frame else -1
    frames = [source_frame]
    current = source_frame
    while abs(target_frame - current) > step:
        current += direction * step
        frames.append(current)
    frames.append(target_frame)
    return frames


def run_bridge_registration(
    frames: list[int],
    bundles: dict[int, FrameBundle],
    bundle_getter: Any,
    endpoint_source_match_ids: np.ndarray,
    endpoint_target_match_ids: np.ndarray,
    *,
    ratio_threshold: float,
    max_descriptor_distance: float,
    ransac_distance: float,
    ransac_iterations: int,
    icp_distance: float,
    icp_iterations: int,
    use_icp: bool,
    seed: int,
) -> dict[str, Any]:
    """Compose reliable short-baseline registrations when endpoints barely overlap."""
    composed = np.eye(4)
    legs: list[dict[str, Any]] = []
    support_residuals: list[np.ndarray] = []
    total_ransac_time = 0.0
    total_icp_time = 0.0

    log("[BRIDGE] direct registration failed; chaining " + " -> ".join(map(str, frames)))
    for leg_number, (source_index, target_index) in enumerate(zip(frames[:-1], frames[1:]), 1):
        if source_index not in bundles:
            bundles[source_index] = bundle_getter(source_index)
        if target_index not in bundles:
            bundles[target_index] = bundle_getter(target_index)
        source = bundles[source_index]
        target = bundles[target_index]
        source_ids, target_ids, descriptor_distances = mutual_descriptor_matches(
            source.descriptors,
            target.descriptors,
            ratio_threshold,
            max_descriptor_distance,
        )
        try:
            leg = run_registration(
                source.keypoints_m,
                target.keypoints_m,
                source_ids,
                target_ids,
                source.cloud_m,
                target.cloud_m,
                ransac_distance,
                ransac_iterations,
                icp_distance,
                icp_iterations,
                use_icp,
                seed + leg_number,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Bridge leg {source_index}->{target_index} failed: {error}. "
                "Retry with a smaller --bridge-step."
            ) from error
        composed = leg["transform"] @ composed
        leg_inliers = int(np.count_nonzero(leg["inliers"]))
        support_residuals.append(leg["residuals"][leg["inliers"]])
        total_ransac_time += float(leg["ransac_time_s"])
        total_icp_time += float(leg["icp_time_s"])
        legs.append(
            {
                "source_frame": source_index,
                "target_frame": target_index,
                "descriptor_matches": int(len(source_ids)),
                "inlier_count": leg_inliers,
                "dense_fitness_30mm": float(leg["final_dense_fitness"]),
                "transform": leg["transform"].tolist(),
                "descriptor_distance_mean": float(np.mean(descriptor_distances)),
            }
        )
        log(
            f"[BRIDGE {leg_number}/{len(frames)-1}] {source_index}->{target_index}: "
            f"matches={len(source_ids)}, inliers={leg_inliers}, "
            f"dense@30mm={leg['final_dense_fitness']:.3f}"
        )

    source = bundles[frames[0]]
    target = bundles[frames[-1]]
    source_pcd = make_point_cloud(source.cloud_m)
    target_pcd = make_point_cloud(target.cloud_m)
    coarse_transform = composed.copy()
    dense_distance = 0.030
    coarse_evaluation = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd, dense_distance, coarse_transform
    )

    final_transform = coarse_transform.copy()
    endpoint_icp = None
    endpoint_icp_time = 0.0
    icp_rejected = False
    if use_icp:
        target_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.060, max_nn=50)
        )
        target_pcd.normalize_normals()
        started = time.perf_counter()
        coarse_threshold = max(0.10, icp_distance)
        thresholds = (coarse_threshold, max(0.05, coarse_threshold * 0.5), 0.025)
        iteration_counts = (
            max(20, int(round(icp_iterations * 0.50))),
            max(15, int(round(icp_iterations * 0.31))),
            max(10, icp_iterations - int(round(icp_iterations * 0.81))),
        )
        for threshold, iterations in zip(thresholds, iteration_counts):
            try:
                loss = o3d.pipelines.registration.TukeyLoss(k=threshold)
                estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
            except TypeError:
                estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            endpoint_icp = o3d.pipelines.registration.registration_icp(
                source_pcd,
                target_pcd,
                threshold,
                final_transform,
                estimation,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations),
            )
            final_transform = np.asarray(endpoint_icp.transformation, dtype=np.float64)
        endpoint_icp_time = time.perf_counter() - started
        final_evaluation = o3d.pipelines.registration.evaluate_registration(
            source_pcd, target_pcd, dense_distance, final_transform
        )
        drift_m, drift_deg = transform_delta(coarse_transform, final_transform)
        if (
            drift_m > 0.25
            or drift_deg > 15.0
            or final_evaluation.fitness < 0.8 * coarse_evaluation.fitness
        ):
            log(
                f"[BRIDGE ICP] rejected endpoint refinement ({drift_m:.3f} m, "
                f"{drift_deg:.2f} deg)"
            )
            final_transform = coarse_transform.copy()
            icp_rejected = True
    total_icp_time += endpoint_icp_time

    final_evaluation = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd, dense_distance, final_transform
    )
    matched_source = source.keypoints_m[endpoint_source_match_ids]
    matched_target = target.keypoints_m[endpoint_target_match_ids]
    coarse_residuals = np.linalg.norm(
        transform_points(matched_source, coarse_transform) - matched_target, axis=1
    )
    final_residuals = np.linalg.norm(
        transform_points(matched_source, final_transform) - matched_target, axis=1
    )
    all_support = np.concatenate(support_residuals) if support_residuals else np.empty(0)
    return {
        "mode": "bridge",
        "bridge_frames": frames,
        "bridge_legs": legs,
        "bridge_support_matches": int(sum(item["descriptor_matches"] for item in legs)),
        "bridge_support_inliers": int(sum(item["inlier_count"] for item in legs)),
        "bridge_support_rmse_m": None
        if len(all_support) == 0
        else float(np.sqrt(np.mean(all_support**2))),
        "ransac": None,
        "icp": endpoint_icp,
        "raw_ransac_transform": coarse_transform,
        "ransac_transform": coarse_transform,
        "transform": final_transform,
        "ransac_residuals": coarse_residuals,
        "ransac_inliers": coarse_residuals <= ransac_distance,
        "residuals": final_residuals,
        "inliers": final_residuals <= ransac_distance,
        "ransac_time_s": total_ransac_time,
        "icp_time_s": total_icp_time,
        "icp_rejected": icp_rejected,
        "dense_evaluation_distance_m": dense_distance,
        "coarse_dense_fitness": float(coarse_evaluation.fitness),
        "coarse_dense_rmse_m": float(coarse_evaluation.inlier_rmse),
        "final_dense_fitness": float(final_evaluation.fitness),
        "final_dense_rmse_m": float(final_evaluation.inlier_rmse),
    }


def save_preview(
    output: Path,
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    source_matches: np.ndarray,
    target_matches: np.ndarray,
    inlier_mask: np.ndarray,
    seed: int,
    registration_mode: str,
) -> None:
    rng = np.random.default_rng(seed)

    def sample(points: np.ndarray, count: int = 18000) -> np.ndarray:
        if len(points) <= count:
            return points
        return points[rng.choice(len(points), count, replace=False)]

    source_view = sample(source)
    target_view = sample(target)
    aligned_source = transform_points(source_view, transform)
    transformed_matches = transform_points(source_matches, transform)
    figure = plt.figure(figsize=(14, 7), dpi=150)
    axes = [figure.add_subplot(121, projection="3d"), figure.add_subplot(122, projection="3d")]

    axes[0].scatter(*target_view.T, s=0.35, c="#27b6d6", alpha=0.55, rasterized=True)
    axes[0].scatter(*source_view.T, s=0.35, c="#ff8a35", alpha=0.55, rasterized=True)
    axes[0].set_title("Before registration (camera coordinates)")
    axes[1].scatter(*target_view.T, s=0.35, c="#27b6d6", alpha=0.55, rasterized=True)
    axes[1].scatter(*aligned_source.T, s=0.35, c="#ff8a35", alpha=0.55, rasterized=True)
    if registration_mode == "bridge":
        axes[1].set_title("After temporal Mini-SpinNet bridge + endpoint ICP")
    else:
        axes[1].set_title("After Mini-SpinNet + RANSAC + ICP")
    outlier_ids = np.flatnonzero(~inlier_mask)[:80]
    for index in outlier_ids:
        line = np.vstack((transformed_matches[index], target_matches[index]))
        axes[1].plot(*line.T, color="#ef6a6a", linewidth=0.40, alpha=0.24)
    inlier_ids = np.flatnonzero(inlier_mask)[:80]
    for index in inlier_ids:
        line = np.vstack((transformed_matches[index], target_matches[index]))
        axes[1].plot(*line.T, color="#56e39f", linewidth=0.75, alpha=0.82)

    for axis in axes:
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        axis.view_init(elev=20, azim=-78)
        axis.set_box_aspect((1.0, 0.75, 0.75))
    figure.suptitle(
        "target = cyan, source = orange, accepted matches = green, rejected matches = red",
        y=0.985,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def correspondence_display_shifts(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-overlapping, centered offsets used only for match visualization."""
    source_min, source_max = source.min(axis=0), source.max(axis=0)
    target_min, target_max = target.min(axis=0), target.max(axis=0)
    source_extent = source_max - source_min
    target_extent = target_max - target_min
    gap = max(0.25, 0.25 * float(max(np.linalg.norm(source_extent), np.linalg.norm(target_extent))))
    center_distance = 0.5 * float(source_extent[0] + target_extent[0]) + gap
    source_center = 0.5 * (source_min + source_max)
    target_center = 0.5 * (target_min + target_max)
    source_shift = np.array((-0.5 * center_distance, 0.0, 0.0)) - source_center
    target_shift = np.array((0.5 * center_distance, 0.0, 0.0)) - target_center
    return source_shift, target_shift


def save_correspondence_preview(
    output: Path,
    source: np.ndarray,
    target: np.ndarray,
    source_matches: np.ndarray,
    target_matches: np.ndarray,
    descriptor_distances: np.ndarray,
    inlier_mask: np.ndarray,
    seed: int,
    max_lines: int = 160,
) -> None:
    """Save a classic side-by-side descriptor-correspondence visualization."""
    rng = np.random.default_rng(seed)

    def sample(points: np.ndarray, count: int = 16000) -> np.ndarray:
        if len(points) <= count:
            return points
        return points[rng.choice(len(points), count, replace=False)]

    source_view = sample(source)
    target_view = sample(target)
    source_shift, target_shift = correspondence_display_shifts(source, target)

    source_display = source_view + source_shift
    target_display = target_view + target_shift
    source_match_display = source_matches + source_shift
    target_match_display = target_matches + target_shift

    inlier_ids = np.flatnonzero(inlier_mask)
    outlier_ids = np.flatnonzero(~inlier_mask)
    # Always retain every accepted match first, then the strongest rejected
    # descriptor matches until the plot reaches its clutter limit.
    if len(outlier_ids):
        outlier_ids = outlier_ids[np.argsort(descriptor_distances[outlier_ids])]
    keep_inliers = inlier_ids[:max_lines]
    keep_outliers = outlier_ids[: max(0, max_lines - len(keep_inliers))]

    figure = plt.figure(figsize=(14, 8), dpi=160)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*source_display.T, s=0.32, c="#ff8a35", alpha=0.42, rasterized=True)
    axis.scatter(*target_display.T, s=0.32, c="#27b6d6", alpha=0.42, rasterized=True)

    for index in keep_outliers:
        line = np.vstack((source_match_display[index], target_match_display[index]))
        axis.plot(*line.T, color="#ef6a6a", linewidth=0.42, alpha=0.26)
    for index in keep_inliers:
        line = np.vstack((source_match_display[index], target_match_display[index]))
        axis.plot(*line.T, color="#39d98a", linewidth=1.05, alpha=0.90)

    displayed = np.concatenate((keep_inliers, keep_outliers))
    if len(displayed):
        axis.scatter(
            *source_match_display[displayed].T,
            s=8.0,
            c="#ffcf9b",
            edgecolors="none",
            alpha=0.9,
        )
        axis.scatter(
            *target_match_display[displayed].T,
            s=8.0,
            c="#a6efff",
            edgecolors="none",
            alpha=0.9,
        )

    axis.text2D(
        0.02,
        0.96,
        f"mutual matches: {len(inlier_mask)}   accepted: {int(np.count_nonzero(inlier_mask))}"
        f"   rejected: {int(np.count_nonzero(~inlier_mask))}",
        transform=axis.transAxes,
    )
    axis.set_title(
        "Mini-SpinNet descriptor correspondences\n"
        "source and target are separated only for visualization"
    )
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")
    axis.view_init(elev=18, azim=-72)
    axis.set_box_aspect((1.7, 0.8, 0.8))
    figure.text(
        0.5,
        0.025,
        "source = orange, target = cyan, geometric inlier = green, rejected match = red",
        ha="center",
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def save_outputs(
    output_dir: Path,
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    source_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    source_match_ids: np.ndarray,
    target_match_ids: np.ndarray,
    descriptor_distances: np.ndarray,
    registration: dict[str, Any],
    payload: dict[str, Any],
    seed: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = registration["transform"]
    transformed_source = transform_points(source_cloud, transform)
    transformed_keypoints = transform_points(source_keypoints, transform)

    target_pcd = make_point_cloud(target_cloud, (0.10, 0.70, 0.85))
    source_pcd = make_point_cloud(transformed_source, (1.00, 0.42, 0.10))
    merged = source_pcd + target_pcd
    paths = {
        "transform": output_dir / "transform_source_to_target.txt",
        "result": output_dir / "registration_result.json",
        "preview": output_dir / "registration_preview.png",
        "correspondences": output_dir / "correspondence_lines.png",
        "source_registered": output_dir / "source_registered.ply",
        "target": output_dir / "target.ply",
        "merged": output_dir / "registered_merged.ply",
        "matches": output_dir / "descriptor_matches.npz",
    }
    np.savetxt(paths["transform"], transform, fmt="%.10f")
    o3d.io.write_point_cloud(str(paths["source_registered"]), source_pcd, write_ascii=False)
    o3d.io.write_point_cloud(str(paths["target"]), target_pcd, write_ascii=False)
    o3d.io.write_point_cloud(str(paths["merged"]), merged, write_ascii=False)
    np.savez_compressed(
        paths["matches"],
        source_keypoints_m=source_keypoints,
        target_keypoints_m=target_keypoints,
        source_match_ids=source_match_ids,
        target_match_ids=target_match_ids,
        descriptor_distances=descriptor_distances,
        geometric_residuals_m=registration["residuals"],
        inlier_mask=registration["inliers"],
        transformed_source_keypoints_m=transformed_keypoints,
    )
    save_preview(
        paths["preview"],
        source_cloud,
        target_cloud,
        transform,
        source_keypoints[source_match_ids],
        target_keypoints[target_match_ids],
        registration["inliers"],
        seed,
        registration.get("mode", "direct"),
    )
    save_correspondence_preview(
        paths["correspondences"],
        source_cloud,
        target_cloud,
        source_keypoints[source_match_ids],
        target_keypoints[target_match_ids],
        descriptor_distances,
        registration["inliers"],
        seed,
    )
    paths["result"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def show_interactive(
    source_cloud: np.ndarray,
    target_cloud: np.ndarray,
    source_keypoints: np.ndarray,
    target_keypoints: np.ndarray,
    source_match_ids: np.ndarray,
    target_match_ids: np.ndarray,
    inliers: np.ndarray,
    transform: np.ndarray,
) -> None:
    if not (
        len(source_match_ids) == len(target_match_ids) == len(inliers)
    ):
        raise ValueError("GUI correspondence indices and inlier mask must have equal lengths")
    inliers = np.asarray(inliers, dtype=bool)

    def line_set(
        source_keys: np.ndarray,
        target_keys: np.ndarray,
    ) -> Any | None:
        match_pairs = np.column_stack((source_match_ids, target_match_ids))
        if not len(match_pairs):
            return None
        points = np.vstack((source_keys, target_keys))
        lines = np.column_stack(
            (match_pairs[:, 0], match_pairs[:, 1] + len(source_keypoints))
        )
        geometry = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector2iVector(lines.astype(np.int32)),
        )
        colors = np.tile(np.array((0.95, 0.25, 0.25)), (len(match_pairs), 1))
        colors[inliers] = np.array((0.35, 0.95, 0.55))
        geometry.colors = o3d.utility.Vector3dVector(colors)
        return geometry

    def scene_geometries(mode: str) -> list[Any]:
        if mode == "registered":
            source_view = transform_points(source_cloud, transform)
            source_keys = transform_points(source_keypoints, transform)
            target_view = target_cloud
            target_keys = target_keypoints
        else:
            source_shift, target_shift = correspondence_display_shifts(
                source_cloud, target_cloud
            )
            source_view = source_cloud + source_shift
            source_keys = source_keypoints + source_shift
            target_view = target_cloud + target_shift
            target_keys = target_keypoints + target_shift

        geometries: list[Any] = [
            make_point_cloud(target_view, (0.10, 0.70, 0.85)),
            make_point_cloud(source_view, (1.00, 0.42, 0.10)),
            make_point_cloud(target_keys, (0.15, 1.00, 0.30)),
            make_point_cloud(source_keys, (1.00, 0.10, 0.75)),
        ]
        matches = line_set(source_keys, target_keys)
        if matches is not None:
            geometries.append(matches)
        return geometries

    scenes = {
        "correspondences": scene_geometries("correspondences"),
        "registered": scene_geometries("registered"),
    }
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    created = visualizer.create_window(
        window_name="7-Scenes | C: correspondences, R: registered result",
        width=1280,
        height=800,
    )
    if not created:
        raise RuntimeError("Open3D could not create the interactive visualization window")

    state = {"mode": ""}

    def set_scene(vis: Any, mode: str) -> bool:
        vis.clear_geometries()
        for geometry in scenes[mode]:
            vis.add_geometry(geometry, reset_bounding_box=False)
        vis.reset_view_point(True)
        vis.update_renderer()
        state["mode"] = mode
        if mode == "correspondences":
            log(
                "[GUI correspondence] display-only separation; accepted=green, "
                "rejected=red (press R for registered result)"
            )
        else:
            log(
                "[GUI registered] target=cyan, source=orange; accepted=green, "
                "rejected=red (press C for separated correspondences)"
            )
        return False

    visualizer.register_key_callback(
        ord("C"), lambda vis: set_scene(vis, "correspondences")
    )
    visualizer.register_key_callback(ord("R"), lambda vis: set_scene(vis, "registered"))
    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray((0.035, 0.035, 0.045))
    render_option.point_size = 2.0
    render_option.line_width = 1.5
    log("[GUI controls] C=separated correspondence view, R=registered scene, Q=close")
    set_scene(visualizer, "correspondences")
    visualizer.run()
    visualizer.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Register two Kinect 7-Scenes frames using the existing corner detector "
        "and a pretrained BUFFER-X Mini-SpinNet descriptor.",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=Path.home() / "datasets/7scenes/redkitchen/seq-03.zip",
        help="Extracted seq-XX folder, seq-XX.zip, outer scene zip, or scene folder",
    )
    parser.add_argument("--kinect-seq", default=None, help="Sequence name for a scene folder/zip")
    parser.add_argument("--source-frame", type=int, default=10)
    parser.add_argument("--target-frame", type=int, default=50)
    parser.add_argument("--fps-count", type=int, default=500)
    parser.add_argument("--max-keypoints", type=int, default=500, help="0 keeps every corner")
    parser.add_argument("--descriptor-radius-m", type=float, default=0.30)
    parser.add_argument("--descriptor-batch-size", type=int, default=24)
    parser.add_argument("--ratio-threshold", type=float, default=0.98)
    parser.add_argument("--max-descriptor-distance", type=float, default=1.20)
    parser.add_argument("--ransac-distance-m", type=float, default=0.10)
    parser.add_argument("--ransac-iterations", type=int, default=200_000)
    parser.add_argument(
        "--icp-distance-m",
        type=float,
        default=0.075,
        help="Coarsest ICP threshold; two finer levels use 0.53x and 0.27x",
    )
    parser.add_argument("--icp-iterations", type=int, default=80)
    parser.add_argument(
        "--bridge-step",
        type=int,
        default=20,
        help="On direct failure, chain intermediate frames at this index stride; 0 disables",
    )
    parser.add_argument(
        "--bridge-fps-count",
        type=int,
        default=500,
        help="FPS detector seed count for newly loaded intermediate bridge frames",
    )
    parser.add_argument(
        "--bridge-max-legs",
        type=int,
        default=20,
        help="Maximum number of temporal registration legs in the automatic bridge",
    )
    parser.add_argument("--no-icp", action="store_true")
    parser.add_argument("--force-redetect", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--normal-module", type=Path, default=None)
    parser.add_argument("--kinect-fx", type=float, default=585.0)
    parser.add_argument("--kinect-fy", type=float, default=585.0)
    parser.add_argument("--kinect-cx", type=float, default=320.0)
    parser.add_argument("--kinect-cy", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-gui", action="store_true", help="Save visualization but do not open it")
    parser.add_argument("--no-gt-evaluation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.source_frame < 0 or args.target_frame < 0 or args.source_frame == args.target_frame:
        raise ValueError("Choose two different non-negative frame indices")
    positive = (
        args.fps_count,
        args.descriptor_radius_m,
        args.descriptor_batch_size,
        args.ransac_distance_m,
        args.ransac_iterations,
        args.icp_distance_m,
        args.icp_iterations,
        args.bridge_fps_count,
        args.bridge_max_legs,
    )
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError("Counts, radii, thresholds, and iteration limits must be positive")
    if not 0.0 < args.ratio_threshold <= 1.0:
        raise ValueError("--ratio-threshold must be in (0,1]")
    if not 0.0 < args.max_descriptor_distance <= 2.0:
        raise ValueError("--max-descriptor-distance must be in (0,2]")
    if args.bridge_step < 0:
        raise ValueError("--bridge-step must be non-negative (0 disables bridging)")
    if args.max_keypoints < 0:
        raise ValueError("--max-keypoints must be non-negative (0 keeps every corner)")


def choose_device(request: str) -> torch.device:
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(request)


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    recipe = DetectorRecipe()
    cfg = make_detector_config(recipe)
    sequence = detector.resolve_kinect_sequence(args.capture, args.kinect_seq)
    checkpoint = ensure_checkpoint(args.checkpoint)
    normal_estimator = detector.load_normal_estimator(args.normal_module)
    log(f"[SOURCE] {sequence}")
    log(f"[DEVICE] {device}; PyTorch {torch.__version__}; Open3D {o3d.__version__}")
    log(f"[MODEL] BUFFER-X {BUFFERX_COMMIT[:12]}, checkpoint SHA-256 verified")

    source_cloud, _ = load_frame_cloud(
        sequence,
        args.source_frame,
        recipe,
        args.kinect_fx,
        args.kinect_fy,
        args.kinect_cx,
        args.kinect_cy,
    )
    target_cloud, _ = load_frame_cloud(
        sequence,
        args.target_frame,
        recipe,
        args.kinect_fx,
        args.kinect_fy,
        args.kinect_cx,
        args.kinect_cy,
    )
    source_keypoints, source_scores = detect_keypoints(
        sequence,
        args.source_frame,
        source_cloud,
        cfg,
        recipe,
        args.fps_count,
        normal_estimator,
        args.force_redetect,
    )
    target_keypoints, target_scores = detect_keypoints(
        sequence,
        args.target_frame,
        target_cloud,
        cfg,
        recipe,
        args.fps_count,
        normal_estimator,
        args.force_redetect,
    )
    source_keypoints, source_scores = select_keypoints(
        source_keypoints, source_scores, args.max_keypoints
    )
    target_keypoints, target_scores = select_keypoints(
        target_keypoints, target_scores, args.max_keypoints
    )
    log(f"[KEYPOINTS] source={len(source_keypoints)}, target={len(target_keypoints)}")

    model = load_minispinnet(checkpoint, device)
    started = time.perf_counter()
    source_descriptors, source_valid = describe_keypoints(
        model,
        source_cloud,
        source_keypoints,
        args.descriptor_radius_m,
        args.descriptor_batch_size,
        device,
        args.seed + args.source_frame,
    )
    target_descriptors, target_valid = describe_keypoints(
        model,
        target_cloud,
        target_keypoints,
        args.descriptor_radius_m,
        args.descriptor_batch_size,
        device,
        args.seed + args.target_frame,
    )
    source_keypoints = source_keypoints[source_valid]
    source_scores = source_scores[source_valid]
    target_keypoints = target_keypoints[target_valid]
    target_scores = target_scores[target_valid]
    descriptor_time = time.perf_counter() - started
    source_match_ids, target_match_ids, descriptor_distances = mutual_descriptor_matches(
        source_descriptors,
        target_descriptors,
        args.ratio_threshold,
        args.max_descriptor_distance,
    )
    log(
        f"[DESC] source={len(source_descriptors)}, target={len(target_descriptors)}, "
        f"mutual matches={len(source_match_ids)}, time={descriptor_time:.2f}s"
    )

    source_bundle = FrameBundle(
        args.source_frame,
        source_cloud,
        source_keypoints,
        source_scores,
        source_descriptors,
    )
    target_bundle = FrameBundle(
        args.target_frame,
        target_cloud,
        target_keypoints,
        target_scores,
        target_descriptors,
    )
    bundles = {
        args.source_frame: source_bundle,
        args.target_frame: target_bundle,
    }

    try:
        registration = run_registration(
            source_keypoints,
            target_keypoints,
            source_match_ids,
            target_match_ids,
            source_cloud,
            target_cloud,
            args.ransac_distance_m,
            args.ransac_iterations,
            args.icp_distance_m,
            args.icp_iterations,
            not args.no_icp,
            args.seed,
        )
        registration["mode"] = "direct"
        registration["direct_failure"] = None
    except RuntimeError as direct_error:
        frame_gap = abs(args.target_frame - args.source_frame)
        if args.bridge_step == 0 or frame_gap < 2:
            raise
        # A bridge must contain at least one actual intermediate frame.  For a
        # gap smaller than the requested stride, split it near the midpoint.
        effective_step = min(args.bridge_step, max(1, frame_gap // 2))
        frames = bridge_frame_indices(
            args.source_frame,
            args.target_frame,
            effective_step,
            args.bridge_max_legs,
        )
        log(f"[DIRECT] {direct_error}")

        def get_bridge_bundle(frame_index: int) -> FrameBundle:
            log(f"[BRIDGE] preparing frame {frame_index}")
            return prepare_frame_bundle(
                sequence,
                frame_index,
                recipe,
                cfg,
                normal_estimator,
                model,
                device,
                fps_count=args.bridge_fps_count,
                max_keypoints=args.max_keypoints,
                descriptor_radius_m=args.descriptor_radius_m,
                descriptor_batch_size=args.descriptor_batch_size,
                descriptor_seed=args.seed + frame_index,
                force_redetect=args.force_redetect,
                fx=args.kinect_fx,
                fy=args.kinect_fy,
                cx=args.kinect_cx,
                cy=args.kinect_cy,
            )

        registration = run_bridge_registration(
            frames,
            bundles,
            get_bridge_bundle,
            source_match_ids,
            target_match_ids,
            ratio_threshold=args.ratio_threshold,
            max_descriptor_distance=args.max_descriptor_distance,
            ransac_distance=args.ransac_distance_m,
            ransac_iterations=args.ransac_iterations,
            icp_distance=args.icp_distance_m,
            icp_iterations=args.icp_iterations,
            use_icp=not args.no_icp,
            seed=args.seed,
        )
        registration["direct_failure"] = str(direct_error)

    transform = registration["transform"]
    inlier_count = int(np.count_nonzero(registration["inliers"]))
    inlier_rmse = (
        None
        if inlier_count == 0
        else float(
            np.sqrt(np.mean(registration["residuals"][registration["inliers"]] ** 2))
        )
    )
    if registration["mode"] == "bridge":
        support_rmse = registration["bridge_support_rmse_m"]
        support_rmse_text = "n/a" if support_rmse is None else f"{1000.0 * support_rmse:.2f} mm"
        log(
            f"[REGISTER bridge] legs={len(registration['bridge_legs'])}, "
            f"support inliers={registration['bridge_support_inliers']}/"
            f"{registration['bridge_support_matches']}, support RMSE={support_rmse_text}, "
            f"RANSAC={registration['ransac_time_s']:.2f}s, "
            f"ICP={registration['icp_time_s']:.2f}s"
        )
        log(
            f"[ENDPOINT descriptor check] inliers={inlier_count}/{len(source_match_ids)}"
            + ("" if inlier_rmse is None else f", RMSE={1000.0 * inlier_rmse:.2f} mm")
        )
    else:
        log(
            f"[REGISTER direct] inlier matches={inlier_count}/{len(source_match_ids)}, "
            f"RMSE={1000.0 * inlier_rmse:.2f} mm, "
            f"RANSAC={registration['ransac_time_s']:.2f}s, "
            f"ICP={registration['icp_time_s']:.2f}s"
        )
    log(
        f"[DENSE @ {1000.0 * registration['dense_evaluation_distance_m']:.0f} mm] "
        f"fitness={registration['final_dense_fitness']:.4f}, "
        f"RMSE={1000.0 * registration['final_dense_rmse_m']:.2f} mm"
    )
    log("[TRANSFORM source -> target]\n" + np.array2string(transform, precision=8))

    gt_transform = None
    gt_metrics = None
    if not args.no_gt_evaluation:
        source_pose = read_pose(sequence, args.source_frame)
        target_pose = read_pose(sequence, args.target_frame)
        if source_pose is not None and target_pose is not None:
            gt_transform = np.linalg.inv(target_pose) @ source_pose
            camera_delta_translation, camera_delta_rotation = transform_error(
                gt_transform, np.eye(4)
            )
            translation_error, rotation_error = transform_error(transform, gt_transform)
            gt_metrics = {
                "camera_delta_translation_m": camera_delta_translation,
                "camera_delta_rotation_deg": camera_delta_rotation,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
            }
            log(
                f"[GT camera delta] translation={camera_delta_translation:.4f} m, "
                f"rotation={camera_delta_rotation:.3f} deg"
            )
            log(
                f"[GT registration error] translation={1000.0 * translation_error:.2f} mm, "
                f"rotation={rotation_error:.3f} deg (evaluation only)"
            )

    if args.output_dir is None:
        output_dir = (
            SCRIPT_DIR
            / "registration_results"
            / f"{sequence.stem}_{args.source_frame:06d}_to_{args.target_frame:06d}"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    payload: dict[str, Any] = {
        "source_sequence": str(sequence),
        "source_frame": args.source_frame,
        "target_frame": args.target_frame,
        "transform_convention": "target_xyz = R @ source_xyz + t; coordinates and t in meters",
        "transform_source_to_target": transform.tolist(),
        "ransac_transform_source_to_target": registration["ransac_transform"].tolist(),
        "ground_truth_source_to_target": None if gt_transform is None else gt_transform.tolist(),
        "ground_truth_metrics": gt_metrics,
        "detector": {
            "build": detector.BUILD,
            "recipe": asdict(recipe),
            "fps_count": args.fps_count,
            "bridge_fps_count": args.bridge_fps_count,
            "source_keypoints": len(source_keypoints),
            "target_keypoints": len(target_keypoints),
        },
        "descriptor": {
            "name": "BUFFER-X Mini-SpinNet (pure PyTorch/KD-tree query port)",
            "upstream_commit": BUFFERX_COMMIT,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "dimension": int(source_descriptors.shape[1]),
            "patch_radius_m": args.descriptor_radius_m,
            "device": str(device),
            "elapsed_s": descriptor_time,
        },
        "matching": {
            "mutual_count": len(source_match_ids),
            "ratio_threshold": args.ratio_threshold,
            "max_descriptor_distance": args.max_descriptor_distance,
            "descriptor_distance_mean": float(np.mean(descriptor_distances)),
        },
        "registration": {
            "mode": registration["mode"],
            "method": (
                "temporal_bridge_minispinnet_ransac_icp"
                if registration["mode"] == "bridge"
                else "direct_minispinnet_ransac_icp"
            ),
            "direct_failure": registration.get("direct_failure"),
            "ransac_distance_m": args.ransac_distance_m,
            "ransac_iterations": args.ransac_iterations,
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_count / len(source_match_ids),
            "inlier_rmse_m": inlier_rmse,
            "bridge_step": args.bridge_step,
            "bridge_frames": registration.get("bridge_frames"),
            "bridge_legs": registration.get("bridge_legs"),
            "bridge_support_matches": registration.get("bridge_support_matches"),
            "bridge_support_inliers": registration.get("bridge_support_inliers"),
            "bridge_support_rmse_m": registration.get("bridge_support_rmse_m"),
            "ransac_time_s": registration["ransac_time_s"],
            "icp_enabled": not args.no_icp,
            "icp_rejected": registration["icp_rejected"],
            "icp_distance_m": args.icp_distance_m,
            "icp_time_s": registration["icp_time_s"],
            "icp_fitness": None if registration["icp"] is None else registration["icp"].fitness,
            "icp_inlier_rmse_m": None
            if registration["icp"] is None
            else registration["icp"].inlier_rmse,
            "dense_evaluation_distance_m": registration["dense_evaluation_distance_m"],
            "coarse_dense_fitness": registration["coarse_dense_fitness"],
            "coarse_dense_rmse_m": registration["coarse_dense_rmse_m"],
            "final_dense_fitness": registration["final_dense_fitness"],
            "final_dense_rmse_m": registration["final_dense_rmse_m"],
        },
    }
    paths = save_outputs(
        output_dir,
        source_cloud,
        target_cloud,
        source_keypoints,
        target_keypoints,
        source_match_ids,
        target_match_ids,
        descriptor_distances,
        registration,
        payload,
        args.seed,
    )
    for name, path in paths.items():
        log(f"[OUTPUT {name}] {path}")

    if not args.no_gui:
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            log("[GUI] no display detected; inspect registration_preview.png instead")
        else:
            show_interactive(
                source_cloud,
                target_cloud,
                source_keypoints,
                target_keypoints,
                source_match_ids,
                target_match_ids,
                registration["inliers"],
                transform,
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\nCancelled.")
        raise SystemExit(130)
    except Exception as error:
        log(f"\nERROR: {type(error).__name__}: {error}")
        if os.environ.get("K7_REG_DEBUG") == "1":
            raise
        raise SystemExit(1)
