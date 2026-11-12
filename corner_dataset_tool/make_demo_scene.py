"""Generate a reproducible, camera-free demo scene to exercise annotation GUI."""
import argparse
from pathlib import Path

import numpy as np

from dataset_io import create_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dataset/demo_corner"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    grid = np.linspace(0, 95, 70)
    u, v = np.meshgrid(grid, grid)
    uv = np.stack((u.ravel(), v.ravel()), axis=1)
    # Three orthogonal faces meeting at (0, 0, 550), with 0.35mm noise.
    faces = [
        np.column_stack((np.zeros(len(uv)), uv[:, 0], 550 + uv[:, 1])),
        np.column_stack((uv[:, 0], np.zeros(len(uv)), 550 + uv[:, 1])),
        np.column_stack((uv[:, 0], uv[:, 1], np.full(len(uv), 550))),
    ]
    faces = [p + rng.normal(0, 0.35, p.shape) for p in faces]
    backdrop = np.column_stack((rng.uniform(-180, 180, 7000), rng.uniform(-130, 180, 7000),
                                rng.normal(760, 1.0, 7000)))
    xyz = np.concatenate([*faces, backdrop]).astype(np.float32)
    rgb = np.concatenate([
        np.tile([93, 174, 224], (len(faces[0]), 1)),
        np.tile([128, 196, 231], (len(faces[1]), 1)),
        np.tile([178, 220, 242], (len(faces[2]), 1)),
        np.tile([100, 103, 112], (len(backdrop), 1)),
    ]).astype(np.uint8)
    root = create_scene(args.output, xyz, rgb, metadata={"source": "synthetic_demo"})
    print(f"Demo scene: {root} | {len(xyz):,} points")


if __name__ == "__main__":
    main()
