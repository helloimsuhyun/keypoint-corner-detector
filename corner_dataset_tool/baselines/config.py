"""Dataset presets and reproducible author settings for external baselines."""
from dataclasses import dataclass

import numpy as np


BASELINE_METHODS = ("orthogonal", "ahmed", "harris3d")


@dataclass(frozen=True)
class AhmedConfig:
    voxel_mm: float = 10.
    edge_neighbors: int = 400
    edge_threshold: float = 4.5
    thinning_neighbors: int = 150
    corner_neighbors: int = 20
    pca_neighbors: int = 10
    span_mm: float = 5.
    angle_min_deg: float = 50.
    angle_max_deg: float = 150.

    def __post_init__(self):
        for name in ("voxel_mm", "edge_threshold", "span_mm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Ahmed {name} must be finite and positive")
        for name in ("edge_neighbors", "thinning_neighbors", "corner_neighbors"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f"Ahmed {name} must be an integer >=2")
        if not isinstance(self.pca_neighbors, (int, np.integer)) or self.pca_neighbors < 3:
            raise ValueError("Ahmed pca_neighbors must be an integer >=3")
        if not 0 <= self.angle_min_deg < self.angle_max_deg <= 180:
            raise ValueError("Ahmed angles must satisfy 0 <= min < max <= 180 degrees")


AUTHOR_AHMED = AhmedConfig()
DATASET_AHMED = AhmedConfig(voxel_mm=5., edge_neighbors=100, edge_threshold=2.,
                            thinning_neighbors=30, corner_neighbors=20)
# shortcut: selected on scene_0002 ROI; validate on held-out scenes before benchmarking.
SCENE2_AHMED = AhmedConfig(voxel_mm=5., edge_neighbors=100, edge_threshold=2.,
                          thinning_neighbors=5, corner_neighbors=27, pca_neighbors=5,
                          span_mm=8., angle_min_deg=53.3, angle_max_deg=116.)
AUTHOR_SOMMER = dict(normal_radius_mm=20., dmin_mm=100., dmax_mm=1000., min_votes=5, sampling=10,
                     sommer_voxel_mm=0.)
DATASET_SOMMER = dict(normal_radius_mm=10., dmin_mm=5., dmax_mm=100., min_votes=3, sampling=5,
                      sommer_voxel_mm=0.)
