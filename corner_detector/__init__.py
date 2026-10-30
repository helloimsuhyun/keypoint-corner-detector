"""Boundary-weighted point-cloud corner and finite-edge detector."""
from .boundary_support import boundary_weights
from .corner_2d import estimate_boundary_uv
from .candidate_extraction import extract_corner_candidates
from .extremity import ExtremityConfig, run_extremity_refinement, refine_ced_candidates
from .edge_suppression import (
    ced_weighted_pca,
    dense_ced_weighted_pca_scores,
    saturate_pca_ratio,
    weighted_pca_query_debug,
)
from .line_refinement_2d import LineRefinement2DConfig, refine_hough_lines_2d
from .plane_edge_graph_pipeline import PlaneEdgeGraphConfig, analyze_plane_edge_graph
from .refiner import HoughEdgeRefinerConfig, refine_hough_edge_graph
from .cornervins_plane_extraction import (
    CornerVINSPlaneConfig,
    PlaneExtractionResult,
    PlanePatch,
    extract_planes,
)

__all__ = [
    "boundary_weights", "estimate_boundary_uv",
    "ced_weighted_pca",
    "dense_ced_weighted_pca_scores",
    "saturate_pca_ratio",
    "weighted_pca_query_debug",
    "extract_corner_candidates",
    "ExtremityConfig", "run_extremity_refinement", "refine_ced_candidates",
    "PlaneEdgeGraphConfig", "analyze_plane_edge_graph",
    "HoughEdgeRefinerConfig", "refine_hough_edge_graph",
    "LineRefinement2DConfig", "refine_hough_lines_2d",
    "CornerVINSPlaneConfig",
    "PlaneExtractionResult",
    "PlanePatch",
    "extract_planes",
]
