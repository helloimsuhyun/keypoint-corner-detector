"""Regression guards for future AI-assisted changes."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "corner_detector"
NUMERIC = (
    "corner_2d.py", "boundary_support.py", "edge_suppression.py",
    "candidate_extraction.py", "cornervins_plane_extraction.py",
    "plane_edge_graph_pipeline.py", "line_refinement_2d.py", "refiner.py",
    "extremity.py",
    "extremity_voting.py",
    "guided_ransac.py", "validator_fine.py",
)


class ArchitectureTest(unittest.TestCase):
    def test_numerical_modules_dont_import_renderers(self):
        for filename in NUMERIC:
            tree = ast.parse((PACKAGE / filename).read_text(encoding="utf8"))
            modules = [
                n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
            ] + [
                a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names
            ]
            for module in modules:
                self.assertNotIn("visualization", module, (filename, module))
                self.assertNotIn("matplotlib", module, (filename, module))
                self.assertNotIn("tkinter", module, (filename, module))

    def test_no_legacy_main_and_visualization_scattered(self):
        for filename in ("main.py", "plots.py", "view3d.py", "diagnostics.py", "query_ui.py"):
            self.assertFalse((PACKAGE / filename).exists(), filename)
        for filename in (
            "plane_verification.py",
            "FINITE_SEGMENT_NMS_CHANGELOG.md",
            "PLANE_EDGE_DEBUG_VISUAL_ONLY.md",
            "VOTE_ONLY_REMOVAL_CHANGELOG.md",
        ):
            self.assertFalse((PACKAGE / filename).exists(), filename)
        for filename in ("manager.py", "plots.py", "view3d.py", "query_ui.py"):
            self.assertTrue((PACKAGE / "visualization" / filename).exists(), filename)

        legacy_terms = ("legacy", "force_field", "localize_corner_iterative")
        for path in PACKAGE.rglob("*.py"):
            text = path.read_text(encoding="utf8").lower()
            for term in legacy_terms:
                self.assertNotIn(term, text, (path, term))

    def test_cli_help(self):
        run = subprocess.run([sys.executable, "-m", "corner_detector", "--help"],
                             cwd=ROOT, text=True, capture_output=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--source", run.stdout)
        self.assertIn("--mode", run.stdout)
        self.assertIn("{fine,detector,ball,extremity256,validator}", run.stdout)
        self.assertIn("--ball-radii-x", run.stdout)
        self.assertNotIn("{query,heatmap,fps}", run.stdout)
        self.assertIn("--centroid-radius-x", run.stdout)
        self.assertIn("--pca-radius-factor", run.stdout)
        self.assertIn("--pca-ratio-saturation-tau", run.stdout)
        self.assertIn("--fusion-score-threshold", run.stdout)
        self.assertIn("--nms-radius-mm", run.stdout)
        self.assertIn("fine 1/6: query-core CornerVINS", run.stdout)
        self.assertNotIn("--plane-extraction-mode", run.stdout)
        self.assertNotIn("--plane-cluster-analysis", run.stdout)
        self.assertNotIn("--plane-ref-sampling-mode", run.stdout)
        self.assertNotIn("--cornervins-sah-comparison", run.stdout)
        self.assertNotIn("--plane-line-analysis", run.stdout)
        self.assertIn("fine 0/6: query geometry", run.stdout)
        self.assertIn("fine 2/6: padded projection and continuous boundary", run.stdout)
        self.assertIn("fine 3/6: weighted Hough and accumulator NMS", run.stdout)
        self.assertIn("fine 4/6: continuous 2-D line refinement and final NMS", run.stdout)
        self.assertIn("fine 5/6: 3-D line reconstruction", run.stdout)
        self.assertIn("fine 6/6: graph-local edge refinement and corners", run.stdout)
        self.assertIn("detector: dense CED -> fusion -> NMS", run.stdout)
        for removed in (
            "--candidate-score-threshold", "--candidate-mode",
            "--ced-fusion-sigma", "--ced-filter-threshold",
            "--pca-support-ced-threshold", "--pca-edge-ratio-threshold",
            "--corner-nms-radius-mm", "--plane-edge-scope",
            "--no-plane-verifier", "--plane-roi-radius-mm",
            "--plane-noise-threshold-mm", "--plane-harris-search-radius-x",
            "--multiscale-harris", "--sah-graph-analysis", "--edge3d",
            "--normal-module", "--gaussian-scale-count",
        ):
            self.assertNotIn(removed, run.stdout)
        self.assertNotIn("--fusion ", run.stdout)
        self.assertNotIn("--corner-method", run.stdout)

    def test_whole_cloud_detector_has_no_removed_experiment_symbols(self):
        removed = (
            "gaussian_ced_weight", "ced_soft_weight", "hamacher_product",
            "filtered_ced_support_weights", "pca_edge_rejection_masks",
            "candidate_mode", "fusion_mode", "ced_filter_threshold",
            "pca_support_ced_threshold", "pca_edge_ratio_threshold",
            "candidate_score_threshold",
        )
        files = (
            PACKAGE / "settings.py",
            PACKAGE / "edge_suppression.py",
            PACKAGE / "candidate_extraction.py",
            PACKAGE / "dense_heatmap.py",
            PACKAGE / "runner.py",
            PACKAGE / "visualization" / "view3d.py",
        )
        for path in files:
            text = path.read_text(encoding="utf8")
            for symbol in removed:
                self.assertNotIn(symbol, text, (path, symbol))


if __name__ == "__main__":
    unittest.main()
