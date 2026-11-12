"""Numerical/ordering contracts for bounded neighborhood reuse and lazy Q10."""

from math import ceil
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from corner_detector import edge_suppression as dense, refiner


class DetectorOptimizationTest(unittest.TestCase):
    def test_batched_neighbors_preserve_scalar_order_and_radius_boundaries(self):
        rng = np.random.default_rng(21)
        points = np.vstack((rng.normal(size=(145, 3)), np.eye(3), -np.eye(3), np.zeros((2, 3))))
        for leaf in (1, 16, 80):
            tree = cKDTree(points, leafsize=leaf)
            for batch in dense._neighborhood_batches(points, tree, np.arange(len(points)), 3.):
                self.assertLessEqual(len(batch), 64)
                for index, ids, distance2 in batch:
                    for radius in (.25, 1., 2.):
                        expected = np.asarray(tree.query_ball_point(points[index], radius), dtype=np.int64)
                        delta = points[expected] - points[index]
                        d2 = np.einsum('ij,ij->i', delta, delta)
                        for strict in (True, False):
                            actual = ids[distance2 < radius**2] if strict else ids[distance2 <= radius**2]
                            wanted = expected[d2 < radius**2] if strict else expected[d2 <= radius**2]
                            np.testing.assert_array_equal(actual, wanted)

    def test_dense_fixed_numerical_regression(self):
        points = np.array([[0.,0,0], [1.,0,0], [2.,0,0], [3.,0,0],
                           [0,1.,0], [0,2.,0], [0,3.,0], [1.,1.,0]])
        result = dense.dense_ced_weighted_pca_scores(points, 1.6, 4., 2)
        # Recorded from the independent three-pass implementation before neighborhood reuse.
        np.testing.assert_allclose(result['fusion_score'], [
            .14792906126838953, .09264084727030343, .08058520111115094, .07884430630264133,
            .09264084727030342, .08058520111115094, .07884430630264133, .09501201441342522], rtol=0, atol=1e-15)
        np.testing.assert_allclose(result['ced_score'], [
            .4419417382415922, .27950849718747367, .2209708691207961, .3125,
            .27950849718747367, .2209708691207961, .3125, .2946278254943948], rtol=0, atol=1e-15)

    def test_dense_keeps_validation_of_derived_ced_before_shared_support(self):
        with (patch.object(dense, 'ced_neighborhood_response', return_value=(np.zeros(3), 0., np.nan)),
              self.assertRaisesRegex(ValueError, 'ced_score')):
            dense.dense_ced_weighted_pca_scores(np.zeros((3, 3)), 1., 1.5, 2)

    def test_dense_uses_two_bounded_search_passes_and_reuses_bilateral_support(self):
        points = np.random.default_rng(3).normal(size=(145, 3))
        class Tree:
            def __init__(self):
                self.tree, self.sizes = cKDTree(points), []
            def query_ball_point(self, queries, radius, **kwargs):
                self.sizes.append(len(queries) if np.ndim(queries) == 2 else 1)
                return self.tree.query_ball_point(queries, radius, **kwargs)
        for pca_radius, bilateral_radius in ((2., 1.), (1., 2.)):
            tree = Tree()
            result = dense.dense_ced_weighted_pca_scores(points, 1.5, pca_radius, 3, point_tree=tree,
                                                        bilateral_radius_mm=bilateral_radius)
            expected_calls = ceil(len(points)/64) + ceil(np.count_nonzero(result['pca_query_mask'])/64)
            self.assertEqual(len(tree.sizes), expected_calls)
            self.assertLessEqual(max(tree.sizes), 64)
            wanted = dense.bilateral_filter_ced(points, result['ced_score'], result['ced_valid_mask'],
                                                bilateral_radius, .75, .05)
            np.testing.assert_array_equal(result['bilateral_ced_score'], wanted['score'])
            np.testing.assert_array_equal(result['bilateral_neighbor_count'], wanted['neighbor_count'])

    def test_shared_neighborhood_requeries_ambiguous_floating_point_boundaries(self):
        rng = np.random.default_rng(17)
        directions = rng.normal(size=(200, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        for radius in (.1, 1., 45., 67.5):
            points = np.vstack((np.zeros(3), directions*radius, directions*np.nextafter(radius, np.inf),
                                directions*np.nextafter(radius, 0)))
            tree = cKDTree(points)
            ids = np.asarray(tree.query_ball_point(points[0], 2*radius), dtype=np.int64)
            distance2 = np.einsum('ij,ij->i', points[ids], points[ids])
            actual, d2 = dense._radius_neighborhood(points, tree, 0, ids, distance2, radius, 2*radius)
            expected = np.asarray(tree.query_ball_point(points[0], radius), dtype=np.int64)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(d2, np.einsum('ij,ij->i', points[expected], points[expected]))

    def test_support_gate_skips_quantile_without_poisoning_cache(self):
        points = np.vstack((np.zeros(3), *(np.eye(3)*r for r in range(1, 5))))
        for cc in (False, True):
            r = refiner.Q10PositionRefiner(points, 10., 1., cc_factor=1.1, cc_enabled=cc)
            with (patch.object(refiner, 'mean_direction_q10', wraps=refiner.mean_direction_q10) as quantile,
                  patch.object(refiner, '_components', wraps=refiner._components) as graph):
                skipped = r.evaluate(0, minimum_support=20)
                self.assertIsNone(skipped[0])
                self.assertEqual(skipped[1], 13)
                quantile.assert_not_called()
                self.assertNotIn(0, r.cache)
                self.assertIn(0, r.support_cache)
                result = r.evaluate(0)
                quantile.assert_called_once()
                self.assertEqual(graph.call_count, int(cc))
            self.assertAlmostEqual(result[0], 1/np.sqrt(3))
            np.testing.assert_allclose(result[2], np.ones(3)/np.sqrt(3))
            self.assertIs(r.evaluate(0, minimum_support=100), r.cache[0])

    def test_refined_positions_and_traces_match_exhaustive_quality_evaluation(self):
        class Exhaustive(refiner.Q10PositionRefiner):
            def evaluate(self, index, **_kwargs):
                return super().evaluate(index)
        x, y = np.meshgrid(np.arange(8.), np.arange(8.))
        zero = np.zeros(x.size)
        points = np.unique(np.vstack((np.column_stack((x.ravel(), y.ravel(), zero)),
                                      np.column_stack((x.ravel(), zero, y.ravel())))), axis=0)
        for cc in (False, True):
            for mode in ('unit_mean', 'unit_median', 'centroid'):
                options = dict(cc_enabled=cc, direction_mode=mode)
                actual = refiner.Q10PositionRefiner(points, 4., 2., **options)
                expected = Exhaustive(points, 4., 2., **options)
                for index in (45, 0, 97, 12):
                    with self.subTest(cc=cc, mode=mode, index=index):
                        np.testing.assert_equal(actual.refine(index), expected.refine(index))


if __name__ == '__main__':
    unittest.main()
