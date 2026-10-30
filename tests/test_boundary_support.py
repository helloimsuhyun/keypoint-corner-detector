"""Numerical contract for Gaussian boundary support used by 2-D edges."""

import unittest

import numpy as np

from corner_detector.boundary_support import boundary_weights


class BoundarySupportTest(unittest.TestCase):
    def test_gaussian_formula_and_invalid_samples(self):
        distance = np.array([0.0, 2.0, 4.0, np.nan, -1.0])
        valid = np.array([True, True, True, True, True])
        actual = boundary_weights(distance, valid, sigma_mm=2.0)
        expected = np.array([1.0, np.exp(-0.5), np.exp(-2.0), 0.0, 0.0])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_direction_mask_and_input_validation(self):
        np.testing.assert_array_equal(
            boundary_weights([0.0, 1.0], [False, True], 1.0),
            [0.0, np.exp(-0.5)],
        )
        with self.assertRaises(ValueError):
            boundary_weights([0.0], [True, False], 1.0)
        with self.assertRaises(ValueError):
            boundary_weights([0.0], [True], 0.0)


if __name__ == "__main__":
    unittest.main()
