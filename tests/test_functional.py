import unittest
import numpy as np
from BuildYourOwnEmbedding.functional import (
    inverse_correlation,
    manhattan_distance,
    euclidian_distance,
    cosine_similarity,
)

EPSILON = 1e-6


class TestFunctional(unittest.TestCase):

    # Inverse Correlation Tests
    def test_inverse_correlation_identical_vectors(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([1, 2, 3])
        result = inverse_correlation(x1, x2)
        self.assertAlmostEqual(
            result,
            0,
            msg=f"Inverse correlation should be 0 for identical vectors. Inputs: {x1=}, {x2=}. Expected: 0, Got: {result}",
        )

    def test_inverse_correlation_opposite_vectors(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([-1, -2, -3])
        result = inverse_correlation(x1, x2)
        self.assertAlmostEqual(
            result,
            2,
            msg=f"Inverse correlation should be 2 for opposite vectors. Inputs: {x1=}, {x2=}. Expected: 2, Got: {result}",
        )

    def test_inverse_correlation_unrelated_vectors(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])
        result = inverse_correlation(x1, x2)
        self.assertGreaterEqual(
            result,
            0,
            msg=f"Inverse correlation should be greater than or equal to zero for unrelated vectors. Inputs: {x1=}, {x2=}. Expected: 0, Got: {result}",
        )

    def test_inverse_correlation_different_lengths(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([1, 2])
        with self.assertRaises(ValueError):
            inverse_correlation(x1, x2)

    # Manhattan Distance Tests
    def test_manhattan_distance_identical_vectors(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([1, 2, 3])
        result = manhattan_distance(x1, x2)
        self.assertEqual(
            result,
            0,
            msg=f"Manhattan distance should be 0 for identical vectors. Inputs: {x1=}, {x2=}. Expected: 0, Got: {result}",
        )

    def test_manhattan_distance_simple_case(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 5, 6])
        result = manhattan_distance(x1, x2)
        self.assertEqual(
            result,
            9,
            msg=f"Manhattan distance calculation is incorrect. Inputs: {x1=}, {x2=}. Expected: 9, Got: {result}",
        )

    def test_manhattan_distance_with_negative_values(self):
        x1 = np.array([1, -2, 3])
        x2 = np.array([-1, 2, -3])
        result = manhattan_distance(x1, x2)
        self.assertEqual(
            result,
            12,
            msg=f"Manhattan distance should work with negative values. Inputs: {x1=}, {x2=}. Expected: 12, Got: {result}",
        )

    def test_manhattan_distance_different_lengths(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([1, 2])
        with self.assertRaises(ValueError):
            manhattan_distance(x1, x2)

    # Euclidean Distance Tests
    def test_euclidean_distance_identical_vectors(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([1, 2, 3])
        result = euclidian_distance(x1, x2)
        self.assertAlmostEqual(
            result,
            0,
            msg=f"Euclidean distance should be 0 for identical vectors. Inputs: {x1=}, {x2=}. Expected: 0, Got: {result}",
        )

    def test_euclidean_distance_simple_case(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([4, 6, 8])
        result = euclidian_distance(x1, x2)
        expected = np.linalg.norm([3, 4, 5])
        self.assertAlmostEqual(
            result,
            expected,
            msg=f"Euclidean distance calculation is incorrect. Inputs: {x1=}, {x2=}. Expected: {expected}, Got: {result}",
        )

    def test_euclidean_distance_with_negative_values(self):
        x1 = np.array([1, -2, 3])
        x2 = np.array([-1, 2, -3])
        result = euclidian_distance(x1, x2)
        expected = np.linalg.norm([-2, 4, -6])
        self.assertAlmostEqual(
            result,
            expected,
            msg=f"Euclidean distance should work with negative values. Inputs: {x1=}, {x2=}. Expected: {expected}, Got: {result}",
        )

    def test_euclidean_distance_different_lengths(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([1, 2])
        with self.assertRaises(ValueError):
            euclidian_distance(x1, x2)

    # Cosine Similarity Tests
    def test_cosine_similarity_identical_vectors(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([1, 2, 3])
        result = cosine_similarity(x1, x2)
        self.assertAlmostEqual(
            result,
            1.0,
            msg=f"Cosine similarity should be 1 for identical vectors. Inputs: {x1=}, {x2=}. Expected: {1.0}, Got: {result}",
        )

    def test_cosine_similarity_opposite_vectors(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([-1, -2, -3])
        result = cosine_similarity(x1, x2)
        self.assertAlmostEqual(
            result,
            -1.0,
            msg=f"Cosine similarity should be -1 for opposite vectors. Inputs: {x1=}, {x2=}. Expected: {-1.0}, Got: {result}",
        )

    def test_cosine_similarity_orthogonal_vectors(self):
        x1 = np.array([1, 0])
        x2 = np.array([0, 1])
        result = cosine_similarity(x1, x2)
        self.assertAlmostEqual(
            result,
            0.0,
            msg=f"Cosine similarity should be 0 for orthogonal vectors. Inputs: {x1=}, {x2=}. Expected: {0.0}, Got: {result}",
        )

    def test_cosine_similarity_different_lengths(self):
        x1 = np.array([1, 2, 3])
        x2 = np.array([1, 2])
        with self.assertRaises(ValueError):
            cosine_similarity(x1, x2)
