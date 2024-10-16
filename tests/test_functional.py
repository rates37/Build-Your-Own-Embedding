import unittest
import numpy as np
from BuildYourOwnEmbedding.functional import (
    inverse_correlation,
    manhattan_distance,
    euclidian_distance,
    cosine_similarity,
    mutual_information,
    fisher_information,
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

    # mutual_information tests:

    def test_mutual_information_uniform_response(self):
        stimulus = np.array([0.1, 0.2, 0.3, 0.4])
        response = np.array([0.25, 0.25, 0.25, 0.25])  # Uniform distribution
        result = mutual_information(stimulus, response)
        self.assertAlmostEqual(
            result, 0, msg="Mutual information should be 0 for a uniform response"
        )

    def test_mutual_information_non_uniform_response(self):
        stimulus = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        response = np.array([0.0, 0.25, 0.5, 0.25, 0])
        result = mutual_information(stimulus, response)
        expectedResult = 1.0549201679
        self.assertAlmostEqual(result, expectedResult)

    def test_mutual_information_invalid_input(self):
        # Test mutual information with differently shaped inputs
        stimulus = np.array([0.1, 0.2])  # Does not sum to 1
        response = np.array([0.4, 0.5, 0.1])
        with self.assertRaises(
            ValueError,
            msg="Mutual information should raise ValueError if inputs are not same shape",
        ):
            mutual_information(stimulus, response)

    # fisher information tests:

    def test_fisher_information_1d(self):
        # Test Fisher Information with a 1D set of tuning curves
        curves = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
        result = fisher_information(curves)
        expectedResult = np.array([0.02, 0.02, 0.02, 0.02])
        self.assertEqual(
            result.shape,
            (4,),
            msg="Fisher information result shape should match the response dimension",
        )
        self.assertTrue(
            np.all(result >= 0), msg="Fisher information should be non-negative"
        )
        self.assertTrue(np.allclose(result, expectedResult, atol=EPSILON))

    def test_fisher_information_2d(self):
        # Test Fisher Information with a 2D tuning curve
        curves = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.2, 0.3], [0.4, 0.5]]])
        result = fisher_information(curves)
        expectedResult = np.array([[0.1, 0.1], [0.1, 0.1]])
        self.assertEqual(
            result.shape,
            (2, 2),
            msg="Fisher information shape should match the input spatial dimensions",
        )
        self.assertTrue(
            np.all(result >= 0), msg="Fisher information should be non-negative"
        )
        self.assertTrue(np.allclose(result, expectedResult, atol=EPSILON))

    def test_fisher_information_3d(self):
        # Test Fisher Information with a 3D tuning curve
        curves = np.random.rand(3, 4, 4, 4)  # 3 tuning curves over a 4x4x4 space
        result = fisher_information(curves)
        self.assertEqual(
            result.shape,
            (4, 4, 4),
            msg="Fisher information shape should match the input spatial dimensions",
        )
        self.assertTrue(
            np.all(result >= 0), msg="Fisher information should be non-negative"
        )
        self.assertTrue(
            np.isfinite(result).all(),
            msg="Fisher information should return finite values",
        )

    def test_fisher_information_invalid_input(self):
        # Test Fisher Information with invalid input (e.g., fewer than 2 curves)
        curves = np.array([0.1, 0.2, 0.3])  # Only 1 curve
        with self.assertRaises(
            ValueError,
            msg="Fisher information should raise ValueError for input with dimensionality lower than 2",
        ):
            fisher_information(curves)

    def test_fisher_information_high_values(self):
        # Test Fisher Information with high tuning curve values
        curves = np.array(
            [[1000, 2000, 3000], [1500, 2500, 3500]]
        )  # 2 tuning curves with large values
        result = fisher_information(curves)
        expectedResult = np.array([2000000.0, 2000000.0, 2000000.0])
        self.assertEqual(
            result.shape,
            (3,),
            msg="Fisher information result shape should match the response dimension",
        )
        self.assertTrue(
            np.all(result >= 0), msg="Fisher information should be non-negative"
        )
        self.assertTrue(np.allclose(result, expectedResult, atol=EPSILON))


if __name__ == "__main__":
    unittest.main()
