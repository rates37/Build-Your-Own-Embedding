"""
Similarity and distance metrics for comparing neural responses

This module provides functions for computing various similarity and distance
metrics between response arrays, commonly used in RSA

Functions:
    inverse_correlation: Computes inverse of the Pearson correlation coefficient
    manhattan_distance: Computes L1 distance
    euclidean_distance: Computes L2 distance
    cosine_similarity: Computes cosine similarity
"""

import numpy as np
import numpy.typing as npt


def inverse_correlation(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    """
    Computes the inverse correlation between two numpy arrays.

    This function calculates the inverse of the Pearson correlation coefficient
    between two input arrays, `x1` and `x2`, and returns the result cast to the
    specified `dtype`.

    Args:
        x1: First input array for comparison.
        x2: Second input array for comparison.
        dtype: The data type for the output. Defaults to np.float64.

    Returns:
        The inverse correlation coefficient between `x1` and `x2`, cast to the
        specified `dtype`.
    """
    return dtype(1 - np.corrcoef(x1.flatten(), x2.flatten())[0, 1])


def manhattan_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    """
    Computes the Manhattan distance between two numpy arrays.

    This function calculates the Manhattan distance (L1 norm), between two input
    arrays, `x1` and `x2`, and returns the result cast to the specified `dtype`.

    Args:
        x1: First input array for comparison.
        x2: Second input array for comparison.
        dtype: The data type for the output. Defaults to np.float64.

    Returns:
        The Manhattan distance between `x1` and `x2`, cast to the specified `dtype`.
    """
    return dtype(np.sum(np.abs(x1.flatten() - x2.flatten())))


def euclidean_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    """
    Computes the Euclidean distance between two numpy arrays.

    This function calculates the Euclidean distance (L2 norm), between two input
    arrays, `x1` and `x2`, and returns the result cast to the specified `dtype`.

    Args:
        x1: First input array for comparison.
        x2: Second input array for comparison.
        dtype: The data type for the output. Defaults to np.float64.

    Returns:
        The Euclidean distance between `x1` and `x2`, cast to the specified `dtype`.
    """
    return dtype(np.sqrt(np.sum((x1.flatten() - x2.flatten()) ** 2)))


def cosine_similarity(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    """
    Computes the cosine similarity between two numpy arrays.

    This function calculates the cosine similarity between two input
    arrays, `x1` and `x2`, and returns the result cast to the specified `dtype`.

    The cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical direction).

    Args:
        x1: First input array for comparison.
        x2: Second input array for comparison.
        dtype: The data type for the output. Defaults to np.float64.

    Returns:
        The cosine similarity between `x1` and `x2`, cast to the specified `dtype`.
    """
    x1_flat = x1.flatten()
    x2_flat = x2.flatten()
    return dtype(np.dot(x1_flat, x2_flat) / (np.linalg.norm(x1_flat) * np.linalg.norm(x2_flat)))
