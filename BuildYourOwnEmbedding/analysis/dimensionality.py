"""
Dimensionality reduction functions for neural response analysis

This module provides functions for reducing the dimensionality of
neural response data.

Functions:
    mds: Classical Multidimensional Scaling
    pca: Principal Component Analysis
    pca_variance_explained: Compute PCA variance explained ratios
"""

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA


def mds(dissimilarity_matrix: npt.NDArray, n_components: int = 3) -> npt.NDArray:
    """
    Perform classical Multidimensional Scaling (MDS) on a dissimilarity matrix.

    This function implements classical MDS, a technique for visualising the level
    of similarity or dissimilarity between sets of data. Given a square dissimilarity
    matrix, the function computes a low-dimensional embedding of the data by finding
    the eigenvectors and eigenvalues of the matrix, resulting in a set of coordinates
    that best preserve the pairwise distances in the original matrix.

    The algorithm follows these steps:
        1. Constructs a centering matrix `J` and computes the double-centered matrix `G`.
        2. Calculates the eigenvalues and eigenvectors of `G`.
        3. Sorts the eigenvectors based on their corresponding eigenvalues in descending order.
        4. Selects the top `n_components` eigenvectors corresponding to the largest eigenvalues.
        5. Computes the coordinates of the points in the reduced dimensionality space.

    Args:
        dissimilarity_matrix: A square matrix of shape `(n, n)` representing the
            dissimilarities between `n` samples. Each element at position `(i, j)`
            represents the dissimilarity between the `i-th` and `j-th` samples.
        n_components: The number of dimensions for the output embedding. Defaults to 3.

    Returns:
        An array of shape `(n, n_components)` containing the coordinates of the points
        in the reduced dimensionality space.

    Raises:
        ValueError: If `dissimilarity_matrix` is not a square matrix.

    Example:
        >>> dissimilarity_matrix = np.array([[0.0, 0.5, 0.2],
        ...                                  [0.5, 0.0, 0.8],
        ...                                  [0.2, 0.8, 0.0]])
        >>> embedding = mds(dissimilarity_matrix, n_components=2)
        >>> print(embedding.shape)
        (3, 2)
    """
    # Reference: https://www.sjsu.edu/faculty/guangliang.chen/Math253S20/lec9mds.pdf
    num_samples: int = dissimilarity_matrix.shape[0]

    # Compute G matrix (double-centered)
    J = np.eye(num_samples) - np.ones((num_samples, num_samples)) / num_samples
    G = -0.5 * J @ (dissimilarity_matrix**2) @ J

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    # Sort eigenvectors based on their eigenvalues in descending order
    indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    # Get top n_components
    top_eigenvalues = eigenvalues[:n_components]
    top_eigenvectors = eigenvectors[:, :n_components]

    # Compute coordinates of points
    return top_eigenvectors * np.sqrt(np.maximum(top_eigenvalues, 0))


def pca_variance_explained(dissimilarity_matrix: npt.NDArray) -> npt.NDArray:
    """
    Computes PCA Variance Explained Ratios given a dissimilarity matrix.

    Args:
        dissimilarity_matrix: A square matrix of shape `(n, n)` representing the
            dissimilarities between `n` samples.

    Returns:
        An array of shape `n` containing the PCA variance explained ratios.
    """
    pca_model = PCA()
    pca_model.fit(dissimilarity_matrix)
    return pca_model.explained_variance_ratio_


def pca(matrix: npt.NDArray, n_components: int) -> npt.NDArray:
    """
    Computes the principal components of a given 2D matrix.

    Args:
        matrix: A 2D array of the responses.
        n_components: The number of principal components to compute.

    Returns:
        An `n x n_components` array containing the projected data in
        principal component space.
    """
    pca_model = PCA(n_components=n_components)
    return pca_model.fit_transform(matrix)
