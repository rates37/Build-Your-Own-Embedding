import numpy as np
import numpy.typing as npt
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

##! =====================================
##!          Similarity Metrics:
##! =====================================


def inverse_correlation(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    """Computes the inverse correlation between two numpy arrays.

    This function calculates the inverse of the Pearson correlation coefficient
    between two input arrays, `x1` and `x2`, and returns the result cast to the
    specified `dtype`.

    Args:
        x1 (npt.NDArray):
            The first input array
        x2 (npt.NDArray):
            The second input array
        dtype (npt.DTypeLike, optional):
            The desired data type of the result. Defaults to np.float64.

    Returns:
        np.number: The inverse correlation coefficient between `x1` and `x2`, cast to the specified `dtype`.
    """
    return dtype(1 - np.corrcoef(x1, x2)[0, 1])


def manhattan_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    """Computes the Manhattan distance between two numpy arrays.

    This function calculates the Manhattan distance (L1 norm), between two input
    arrays, `x1` and `x2`, and returns the result cast to the specified `dtype`.

    Args:
        x1 (npt.NDArray):
            The first input array.
        x2 (npt.NDArray):
            The second input array.
        dtype (npt.DTypeLike, optional):
            The desired data type of the result. Defaults to np.float64.

    Returns:
        np.number: The Manhattan distance between `x1` and `x2`, cast to the specified `dtype`.
    """
    return dtype(np.sum(np.abs(x1 - x2)))


def euclidian_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    """Computes the Euclidian distance between two numpy arrays.

    This function calculates the Euclidian distance (L2 norm), between two input
    arrays, `x1` and `x2`, and returns the result cast to the specified `dtype`.

    Args:
        x1 (npt.NDArray):
            The first input array.
        x2 (npt.NDArray):
            The second input array.
        dtype (npt.DTypeLike, optional):
            The desired data type of the result. Defaults to np.float64.

    Returns:
        np.number: The Euclidian distance between `x1` and `x2`, cast to the specified `dtype`.
    """
    return dtype(np.linalg.norm(x1 - x2))


def cosine_similarity(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    """Computes the cosine similarity between two numpy arrays.

    This function calculates the cosine similarity, between two input
    arrays, `x1` and `x2`, and returns the result cast to the specified `dtype`.
    If the input arrays are not 1D, they are flattened and then the cosine
    similarity is computed.

    Args:
        x1 (npt.NDArray):
            The first input array.
        x2 (npt.NDArray):
            The second input array.
        dtype (npt.DTypeLike, optional):
            The desired data type of the result. Defaults to np.float64.

    Returns:
        np.number: The cosine similarity between `x1` and `x2`, cast to the specified `dtype`.
    """
    x1Flat = np.hstack(x1)
    x2Flat = np.hstack(x2)
    return dtype(
        np.dot(x1Flat, x2Flat) / (np.linalg.norm(x1Flat) * np.linalg.norm(x2Flat))
    )


##! =====================================
##!         RDM-Related Functions:
##! =====================================
def plot_rdm(
    rdm: npt.NDArray,
    labels: Union[List[str], None] = None,
    cmap: str = "viridis",
    title: str = None,
    figsize: Tuple[int] = (7, 7),
    dissimilarityLabel: str = "Dissimilarity",
) -> None:
    """Plots a Representational Dissimilarity Matrix (RDM).

    This function visualises a given RDM given as a square matrix that represents the
    pairwise dissimilarities between different data points or conditions. The
    function provides customisation options for colormap, title, axis labels, and
    figure size, allowing users to tailor the appearance of the plot.

    Args:
        rdm (npt.NDArray):
            A 2D numpy array representing the RDM, where the element at position `(i,j)`
            indicates the dissimilarity between response `i` and response `j`.
        labels (Union[List[str], None], optional):
            A list of string labels for the x and y axes, corresponding to each data
            point. If `None`, the axes will be labeled with indices. Defaults to `None`.
        cmap (str, optional):
            The colormap used to display the RDM. The default colormap is "viridis", but
            any valid Matplotlib colormap string can be used.
        title (str, optional):
            The title of the plot. If `None`, a default title "RDM of Tuning Curves" will
            be used.
        figsize (Tuple[int], optional):
            The size of the figure in inches, specified as a tuple `(width, height)`.
            Defaults to `(7, 7)`.
        dissimilarityLabel (str, optional):
            The label for the colorbar, indicating the metric used for dissimilarity.
            Defaults to "Dissimilarity".
    """
    title = "RDM of Tuning Curves" if title is None else title
    fig: plt.Figure = plt.figure(figsize=figsize)

    plt.imshow(rdm, cmap=cmap)
    plt.colorbar(label=dissimilarityLabel)
    plt.title(title)

    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=75, ha="left")
        plt.yticks(range(len(labels)), labels)
    plt.show()


##! =====================================
##!             Metrics:
##! =====================================
def mds(dissimilarityMatrix: npt.NDArray, nComponents: int = 3) -> npt.NDArray:
    """Perform classical Multidimensional Scaling (MDS) on a dissimilarity matrix.

    This function implements classical MDS, a technique for visualising the level
    of similarity of dissimilarity between sets of data. Given a square dissimilarity
    matrix, the function computes a low-dimensional embedding of the data by finding
    the eigenvectors and eigenvalues of the matrix, resulting in a set of coordinates
    that best preserve the pairwise distances in the original matrix.

    This function implements an algorithm that follows these steps:

    1. Constructs a centreing matrix `J` and computes the double-centred matrix `G`.
    2. Calculates the eigenvalues and eigenvectors of `G`.
    3. Sorts the eignevectors based on their corresponding eigenvalues in descending order.
    4. Selects the top `nComponents` eigenvectors corresponding to the largest eigenvalues.
    5. Computes the coordinates of the points in the reduced dimensionality space.

    Args:
        dissimilarityMatrix (npt.NDArray):
            A square matrix of shape `(n, n)` representing the dissimilarities between
            `n` samples. Each element at position `(i, j)` represents the dissimilarity
            between the `i-th` and `j-th` samples.
        nComponents (int, optional):
            The number of dimensions for the output embedding. Defaults to 3.

    Returns:
        npt.NDArray:
            An array of shape `(n, nComponents)` containing the coordinates of the points
            in the reduced dimensionality space, where `n` is the number of samples and
            `nComponents` is the number of dimensions specified.

    Example:
        >>> dissimilarity_matrix = np.array([[0.0, 0.5, 0.2],
                                             [0.5, 0.0, 0.8],
                                             [0.2, 0.8, 0.0]])
        >>> embedding = mds(dissimilarity_matrix, nComponents=2)
        >>> print(embedding)

    Raises:
        ValueError: If `dissimilarityMatrix` is not a square matrix.

    """
    # used notes from: https://www.sjsu.edu/faculty/guangliang.chen/Math253S20/lec9mds.pdf
    numSamples: int = dissimilarityMatrix.shape[0]

    # compute G matrix:
    J = np.eye(numSamples) - np.ones((numSamples, numSamples)) / numSamples
    G = -0.5 * J @ (dissimilarityMatrix**2) @ J

    # find eigenvalues and eigenvectors:
    eigenValues, eigenVectors = np.linalg.eigh(G)

    # sort eigenvectors based on their eigenvalues in descending order:
    indices = np.argsort(eigenValues)  # ascending order
    indices = indices[::-1]  # descending order
    eigenValues = eigenValues[indices]
    eigenVectors = eigenVectors[:, indices]

    # get top nComponents:
    topNEigenValues = eigenValues[:nComponents]
    topNEigenVectors = eigenVectors[:, :nComponents]

    # compute coordinates of points:
    return topNEigenVectors * np.sqrt(topNEigenValues)


def pca_variance_explained(dissimilarityMatrix: npt.NDArray) -> npt.NDArray:
    """Computes PCA Variance Explained Ratios given a dissimilarity matrix.

    Args:
        dissimilarityMatrix (npt.NDArray):
            A square matrix of shape `(n, n)` representing the dissimilarities between
            `n` samples. Each element at position `(i, j)` represents the dissimilarity
            between the `i-th` and `j-th` samples.

    Returns:
        npt.NDArray:
            An array of shape `n` containing the PCA variance explained ratios of
            the samples in the input matrix.
    """
    pca = PCA()
    pca.fit(dissimilarityMatrix)
    return pca.explained_variance_ratio_


def pca(matrix: npt.NDArray, nComponents) -> npt.NDArray:
    """Computes the principal components of a given 2d matrix

    Args:
        matrix (npt.NDArray): A 2D array of the responses
        nComponents (_type_): The number of principal components to consider

    Returns:
        npt.NDArray: A n x `nComponents` array, containing the values of the principal components.
    """
    pca = PCA(n_components=nComponents)
    return pca.fit_transform(matrix)


##! =====================================
##!             Information:
##! =====================================
def fisher_information(curves: npt.NDArray) -> npt.NDArray:
    """Calculates the Fisher Information for a set of tuning curves.

    This function supports both 1-D and arbitrary-dimensional tuning curves

    Args:
        curves (npt.NDArray): A numpy array of the shape (numCurves, dim1, dim2,
        ..., dimN) representing the tuning curves, where `numCurves` is the number
        of curves and `dim1, dim2, ..., dimN` are the dimensions of each response.

    Raises:
        ValueError: If the dimensionality of `curves` is not greater than a 2-dimensional
        array.

    Returns:
        npt.NDArray: A numpy array representing the FI values, with shape (dim1, dim2,
        ..., dimN) for the input curves.
    """
    if len(curves.shape) < 2:
        raise ValueError("curves must be at least two dimensions")
    if len(curves.shape) == 2:
        fiValues = np.zeros(curves[0].shape)

        for i in range(curves.shape[0]):
            derivative = np.gradient(curves[i])
            fi = derivative**2
            fiValues += fi
        return fiValues
    else:
        fiValues = np.zeros(curves.shape[1:])
        for i in range(curves.shape[0]):
            derivatives = np.gradient(curves[i])
            fi = sum(derivative**2 for derivative in derivatives)
            fiValues += fi
        return fiValues


def mutual_information(
    response: npt.NDArray, stimulus: npt.NDArray, integrationPrecision: int = 1000
) -> np.number:
    """Computes the mutual information between a single neural response and its stimulus.

    Args:
        response (npt.NDArray): Neural response with shape (dim1, dim2, ..., dimN).
        stimulus (npt.NDArray): Corresponding stimulus with shape (dim1, dim2, ..., dimN).
        integrationPrecision (int, optional): Number of steps to use in integration. Defaults to 1000.

    Raises:
        ValueError: If the stimulus and response are different dimensions.

    Returns:
        np.number: The mutual information between the response and the stimulus.
    """
    if response.shape != stimulus.shape:
        raise ValueError("Response and Stimulus must have the same dimensions")

    responseFlat = response.flatten()
    stimulusFlat = stimulus.flatten()

    # compute joint distribution:
    jointValues = np.stack((responseFlat, stimulusFlat), axis=-1)
    uniqueJoint, countsJoint = np.unique(jointValues, axis=0, return_counts=True)
    jointProbabilities = countsJoint / len(responseFlat)

    # Compute marginal distributions
    uniqueResponse, countsResponse = np.unique(responseFlat, return_counts=True)
    probResponse = countsResponse / len(responseFlat)

    uniqueStimulus, countsStimulus = np.unique(stimulusFlat, return_counts=True)
    probStimulus = countsStimulus / len(stimulusFlat)

    responsePDict = {val: probResponse[i] for i, val in enumerate(uniqueResponse)}
    stimulusPDict = {val: probStimulus[i] for i, val in enumerate(uniqueStimulus)}

    mi = 0.0
    for i, jointVal in enumerate(uniqueJoint):
        respVal, stimVal = jointVal
        joint_prob = jointProbabilities[i]
        if joint_prob > 0 and responsePDict[respVal] > 0 and stimulusPDict[stimVal] > 0:
            mi += joint_prob * np.log(
                joint_prob / (responsePDict[respVal] * stimulusPDict[stimVal])
            )

    return mi
