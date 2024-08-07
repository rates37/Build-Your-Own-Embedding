import numpy as np
import numpy.typing as npt
from typing import List, Union, Tuple
import matplotlib.pyplot as plt


##! =====================================
##!          Similarity Metrics:
##! =====================================


def inverse_correlation(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> npt.number:
    """Computes the inverse correlation between two numpy arrays.

    This function calculates the inverse of the Pearson correlation coefficient
    between two input arrays, `x1` and `x2`, and returns the result cast to the
    specified `dtype`.

    Args:
        x1 (npt.NDArray): The first input array
        x2 (npt.NDArray): The second input array
        dtype (npt.DTypeLike, optional): The desired data type of the result. 
            Defaults to np.float64.

    Returns:
        npt.number: The inverse correlation coefficient between `x1` and `x2`,
            cast to the specified `dtype`.
    """
    return dtype(1 - np.corrcoef(x1, x2)[0, 1])


def manhattan_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> npt.number:
    """Computes the Manhattan distance between two numpy arrays.
    
    This function calculates the Manhattan distance (L1 norm), between two input 
    arrays, `x1` and `x2`, and returns the result cast to the specified `dtype`.

    Args:
        x1 (npt.NDArray): The first input array.
        x2 (npt.NDArray): The second input array.
        dtype (npt.DTypeLike, optional): The desired data type of the result. 
            Defaults to np.float64.

    Returns:
        npt.number: The Manhattan distance between `x1` and `x2`, cast to the 
            specified `dtype`.
    """
    return dtype(np.sum(np.abs(x1 - x2)))


def euclidian_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> npt.number:
    """Computes the Euclidian distance between two numpy arrays.
    
    This function calculates the Euclidian distance (L2 norm), between two input
    arrays, `x1` and `x2`, and returns the result cast to the specified `dtype`.

    Args:
        x1 (npt.NDArray): The first input array.
        x2 (npt.NDArray): The second input array.
        dtype (npt.DTypeLike, optional): The desired data type of the result. 
            Defaults to np.float64.

    Returns:
        npt.number: The Euclidian distance between `x1` and `x2`, cast to the 
            specified `dtype`.
    """
    return dtype(np.linalg.norm(x1 - x2))


def cosine_similarity(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> npt.number:
    """Computes the cosine similarity between two numpy arrays.
    
    This function calculates the cosine similarity, between two input
    arrays, `x1` and `x2`, and returns the result cast to the specified `dtype`.

    Args:
        x1 (npt.NDArray): The first input array.
        x2 (npt.NDArray): The second input array.
        dtype (npt.DTypeLike, optional): The desired data type of the result. 
            Defaults to np.float64.

    Returns:
        npt.number: The cosine similarity between `x1` and `x2`, cast to the 
            specified `dtype`.
    """
    return dtype(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))


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
        rdm (npt.NDArray): A 2D numpy array representing the RDM, where the 
            element at position `(i,j)` indicates the dissimilarity between response
            `i` and response `j`.
        labels (Union[List[str], None], optional): A list of string labels for
            the x and y axes, corresponding to each data point. If `None`, the
            axes will be labeled with indices. Defaults to `None`.
        cmap (str, optional): The colormap used to display the RDM. The default
            colormap is "viridis", but any valid Matplotlib colormap string
            can be used.
        title (str, optional): The title of the plot. If `None`, a default title
            "RDM of Tuning Curves" will be used.
        figsize (Tuple[int], optional): The size of the figure in inches,
            specified as a tuple `(width, height)`. Defaults to `(7, 7)`.
        dissimilarityLabel (str, optional): The label for the colorbar, indicating
            the metric used for dissimilarity. Defaults to "Dissimilarity".
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
##!         RDM-Related Functions:
##! =====================================
def mds(dissimilarityMatrix: npt.NDArray, nComponents: int = 3) -> npt.NDArray:
    # implements classical MDS
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
