import numpy as np
import numpy.typing as npt
from typing import List, Union, Tuple
import matplotlib.pyplot as plt


##! =====================================
##!          Similarity Metrics:
##! =====================================


def inverse_correlation(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    return dtype(1 - np.corrcoef(x1, x2)[0, 1])


def manhattan_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    return dtype(np.sum(np.abs(x1 - x2)))


def euclidian_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
    return dtype(np.linalg.norm(x1 - x2))


def cosine_similarity(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> np.number:
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
