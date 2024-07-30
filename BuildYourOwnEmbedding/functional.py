import numpy as np
import numpy.typing as npt




##! =====================================
##!          Similarity Metrics:
##! =====================================

def inverse_correlation(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> npt.number:
    return dtype(1 - np.corrcoef(x1, x2)[0, 1])


def manhattan_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> npt.number:
    return dtype(np.sum(np.abs(x1 - x2)))


def euclidian_distance(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> npt.number:
    return dtype(np.linalg.norm(x1 - x2))


def cosine_similarity(
    x1: npt.NDArray, x2: npt.NDArray, dtype: npt.DTypeLike = np.float64
) -> npt.number:
    return dtype(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
