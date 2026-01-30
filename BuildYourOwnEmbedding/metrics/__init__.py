"""
Metrics for comparing and analysing neural responses

This module provides similarity metrics, distance functions, and
information-theoretic measures for neural response analysis.

Submodules:
    similarity: Distance and similarity metrics
    information: Information-theoretic measures
"""

from BuildYourOwnEmbedding.metrics.information import (
    fisher_information,
    mutual_information,
)
from BuildYourOwnEmbedding.metrics.similarity import (
    cosine_similarity,
    euclidean_distance,
    inverse_correlation,
    manhattan_distance,
)

__all__ = [
    # Similarity metrics
    "inverse_correlation",
    "manhattan_distance",
    "euclidean_distance",
    "cosine_similarity",
    # Information metrics
    "fisher_information",
    "mutual_information",
]
