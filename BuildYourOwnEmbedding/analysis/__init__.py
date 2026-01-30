"""
Analysis tools for neural response data

This module provides dimensionality reduction and visualisation functions
for analysing neural response patterns.

Submodules:
    dimensionality: MDS, PCA, and other dimensionality reduction functions
    visualisation: Plotting utilities for RDMs
"""

from BuildYourOwnEmbedding.analysis.dimensionality import mds, pca, pca_variance_explained
from BuildYourOwnEmbedding.analysis.visualisation import (
    plot_rdm,
)

__all__ = [
    # dimensionality:
    "mds",
    "pca",
    "pca_variance_explained",
    # visualisation:
    "plot_rdm",
]
