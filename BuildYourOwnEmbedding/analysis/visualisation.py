"""
Visualisation utilities for neural response analysis

This module provides functions for plotting RDMs and other visualisations
commonly used in RSA

Functions:
    plot_rdm: Visualise a Representation Dissimilarity Matrix
"""

import matplotlib.pyplot as plt
import numpy.typing as npt


def plot_rdm(
    rdm: npt.NDArray,
    labels: list[str] | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    figsize: tuple[int, int] = (7, 7),
    dissimilarity_label: str = "Dissimilarity",
) -> None:
    """
    Plots a Representational Dissimilarity Matrix (RDM).

    This function visualises a given RDM as a heatmap, where each cell represents
    the pairwise dissimilarity between different neural responses or conditions.

    Args:
        rdm: A 2D numpy array representing the RDM, where the element at position
            `(i, j)` indicates the dissimilarity between response `i` and response `j`.
        labels: A list of string labels for the x and y axes, corresponding to each
            data point. If None, the axes will be labeled with indices.
        cmap: The colormap used to display the RDM. Defaults to "viridis".
        title: The title of the plot. If None, defaults to "RDM of Tuning Curves".
        figsize: The size of the figure in inches as (width, height). Defaults to (7, 7).
        dissimilarity_label: The label for the colorbar. Defaults to "Dissimilarity".
    """
    title = "RDM of Tuning Curves" if title is None else title
    fig: plt.Figure = plt.figure(figsize=figsize)

    plt.imshow(rdm, cmap=cmap)
    plt.colorbar(label=dissimilarity_label)
    plt.title(title)

    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=75, ha="left")
        plt.yticks(range(len(labels)), labels)
    plt.show()
