"""
Data classes for storing and analysing neural responses.

This module provides the ResponseData and ResponseSet classes for storing
neural response data.

Classes:
    ResponseData: Stores a single response with its parameters
    ResponseSet: Stores and analyses a collections of responses
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import numpy.typing as npt
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D

from BuildYourOwnEmbedding.analysis.dimensionality import mds, pca, pca_variance_explained
from BuildYourOwnEmbedding.analysis.visualisation import plot_rdm as _plot_rdm
from BuildYourOwnEmbedding.metrics.information import fisher_information as fi
from BuildYourOwnEmbedding.metrics.similarity import inverse_correlation


@dataclass
class ResponseData:
    """
    Data class for storing a single response output and the parameters that created it.

    Attributes:
        params: The parameters used for generating the response.
        response: The response values as a numpy array.
        response_function: The name of the ResponseFunction that generated this response.
        x: The input stimulus that generated this response.
    """

    params: dict[str, npt.number]
    response: npt.NDArray
    response_function: str
    x: npt.NDArray

    def __str__(self) -> str:
        """Return a string representation of the response data."""
        return f"{self.response_function}:\n" + "\n".join(
            f"{param_name} = {self.params[param_name]}" for param_name in self.params.keys()
        )


class ResponseSet:
    """
    A class for storing and analysing a set of neural responses.

    The ResponseSet class provides methods to compute representational
    dissimilarity matrices (RDM) and representational geodesic topological
    matrices (RGTM), as well as functions for visualising and analysing
    responses. It supports both 1D and 2D response visualisations and also
    allows interactive plots.

    Attributes:
        responses: A list of ResponseData objects that store the neural
            responses and their associated parameters.
    """

    def __init__(self, responses: list[ResponseData] | None = None) -> None:
        """
        Initialise the ResponseSet with a list of response objects.

        Args:
            responses: A list of ResponseData objects representing neural responses.
                Defaults to None (empty list).
        """
        if responses:
            self.responses = responses
        else:
            self.responses = []

    def __len__(self) -> int:
        """Return the number of responses in the set."""
        return len(self.responses)

    def __iter__(self):
        """Iterate over responses in the set."""
        return iter(self.responses)

    def __getitem__(self, index: int) -> ResponseData:
        """Get a response by index."""
        return self.responses[index]

    def compute_rdm(
        self,
        dissimilarity_metric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
    ) -> npt.ArrayLike:
        """
        Computes the Representational Dissimilarity Matrix (RDM) for the stored responses.

        Args:
            dissimilarity_metric: A function to compute dissimilarity between two
                responses. Defaults to inverse_correlation.

        Returns:
            A square RDM where each entry (i, j) represents the dissimilarity
            between the i-th and j-th responses.
        """
        n: int = len(self.responses)
        rdm: npt.NDArray = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                rdm[i, j] = rdm[j, i] = dissimilarity_metric(
                    self.responses[i].response, self.responses[j].response
                )

        return rdm

    def compute_rgtm(
        self,
        lower_bound: npt.number,
        upper_bound: npt.number,
        dissimilarity_metric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
    ) -> npt.ArrayLike:
        """
        Computes the Representational Geodesic Topological Matrix (RGTM).

        The RGTM is computed by applying a piecewise linear transformation on the
        dissimilarity matrix based on the provided lower and upper bounds.

        Args:
            lower_bound: The lower bound for the transformation.
            upper_bound: The upper bound for the transformation.
            dissimilarity_metric: A function to compute dissimilarity between two
                responses. Defaults to inverse_correlation.

        Returns:
            The computed RGTM matrix.

        Raises:
            AssertionError: If lower_bound > upper_bound.
        """
        assert lower_bound <= upper_bound

        n: int = len(self.responses)
        rgtm: npt.NDArray = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Compute dissimilarity
                dissimilarity = dissimilarity_metric(
                    self.responses[i].response, self.responses[j].response
                )

                # Compute geo-topological transform
                if dissimilarity <= lower_bound:
                    transformed = 0
                elif dissimilarity >= upper_bound:
                    transformed = 1
                else:
                    transformed = (dissimilarity - lower_bound) / (upper_bound - lower_bound)
                rgtm[i, j] = rgtm[j, i] = transformed
        return rgtm

    def plot_rgtm(
        self,
        lower_bound: npt.number = 0,
        upper_bound: npt.number = 1,
        dissimilarity_metric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        interactive: bool = True,
        labels: list[str] | None = None,
        cmap: str = "viridis",
        title: str = "Representational Geo-Topological Matrix",
        figsize: tuple[int, int] = (7, 7),
        dissimilarity_label: str = "Dissimilarity",
    ) -> None:
        """
        Plots the Representational Geo-Topological Matrix (RGTM) with optional interactivity.

        Args:
            lower_bound: The lower bound for the RGTM transformation. Defaults to 0.
            upper_bound: The upper bound for the RGTM transformation. Defaults to 1.
            dissimilarity_metric: Function to compute dissimilarity between two responses.
                Defaults to inverse_correlation.
            interactive: Whether to include sliders for adjusting the bounds. Defaults to True.
            labels: List of labels for the axes. Defaults to None.
            cmap: Colormap used for plotting. Defaults to "viridis".
            title: Title of the plot. Defaults to "Representational Geo-Topological Matrix".
            figsize: Size of the figure. Defaults to (7, 7).
            dissimilarity_label: Label for the color bar. Defaults to "Dissimilarity".
        """
        rgtm: npt.NDArray = self.compute_rgtm(lower_bound, upper_bound, dissimilarity_metric)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        heatmap = ax.imshow(rgtm, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(heatmap, ax=ax, label=dissimilarity_label)
        ax.set_title(title)

        if labels:
            ax.set_xticks(labels)
            ax.set_yticks(labels)

        if interactive:
            plt.subplots_adjust(left=0.25, bottom=0.25)

            lower_slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")
            upper_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow")

            lower_slider = Slider(
                ax=lower_slider_ax,
                label="Lower Bound (l)",
                valmin=0,
                valmax=1,
                valinit=lower_bound,
            )
            upper_slider = Slider(
                ax=upper_slider_ax,
                label="Upper Bound (u)",
                valmin=0,
                valmax=1,
                valinit=upper_bound,
            )

            def on_change(*args, **kwargs):
                new_lower = lower_slider.val
                new_upper = upper_slider.val
                if new_lower > new_upper:
                    ax.set_title(
                        "Error: Upper bound must be >= lower bound",
                        color="red",
                    )
                    return

                new_rgtm = self.compute_rgtm(new_lower, new_upper, dissimilarity_metric)
                heatmap.set_data(new_rgtm)
                ax.set_title(title, color="black")
                fig.canvas.draw_idle()

            lower_slider.on_changed(on_change)
            upper_slider.on_changed(on_change)

        plt.show()

    def plot_rdm(
        self,
        dissimilarity_metric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        labels: list[str] | None = None,
        cmap: str = "viridis",
        title: str | None = None,
        figsize: tuple[int, int] = (7, 7),
        dissimilarity_label: str = "Dissimilarity",
    ) -> None:
        """
        Plots the Representational Dissimilarity Matrix (RDM) for the stored responses.

        Args:
            dissimilarity_metric: Function to compute dissimilarity between responses.
                Defaults to inverse_correlation.
            labels: Labels for the plot's x and y axes. Defaults to None.
            cmap: Colormap for the plot. Defaults to "viridis".
            title: Title of the plot. Defaults to None.
            figsize: Size of the figure. Defaults to (7, 7).
            dissimilarity_label: Label for the color bar. Defaults to "Dissimilarity".
        """
        rdm = self.compute_rdm(dissimilarity_metric)
        _plot_rdm(rdm, labels, cmap, title, figsize, dissimilarity_label)

    def plot_responses(
        self,
        figsize: tuple[int, int] = (7, 7),
        xlabel: str = "Stimuli",
        ylabel: str = "Response",
        title: str = "Responses",
        grid: bool = False,
        hover_effects: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        Plots the responses in 1D or 2D format.

        Higher dimensional plots are not currently supported.

        Args:
            figsize: Size of the figure. Defaults to (7, 7).
            xlabel: Label for the x-axis. Defaults to "Stimuli".
            ylabel: Label for the y-axis. Defaults to "Response".
            title: Title of the plot. Defaults to "Responses".
            grid: Whether to display a grid. Defaults to False.
            hover_effects: If True, adds hover effects. Defaults to True.

        Raises:
            NotImplementedError: If responses are 3-dimensional or higher.
        """
        response_shape = list(self.responses[0].response.shape)

        if len(response_shape) == 1:
            self._plot_responses_1d(
                figsize, xlabel, ylabel, title, grid, hover_effects, *args, **kwargs
            )
        elif len(response_shape) == 2:
            self._plot_responses_2d(
                figsize=figsize,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                *args,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Plotting responses higher than two dimensions is not currently supported"
            )

    def _plot_responses_1d(
        self,
        figsize: tuple[int, int] = (7, 7),
        xlabel: str = "Stimuli",
        ylabel: str = "Response",
        title: str = "Responses",
        grid: bool = False,
        hover_effects: bool = True,
    ) -> None:
        """Plot 1D responses with optional hover effects."""
        plt.figure(figsize=figsize)
        plotted_responses = []
        for i in range(len(self.responses)):
            (plotted_response,) = plt.plot(self.responses[i].x, self.responses[i].response)
            plotted_responses.append(plotted_response)

        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.title(label=title)
        plt.grid(visible=grid)

        if hover_effects:
            cursor = mplcursors.cursor(hover=True)

            @cursor.connect("add")
            def on_add(selected_response) -> None:
                curve = selected_response.artist
                idx = plotted_responses.index(curve)
                selected_response.annotation.set(text=f"{str(self.responses[idx])}")
                selected_response.annotation.get_bbox_patch().set(fc="white", alpha=0.8)

        plt.show()

    def _plot_responses_2d(
        self,
        figsize: tuple[int, int] = (7, 7),
        xlabel: str = "Stimuli 1",
        zlabel: str = "Stimuli 2",
        ylabel: str = "Response",
        title: str = "2D Tuning Curve Visualisation",
        cmap: str = "viridis",
        plot_type: str = "heatmap",
    ) -> None:
        """
        Plot 2D responses as heatmap or surface plot.

        Args:
            plot_type: Either "heatmap" or "surfaceplot".
        """
        if plot_type == "surfaceplot":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            self._current_response_index = 0

            def update_plot(response_index: int) -> None:
                ax.clear()
                response = self.responses[response_index].response
                x, y = self.responses[response_index].x

                ax.plot_surface(x, y, response, cmap=cmap)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_zlabel(zlabel)
                ax.set_title(title)
                plt.draw()

            update_plot(self._current_response_index)

            def next_response(e) -> None:
                self._current_response_index = (self._current_response_index + 1) % len(
                    self.responses
                )
                update_plot(self._current_response_index)

            def prev_response(e) -> None:
                self._current_response_index = (self._current_response_index - 1) % len(
                    self.responses
                )
                update_plot(self._current_response_index)

            ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
            btn_next = Button(ax_next, "Next")
            btn_next.on_clicked(next_response)
            ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
            btn_prev = Button(ax_prev, "Previous")
            btn_prev.on_clicked(prev_response)
            plt.show()

        elif plot_type == "heatmap":
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            self._current_response_index = 0

            def update_plot(response_index: int) -> None:
                response = self.responses[response_index].response
                x, y = self.responses[response_index].x

                ax.imshow(
                    response,
                    cmap=cmap,
                    origin="lower",
                    extent=[x.min(), x.max(), y.min(), y.max()],
                )
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                plt.draw()

            update_plot(self._current_response_index)

            def next_response(e) -> None:
                self._current_response_index = (self._current_response_index + 1) % len(
                    self.responses
                )
                update_plot(self._current_response_index)

            def prev_response(e) -> None:
                self._current_response_index = (self._current_response_index - 1) % len(
                    self.responses
                )
                update_plot(self._current_response_index)

            ax_next = plt.axes([0.8, 0, 0.1, 0.075])
            btn_next = Button(ax_next, "Next")
            btn_next.on_clicked(next_response)
            ax_prev = plt.axes([0.7, 0, 0.1, 0.075])
            btn_prev = Button(ax_prev, "Previous")
            btn_prev.on_clicked(prev_response)
            plt.show()

    def plot_3d_mds(
        self,
        dissimilarity_metric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        xlabel: str = "MDS Dimension 1",
        ylabel: str = "MDS Dimension 2",
        zlabel: str = "MDS Dimension 3",
        cmap: str = "viridis",
        cbar_label: str | None = None,
        title: str | None = None,
        figsize: tuple[int, int] = (7, 7),
        response_to_color: Callable[[ResponseData], npt.number] | str | None = None,
    ) -> None:
        """
        Plots a 3D Multidimensional Scaling (MDS) scatter plot.

        Args:
            dissimilarity_metric: Function to compute dissimilarity. Defaults to inverse_correlation.
            xlabel: Label for the x-axis. Defaults to "MDS Dimension 1".
            ylabel: Label for the y-axis. Defaults to "MDS Dimension 2".
            zlabel: Label for the z-axis. Defaults to "MDS Dimension 3".
            cmap: Colormap for the scatter points. Defaults to "viridis".
            cbar_label: Label for the color bar. Defaults to None.
            title: Title of the plot. Defaults to None.
            figsize: Size of the figure. Defaults to (7, 7).
            response_to_color: Function or parameter name to determine point colors.
                Defaults to None (uses first parameter).
        """
        rdm = self.compute_rdm(dissimilarity_metric)
        mds_coords = mds(rdm, n_components=3)

        fig = plt.figure(figsize=figsize)
        ax: Axes3D = fig.add_subplot(projection="3d")

        if response_to_color:
            if isinstance(response_to_color, str):
                colors = [response.params[response_to_color] for response in self.responses]
            elif callable(response_to_color):
                colors = [response_to_color(response) for response in self.responses]
        else:
            colors = [next(iter(response.params.values())) for response in self.responses]

        scat = ax.scatter(
            mds_coords[:, 0],
            mds_coords[:, 1],
            mds_coords[:, 2],
            c=colors,
            cmap=cmap,
            marker="o",
            s=100,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        cb = fig.colorbar(scat, ax=ax, shrink=0.5, aspect=10)
        if cbar_label:
            cb.set_label(cbar_label)

        plt.show()

    def plot_2d_mds(
        self,
        dissimilarity_metric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        xlabel: str = "MDS Dimension 1",
        ylabel: str = "MDS Dimension 2",
        cmap: str = "viridis",
        cbar_label: str | None = None,
        title: str | None = None,
        figsize: tuple[int, int] = (7, 7),
        response_to_color: Callable[[ResponseData], npt.number] | str | None = None,
    ) -> None:
        """
        Plots a 2D Multidimensional Scaling (MDS) scatter plot.

        Args:
            dissimilarity_metric: Function to compute dissimilarity. Defaults to inverse_correlation.
            xlabel: Label for the x-axis. Defaults to "MDS Dimension 1".
            ylabel: Label for the y-axis. Defaults to "MDS Dimension 2".
            cmap: Colormap for the scatter points. Defaults to "viridis".
            cbar_label: Label for the color bar. Defaults to None.
            title: Title of the plot. Defaults to None.
            figsize: Size of the figure. Defaults to (7, 7).
            response_to_color: Function or parameter name to determine point colors.
                Defaults to None (uses first parameter).
        """
        rdm = self.compute_rdm(dissimilarity_metric)
        mds_coords = mds(rdm, n_components=2)

        fig, ax = plt.subplots(figsize=figsize)

        if response_to_color:
            if isinstance(response_to_color, str):
                colors = [response.params[response_to_color] for response in self.responses]
            elif callable(response_to_color):
                colors = [response_to_color(response) for response in self.responses]
        else:
            colors = [next(iter(response.params.values())) for response in self.responses]

        scat = ax.scatter(
            mds_coords[:, 0],
            mds_coords[:, 1],
            c=colors,
            cmap=cmap,
            marker="o",
            s=100,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        cb = fig.colorbar(scat, ax=ax, shrink=0.5, aspect=10)
        if cbar_label:
            cb.set_label(cbar_label)

        plt.show()

    def plot_pca_variance_explained(
        self,
        dissimilarity_metric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        xlabel: str = "Number of Principal Components",
        ylabel: str = "Variance Explained",
        title: str = "Variance Explained vs. Number of Principal Components",
        figsize: tuple[int, int] = (7, 7),
        grid: bool = False,
    ) -> None:
        """
        Plots the variance explained by each principal component.

        Args:
            dissimilarity_metric: Function to compute dissimilarity. Defaults to inverse_correlation.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Title of the plot.
            figsize: Size of the figure. Defaults to (7, 7).
            grid: Whether to show grid lines. Defaults to False.
        """
        rdm = self.compute_rdm(dissimilarity_metric)
        pca_var = pca_variance_explained(rdm)

        plt.figure(figsize=figsize)
        plt.plot(range(1, len(pca_var) + 1), pca_var, marker="o")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(grid)
        plt.show()

    def plot_2d_pca(
        self,
        xlabel: str = "PC 1",
        ylabel: str = "PC 2",
        title: str = "2D PCA of Embedding",
        figsize: tuple[int, int] = (7, 7),
    ) -> None:
        """
        Plots a 2D PCA projection of the responses.

        Args:
            xlabel: Label for the x-axis. Defaults to "PC 1".
            ylabel: Label for the y-axis. Defaults to "PC 2".
            title: Title of the plot. Defaults to "2D PCA of Embedding".
            figsize: Size of the figure. Defaults to (7, 7).
        """
        curves = np.array([r.response for r in self.responses])
        pcs = pca(curves, n_components=2)

        plt.figure(figsize=figsize)
        plt.scatter(pcs[:, 0], pcs[:, 1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def fisher_information(self) -> npt.NDArray:
        """
        Computes the Fisher Information of the set of responses.

        Returns:
            A numpy array containing the Fisher Information of the stored responses.
        """
        curves = np.array([r.response for r in self.responses])
        return fi(curves)
