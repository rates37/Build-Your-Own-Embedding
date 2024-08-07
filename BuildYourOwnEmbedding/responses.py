from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Union, Dict, Tuple, Any, Type, Callable, List
import inspect
import itertools
import mplcursors
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .parameters import Parameter
from .functional import inverse_correlation, plot_rdm, mds

##! =====================================
##!             Responses:
##! =====================================


class ResponseFunction(ABC):
    """
    Base class for defining neural responses.

    The `ResponseFunction` class serves as an abstract base class for creating neural
    response functions that can be evaluated based on input data. This class provides
    a standardised interface for defining neural response curves, allowing users to
    implement custom responses by extending this class and implementing the `evaluate`
    method. It also supports mathematical operations to combine responses, creating
    composite response functions.

    Attributes:
        params (Dict): A dictionary containing response-specific parameters used
                       to define the behavior of the response function.

    Methods:
        evaluate(x: npt.NDArray) -> npt.NDArray:
            Abstract method to evaluate the neural response for a given input.

        __call__(x: npt.NDArray) -> npt.NDArray:
            Calls the `evaluate` method to compute the response for the given input.

        __add__(other: ResponseFunction) -> CompositeResponse:
            Combines this response with another response using addition.

        __sub__(other: ResponseFunction) -> CompositeResponse:
            Combines this response with another response using subtraction.

        __mul__(other: ResponseFunction) -> CompositeResponse:
            Combines this response with another response using multiplication.

        __str__() -> str:
            Returns a string representation of the response function.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialises the response function with specific parameters.

        This constructor accepts named arguments for the parameters that define
        the behavior of the neural response function. These parameters are stored
        in a dictionary for later use by the response function.

        Args:
            **kwargs (Any):
                Named arguments representing the parameters of the response function.

        Notes:
            Subclasses should provide specific parameter requirements, which will be
            documented in their respective class definitions.
        """
        self.params: Dict = kwargs  # Store all parameters for general use

    @abstractmethod
    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the response of the neuron for a given input.

        This abstract method must be implemented by subclasses to define how the
        response function computes output values based on input data.

        Args:
            x (npt.NDArray):
                The input data for which the response is evaluated. This should be
                a Numpy array representing the stimulus or feature input to the neuron.

        Returns:
            npt.NDArray:
                A Numpy array representing the evaluated response values corresponding
                to the input data.

        Raises:
            NotImplementedError:
                This method must be overridden by subclasses, otherwise, calling it
                will raise this error.
        """
        pass

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        """
        Compute the response by calling the evaluate method.

        This method allows the object to be used as a callable function, directly
        invoking the `evaluate` method when the instance is called with input data.

        Args:
            x (npt.NDArray):
                The input data for which the response is evaluated.

        Returns:
            npt.NDArray:
                The response values computed by the `evaluate` method.
        """
        return self.evaluate(x)

    def __add__(self, other: ResponseFunction) -> CompositeResponse:
        """
        Add another response function to this response function.

        This method allows the creation of a composite response by adding the outputs
        of two different response functions. The resulting composite response is
        represented by a new instance of the `CompositeResponse` class.

        Args:
            other (ResponseFunction):
                The other response function to be added to this response.

        Returns:
            CompositeResponse:
                A new `CompositeResponse` instance representing the combined response
                of this response and the other response.
        """
        return CompositeResponse(self, other, np.add)

    def __sub__(self, other: ResponseFunction) -> CompositeResponse:
        """
        Subtract another response function from this response function.

        This method allows the creation of a composite response by subtracting the
        outputs of another response function from this response. The resulting composite
        response is represented by a new instance of the `CompositeResponse` class.

        Args:
            other (ResponseFunction):
                The other response function to be subtracted from this response.

        Returns:
            CompositeResponse:
                A new `CompositeResponse` instance representing the difference between
                this response and the other response.
        """
        return CompositeResponse(self, other, np.subtract)

    def __mul__(self, other: ResponseFunction) -> CompositeResponse:
        """
        Multiply this response function by another response function.

        This method allows the creation of a composite response by multiplying the
        outputs of two different response functions. The resulting composite response
        is represented by a new instance of the `CompositeResponse` class.

        Args:
            other (ResponseFunction):
                The other response function to multiply with this response.

        Returns:
            CompositeResponse:
                A new `CompositeResponse` instance representing the product of this
                response and the other response.
        """
        return CompositeResponse(self, other, np.multiply)

    def __str__(self) -> str:
        """
        Return a string representation of the response function.

        This method provides a human-readable representation of the response function,
        including its class name and the parameters used to define its behavior.

        Returns:
            str:
                A string representation of the response function, displaying the class
                name and its parameter values.
        """
        return f"{type(self).__name__}, {self.params}"


class CompositeResponse(ResponseFunction):
    """
    Composite response class that combines two responses using a specified operation.
    """

    def __init__(
        self,
        response1: ResponseFunction,
        response2: ResponseFunction,
        operation: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    ) -> None:
        """
        Initialises a composite response function from two individual responses and an operation.

        Args:
            response1 (ResponseFunction): First response
            response2 (ResponseFunction): Second response
            operation (Callable[[npt.NDArray, npt.NDArray], npt.NDArray]): An operation applied to the outputs of the individual responses
        """
        super().__init__()
        self.response1: ResponseFunction = response1
        self.response2: ResponseFunction = response2
        self.operation = operation

    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        return self.operation(self.response1(x), self.response2(x))

    def __str__(self) -> str:
        return (
            f"{self.operation.__name__}({str(self.response1)}, {str(self.response2)})"
        )


class GaussianResponse(ResponseFunction):
    def __init__(self, mean: npt.number, std: npt.number) -> None:
        super().__init__(mean=mean, std=std)

    def evaluate(
        self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        mean = self.params["mean"]
        std = self.params["std"]
        return dtype(
            np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        )


class SigmoidResponse(ResponseFunction):
    def __init__(self, alpha: npt.number, beta: npt.number) -> None:
        super().__init__(alpha=alpha, beta=beta)

    def evaluate(
        self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        return dtype(1 / (1 + np.exp(-alpha * (x - beta))))


class VonMisesResponse(ResponseFunction):
    def __init__(self, kappa: npt.number, theta: npt.number) -> None:
        super().__init__(kappa=kappa, theta=theta)

    def evaluate(
        self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        kappa = self.params["kappa"]
        theta = self.params["theta"]
        return np.exp(kappa * np.cos(x - theta)) / (2 * np.pi * np.sinh(kappa))


##! =====================================
##!         Response Manager:
##! =====================================
# this class will generate sets of responses:
class ResponseManager:
    def __init__(
        self,
        responseClasses: Union[
            Type[ResponseFunction], Tuple[Type[ResponseFunction], ...]
        ],
        **parameters: Parameter,
    ) -> None:
        """
        Initialises response manager class with specific response type and parameters.

        Args:
            responseClasses (Type[ResponseFunction]): A single response class or a tuple of response classes to generate responses for.
            **param_strategies (Dict[str, Parameter]): Parameter strategies for each response class.
        """
        self.responseClasses: Type[ResponseFunction] = responseClasses
        self.parameters: Dict[str, Parameter] = parameters

    def _get_response_parameters(self) -> List[str]:
        """
        Gets the names of the parameters from the response class' __init__ method

        Returns:
            List[str]: A list of parameter names
        """
        return list(inspect.signature(self.responseClasses.__init__).parameters.keys())[
            1:
        ]

    def _get_combinations(self) -> List[Dict[str, Any]]:
        """
        Gets all combinations of parameter values based on the input parameters

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing all combinations of parameter values
        """
        # paramValuesDict = {str(i): param.get_values() for i, param in enumerate(self.parameters)} #!
        paramValuesDict = self.parameters
        paramNames = self._get_response_parameters()
        paramCombinations = itertools.product(
            *[p.get_values() for p in [*paramValuesDict.values()]]
        )
        return [dict(zip(paramNames, paramCombo)) for paramCombo in paramCombinations]

    def _generate_response(
        self, x: npt.NDArray, responseParams: Dict[str, Any]
    ) -> ResponseData:
        """
        Generates a single response based on the provided parameter values

        Args:
            x (npt.NDArray): The input data for which the response is generated
            responseParams (Dict[str, Any]): Parameter values for the response

        Returns:
            ResponseData: A ResponseData object containing the response values and the corresponding parameters.
        """
        response = self.responseClasses(**responseParams)
        responseValues = response(x)
        return ResponseData(
            params=responseParams,
            response=responseValues,
            responseFunction=self.responseClasses.__name__,
            x=x,
        )

    def generate_responses(self, x: npt.NDArray) -> List[ResponseData]:
        """
        Generates responses for each combination of parameter values

        Args:
            x (npt.NDArray): The input data for which the response is generated

        Returns:
            List[ResponseData]: List of ResponseData objects containing each response and the corresponding parameters.
        """
        paramCombinations = self._get_combinations()
        return [self._generate_response(x, params) for params in paramCombinations]


@dataclass
class ResponseData:
    """
    Data class for storing a single response output and the parameters that created it.

    Attributes:
        params (Dict[str, np.number]): The parameters used for generating the response.
        response (np.ndarray): The response values.
        responseFunction (str): The name of the Response Function that generated this Response
    """

    params: Dict[str, npt.number]
    response: npt.NDArray
    responseFunction: str
    x: npt.NDArray

    def __str__(self) -> str:
        return f"{self.responseFunction}:\n" + "\n".join(
            f"{paramName} = {self.params[paramName]}"
            for paramName in self.params.keys()
        )


# stores a set of responses
class ResponseSet:
    def __init__(self, responses: Union[None, List[ResponseData]] = None) -> None:
        if responses:
            self.responses = responses
        else:
            self.responses = []

    def compute_rdm(
        self,
        dissimilarityMetric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
    ) -> npt.ArrayLike:
        n: int = len(self.responses)
        rdm: npt.NDArray = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                rdm[i, j] = rdm[j, i] = dissimilarityMetric(
                    self.responses[i].response, self.responses[j].response
                )

        return rdm

    def compute_rgtm(
        self,
        lowerBound: npt.number,
        upperBound: npt.number,
        dissimilarityMetric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
    ) -> npt.ArrayLike:

        assert lowerBound <= upperBound

        n: int = len(self.responses)
        rgtm: npt.NDArray = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # compute dissimilarity:
                dissimilarity = dissimilarityMetric(
                    self.responses[i].response, self.responses[j].response
                )

                # compute geo-topological transform:
                if dissimilarity <= lowerBound:
                    transformedDissimilarity = 0
                elif dissimilarity >= upperBound:
                    transformedDissimilarity = 1
                else:
                    transformedDissimilarity = (dissimilarity - lowerBound) / (
                        upperBound - lowerBound
                    )
                rgtm[i, j] = rgtm[j, i] = transformedDissimilarity
        return rgtm

    def plot_rgtm(
        self,
        lowerBound: npt.number = 0,
        upperBound: npt.number = 1,
        dissimilarityMetric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        interactive: bool = True,
        labels: Union[List[str], None] = None,
        cmap: str = "viridis",
        title: str = "Representational Geo-Topological Matrix",
        figsize: Tuple[int] = (7, 7),
        dissimilarityLabel: str = "Dissimilarity",
    ) -> None:

        rgtm: npt.NDArray = self.compute_rgtm(
            lowerBound, upperBound, dissimilarityMetric
        )

        # create the plot:
        fig, ax = plt.subplots(figsize=figsize)

        # display initial RGTM:
        heatmap = ax.imshow(rgtm, cmap=cmap, vmin=0, vmax=1)  # ?
        plt.colorbar(heatmap, ax=ax, label=dissimilarityLabel)
        ax.set_title(title)

        # todo: show labels if not none:
        # if labels:
        #     ax.set_xticks(labels)
        #     ax.set_yticks(labels)

        # if interactive, add interactivity behaviour:
        if interactive:
            # reposition plot to account for sliders:
            plt.subplots_adjust(left=0.25, bottom=0.25)

            # add axes for the sliders:
            lowerBoundSliderAx = plt.axes(
                [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
            )
            upperBoundSliderAx = plt.axes(
                [0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow"
            )

            # add the sliders:
            lowerBoundSlider = Slider(
                ax=lowerBoundSliderAx,
                label="Lower Bound (l)",
                valmin=0,
                valmax=1,
                valinit=lowerBound,
            )
            upperBoundSlider = Slider(
                ax=upperBoundSliderAx,
                label="Upper Bound (u)",
                valmin=0,
                valmax=1,
                valinit=upperBound,
            )

            # define update function for slider on change event:
            def onChange(*args, **kwargs):
                # re-compute the RGTM:
                newLowerBound = lowerBoundSlider.val
                newUpperBound = upperBoundSlider.val
                if newLowerBound > newUpperBound:
                    # display error message:
                    ax.set_title(
                        "Error: Upper bound must be greater than or equal to the lower bound",
                        color="red",
                    )
                    return  # don't plot if bounds are invalid

                # compute new RGTM:
                newRgtm = self.compute_rgtm(
                    newLowerBound, newUpperBound, dissimilarityMetric
                )

                # update plot:
                heatmap.set_data(newRgtm)
                ax.set_title(title, color="black")
                fig.canvas.draw_idle()

            # add on change function to both sliders
            lowerBoundSlider.on_changed(onChange)
            upperBoundSlider.on_changed(onChange)

        # show the plot:
        plt.show()

    def plot_rdm(
        self,
        dissimilarityMetric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        labels: Union[List[str], None] = None,
        cmap: str = "viridis",
        title: str = None,
        figsize: Tuple[int] = (7, 7),
        dissimilarityLabel: str = "Dissimilarity",
    ) -> None:
        rdm = self.compute_rdm(dissimilarityMetric)
        plot_rdm(rdm, labels, cmap, title, figsize, dissimilarityLabel)
        pass

    def plot_responses(
        self,
        figsize: Tuple[int] = (7, 7),
        xlabel: str = "Stimuli",
        ylabel: str = "Response",
        title: str = "Responses",
        grid: bool = False,
        hoverEffects: bool = True,  # if false will not add hover effect to plot
    ) -> None:

        # plot the responses:
        plt.figure(figsize=figsize)
        plottedResponses = []
        for i in range(len(self.responses)):
            (plottedResponse,) = plt.plot(
                self.responses[i].x, self.responses[i].response
            )
            plottedResponses.append(plottedResponse)

        plt.xlabel(xlabel=xlabel)
        plt.ylabel(ylabel=ylabel)
        plt.title(label=title)
        plt.grid(visible=grid)

        if hoverEffects:
            cursor = mplcursors.cursor(hover=True)

            @cursor.connect("add")
            def on_add(selectedResponse) -> None:
                curve = selectedResponse.artist
                idx = plottedResponses.index(curve)

                # set annotation to show formatted parameters:
                # todo
                selectedResponse.annotation.set(text=f"{str(self.responses[idx])}")
                selectedResponse.annotation.get_bbox_patch().set(
                    fc="white", alpha=0.8
                )  # todo: make these parameters variable

        plt.show()
        pass

    def plot_3D_mds(
        self,
        dissimilarityMetric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        xlabel: str = "MDS Dimension 1",
        ylabel: str = "MDS Dimension 2",
        zlabel: str = "MDS Dimension 3",
        cmap: str = "viridis",
        cbarLabel: Union[str, None] = None,
        title: str = None,
        figsize: Tuple[int] = (7, 7),
        responseToColour: Union[Callable[[ResponseData], npt.number], str, None] = None,
    ) -> None:

        # compute dissimilarity
        rdm = self.compute_rdm(dissimilarityMetric)

        # compute 3D MDS:
        mdsCoords = mds(rdm, nComponents=3)

        # display:
        fig = plt.figure(figsize=figsize)
        ax: Axes3D = fig.add_subplot(projection="3d")

        # get colours / values based on colorFunction (if it exists)
        if responseToColour:
            if type(responseToColour) == str:
                colours = [
                    response.params[responseToColour] for response in self.responses
                ]
            elif callable(responseToColour):
                colours = [responseToColour(response) for response in self.responses]
        else:
            # use first parameter value as default if no colour value is provided
            colours = [
                next(iter(response.params.values())) for response in self.responses
            ]

        # create the scatter plot:
        scat = ax.scatter(
            mdsCoords[:, 0],
            mdsCoords[:, 1],
            mdsCoords[:, 2],
            c=colours,
            cmap=cmap,
            marker="o",
            s=100,
        )

        # set labels and title:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        # add colour bar:
        cb = fig.colorbar(scat, ax=ax, shrink=0.5, aspect=10)  # ?
        if cbarLabel:
            cb.set_label(cbarLabel)

        plt.show()
