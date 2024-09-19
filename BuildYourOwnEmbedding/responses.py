from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Union, Dict, Tuple, Any, Type, Callable, List
import inspect
import itertools
import mplcursors
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

from .parameters import Parameter
from .functional import inverse_correlation, plot_rdm, mds, pca_variance_explained, pca

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

    def __call__(self, x: npt.NDArray, noiseLevel: npt.number = 0, noiseType: str = 'Gaussian') -> npt.NDArray:
        """
        Compute the response by calling the evaluate method.

        This method allows the object to be used as a callable function, directly
        invoking the `evaluate` method when the instance is called with input data.

        Args:
            x (npt.NDArray):
                The input data for which the response is evaluated.
            noiseLevel (npt.number, optional):
                The standard deviation of Gaussian noise to be added to the response. Defaults to 0 (no noise).
            noiseType (str, optional):
                The type of noise to add. Defaults to 'Gaussian'.

        Returns:
            npt.NDArray:
                The response values computed by the `evaluate` method.
        """
        response = self.evaluate(x)
        
        if noiseLevel == 0:
            return response

        if noiseType == 'Gaussian':
            return response + np.random.normal(0, noiseLevel, response.shape)
            
        # Unknown noise type
        warnings.warn(f"Unknown noise type {noiseType}. Defaulting to Gaussian noise.")
        return response + np.random.normal(0, noiseLevel, response.shape)

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
        """
        Evaluates the CompositeResponse for a given input stimuli.

        Args:
            x (npt.NDArray): The input stimuli for which to generate a response.

        Returns:
            npt.NDArray: The response of the CompositeResponse function to the stimuli `x`.

        """
        return self.operation(self.response1(x), self.response2(x))

    def __str__(self) -> str:
        return (
            f"{self.operation.__name__}({str(self.response1)}, {str(self.response2)})"
        )


class GaussianResponse(ResponseFunction):
    """GaussianResponse class. Is the response function for a 1-dimensional gaussian
    response model.
    """

    def __init__(self, mean: npt.number, std: npt.number) -> None:
        """Constructor method for the GaussianResponse response function.

        Args:
            mean (npt.number): The mean value for the desired Gaussian response function.
            std (npt.number): The standard deviation of the desired Gaussian response function.
        """
        super().__init__(mean=mean, std=std)

    def evaluate(
        self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        """Evaluates the GaussianResponse response function for a given stimuli.

        Args:
            x (npt.NDArray): The stimuli for which to generate a response to.
            dtype (npt.DTypeLike, optional): The desired data type of the output. Defaults to np.float64.

        Returns:
            npt.NDArray: A numpy array containing the values of the Response to the stimulus `x`.
        """
        mean = self.params["mean"]
        std = self.params["std"]
        return dtype(
            np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        )


class SigmoidResponse(ResponseFunction):
    """SigmoidResponse class. Is the response function for a 1-dimensional sigmoid
    response model.
    """

    def __init__(self, alpha: npt.number, beta: npt.number) -> None:
        """Constructor for SigmoidResponse response function.

        Args:
            alpha (npt.number): The alpha value (steepness of slope) of the desired sigmoid response function.
            beta (npt.number): The beta value (pivot point) of the desired sigmoid response function.
        """
        super().__init__(alpha=alpha, beta=beta)

    def evaluate(
        self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        """Evaluates the SigmoidResponse response function for a given stimuli.

        Args:
            x (npt.NDArray): The stimuli for which to generate a response to.
            dtype (npt.DTypeLike, optional): The desired data type of the output. Defaults to np.float64.

        Returns:
            npt.NDArray: A numpy array containing the values of the Response to the stimulus `x`.
        """
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        return dtype(1 / (1 + np.exp(-alpha * (x - beta))))


class VonMisesResponse(ResponseFunction):
    """VonMisesResponse class. Is the response function for a 1-dimensional Von-mises
    response model.
    """

    def __init__(self, kappa: npt.number, theta: npt.number) -> None:
        """Constructor for the VonMisesResponse response function.

        Args:
            kappa (npt.number): The kappa value for the desired von-mises response.
            theta (npt.number): The theta value for the desired von-mises response.
        """
        super().__init__(kappa=kappa, theta=theta)

    def evaluate(
        self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        """Evaluates a Von-mises response for a given stimuli

        Args:
            x (npt.NDArray): The stimuli for which to generate a response to.
            dtype (npt.DTypeLike, optional): The desired data type of the output. Defaults to np.float64.

        Returns:
            npt.NDArray: A numpy array containing the values of the Response to the stimulus `x`.
        """
        kappa = self.params["kappa"]
        theta = self.params["theta"]
        return np.exp(kappa * np.cos(x - theta)) / (2 * np.pi * np.sinh(kappa))


class GaussianResponse2D(ResponseFunction):
    def __init__(
        self, xMean: npt.number, yMean: npt.number, xStd: npt.number, yStd: npt.number
    ) -> None:
        """Constructor for 2D Gaussian response function.

        Args:
            xMean (npt.number): The mean of the desired Gaussian response in the x-axis.
            yMean (npt.number): The standard deviation of the desired Gaussian response in the x-axis.
            xStd (npt.number): The mean of the desired Gaussian response in the y-axis.
            yStd (npt.number): The standard deviation of the desired Gaussian response in the y-axis.
        """
        super().__init__(xMean=xMean, yMean=yMean, xStd=xStd, yStd=yStd)

    def evaluate(
        self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        """Evaluates the GaussianResponse2D response function for a given stimuli.

        Args:
            x (npt.NDArray): The stimuli for which to generate a response to. This stimuli should be a 2-dimensional numpy array.
            dtype (npt.DTypeLike, optional): The desired data type of the output. Defaults to np.float64.

        Returns:
            npt.NDArray: A numpy array containing the values of the Response to the stimulus `x`.
        """
        xVals, yVals = x  # unpack
        return dtype(
            np.exp(
                -(
                    ((xVals - self.params["xMean"]) ** 2)
                    / (2 * self.params["xStd"] ** 2)
                    + ((yVals - self.params["yMean"]) ** 2)
                    / (2 * self.params["yStd"] ** 2)
                )
            )
        )


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
        self, x: npt.NDArray, responseParams: Dict[str, Any], noiseLevel: npt.number = 0
    ) -> ResponseData:
        """
        Generates a single response based on the provided parameter values

        Args:
            x (npt.NDArray): The input data for which the response is generated
            responseParams (Dict[str, Any]): Parameter values for the response
            noiseLevel (npt.number, optional):
                The standard deviation of Gaussian noise to be added to the response. Defaults to 0 (no noise).

        Returns:
            ResponseData: A ResponseData object containing the response values and the corresponding parameters.
        """
        response = self.responseClasses(**responseParams)
        responseValues = response(x, noiseLevel)
        return ResponseData(
            params=responseParams,
            response=responseValues,
            responseFunction=self.responseClasses.__name__,
            x=x,
        )

    def generate_responses(
        self, x: npt.NDArray, noiseLevel: npt.number = 0, numSamples: int = 1
    ) -> ResponseSet:
        """
        Generates responses for each combination of parameter values

        Args:
            x (npt.NDArray): The input data for which the response is generated
            noiseLevel (npt.number, optional):
                The standard deviation of Gaussian noise to be added to the response. Defaults to 0 (no noise).
            numSamples (int, optional):
                The number of times to sample each combination of parameters. Defaults to 1.

        Returns:
            ResponseSet: A ResponseSet object containing each response and the corresponding parameters.
        """
        paramCombinations = self._get_combinations()
        responses = []
        for params in paramCombinations:
            for _ in range(numSamples):
                responses.append(self._generate_response(x, params, noiseLevel))
        return ResponseSet(responses=responses)


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
    """
    A class for storing and analysing a set of neural responses.

    The ResponseSet class provides methods to compute representational
    dissimilarity matrices (RDM) and representational geodesic topological
    matrices (RGTM), as well as functions for visualising and analysing
    responses. It supports both 1D and 2D response visualisations and also
    allows interactive plots.

    Attributes:
        responses (List[ResponseData]): A list of ResponseData objects that store
            the neural responses and their associated parameters.
    """

    def __init__(self, responses: Union[None, List[ResponseData]] = None) -> None:
        """
        Initialise the ResponseSet with a list of response objects.

        Args:
            responses (Union[None, List[ResponseData]], optional): A list of ResponseData
            objects representing neural responses. Defaults to None.
        """
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
        """
        Computes the Representational Dissimilarity Matrix (RDM) for the stored responses

        Args:
            dissimilarityMetric (Callable[ [npt.ArrayLike, npt.ArrayLike], npt.number ], optional): A function
            to compute dissimilarity between two responses. Defaults to inverse_correlation.

        Returns:
            npt.ArrayLike: A square RDM where each entry (i, j) represents the
            dissimilarity between the i-th and j-th responses.
        """
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
        """
        Computes the Representational Geodesic Topological Matrix (RGTM).

        The RGTM is computed by applying a piecewise linear transformation on the
        dissimilarity matrix based on the provided lower and upper bounds.

        Args:
            lowerBound (npt.number): The lower bound for the transformation.
            upperBound (npt.number): The upper bound for the transformation.
            dissimilarityMetric (Callable[ [npt.ArrayLike, npt.ArrayLike], npt.number ], optional): A function
            to compute dissimilarity between two responses. Defaults to inverse_correlation.

        Returns:
            npt.ArrayLike: The computed RGTM matrix.
        """
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
        """
        Plots the Representational Geo-Topological Matrix (RGTM) with optional interactivity.

        Args:
            lowerBound (npt.number, optional): The lower bound for the RGTM transformation. Defaults to 0.
            upperBound (npt.number, optional): The upper bound for the RGTM transformation. Defaults to 1.
            dissimilarityMetric (Callable[ [npt.ArrayLike, npt.ArrayLike], npt.number ], optional): Function to compute dissimilarity between two responses. Defaults to inverse_correlation.
            interactive (bool, optional): Whether to include sliders for adjusting the bounds interactively. Defaults to True.
            labels (Union[List[str], None], optional): List of labels for the axes. Defaults to None.
            cmap (str, optional): Colourmap used for plotting. Defaults to "viridis".
            title (str, optional): Title of the plot. Defaults to "Representational Geo-Topological Matrix".
            figsize (Tuple[int], optional):  Size of the figure. Defaults to (7, 7).
            dissimilarityLabel (str, optional): Label for the colour bar. Defaults to "Dissimilarity".
        """
        rgtm: npt.NDArray = self.compute_rgtm(
            lowerBound, upperBound, dissimilarityMetric
        )

        # create the plot:
        fig, ax = plt.subplots(figsize=figsize)

        # display initial RGTM:
        heatmap = ax.imshow(rgtm, cmap=cmap, vmin=0, vmax=1)  # ?
        plt.colorbar(heatmap, ax=ax, label=dissimilarityLabel)
        ax.set_title(title)

        if labels:
            ax.set_xticks(labels)
            ax.set_yticks(labels)

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
        """
        Plots the Representational Dissimilarity Matrix (RDM) for the stored responses.

        Args:
            dissimilarityMetric (Callable[ [npt.ArrayLike, npt.ArrayLike], npt.number ], optional): Function to compute dissimilarity between two responses. Defaults to inverse_correlation.
            labels (Union[List[str], None], optional): Labels for the plot's x and y axes. Defaults to None.
            cmap (str, optional): Colourmap for the plot. Defaults to "viridis".
            title (str, optional): Title of the plot. Defaults to None.
            figsize (Tuple[int], optional): Size of the figure. Defaults to (7, 7).
            dissimilarityLabel (str, optional): Label for the colour bar. Defaults to "Dissimilarity".
        """
        rdm = self.compute_rdm(dissimilarityMetric)
        plot_rdm(rdm, labels, cmap, title, figsize, dissimilarityLabel)

    def plot_responses(
        self,
        figsize: Tuple[int] = (7, 7),
        xlabel: str = "Stimuli",
        ylabel: str = "Response",
        title: str = "Responses",
        grid: bool = False,
        hoverEffects: bool = True,  # if false will not add hover effect to plot
        *args,
        **kwargs
    ) -> None:
        """
        Plots the responses in 1D or 2D format.

        Higher dimensional plots are not currently supported.

        Args:
            figsize (Tuple[int], optional): Size of the figure. Defaults to (7, 7).
            xlabel (str, optional): Label for the x-axis. Defaults to "Stimuli".
            ylabel (str, optional): Label for the y-axis. Defaults to "Response".
            title (str, optional): Title of the plot. Defaults to "Responses".
            grid (bool, optional): Whether to display a grid on the plot. Defaults to False.
            hoverEffects (bool, optional): If True, adds hover effects to the plot. Defaults to True.

        Raises:
            NotImplementedError: If the ResponseData represents 3-dimensional or higher curves.
        """
        responseShape = list(self.responses[0].response.shape)

        # plot 1-d responses:
        if len(responseShape) == 1:
            self._plot_responses_1d(figsize, xlabel, ylabel, title, grid, hoverEffects, *args, **kwargs)
        # plot 2-d responses:
        elif len(responseShape) == 2:
            self._plot_responses_2d(
                figsize=figsize, xlabel=xlabel, ylabel=ylabel, title=title, *args, **kwargs
            )
        else:
            raise NotImplementedError(
                "plotting responses higher than two dimensions is not currently supported"
            )

    def _plot_responses_1d(
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

    def _plot_responses_2d(
        self,
        figsize: Tuple[int] = (7, 7),
        xlabel: str = "Stimuli 1",
        zlabel: str = "Stimuli 2",
        ylabel: str = "Response",
        title: str = "2D Tuning Curve Visualisation",
        cmap: str = "viridis",
        type: str = "heatmap"  # can be "heatmap" or "surfaceplot"
    ) -> None:
        if type == "surfaceplot":
            # plot the responses:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

            self.currentResponseIndex = 0

            # function to click through plot:
            def update_plot(responseIndex: int) -> None:
                ax.clear()
                response = self.responses[responseIndex].response
                x, y = self.responses[responseIndex].x

                ax.plot_surface(x, y, response, cmap=cmap)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_zlabel(zlabel)
                ax.set_title(title)
                plt.draw()

            update_plot(self.currentResponseIndex)

            # function for when next button is clicked:
            def next_response(e) -> None:
                self.currentResponseIndex = (self.currentResponseIndex + 1) % len(
                    self.responses
                )
                update_plot(self.currentResponseIndex)

            # function for when previous button is clicked:
            def prev_response(e) -> None:
                self.currentResponseIndex = (self.currentResponseIndex - 1) % len(
                    self.responses
                )
                update_plot(self.currentResponseIndex)

            # add buttons for interactivity:
            axNext = plt.axes([0.8, 0.05, 0.1, 0.075])
            btnNext = Button(axNext, "Next")
            btnNext.on_clicked(next_response)
            axPrev = plt.axes([0.7, 0.05, 0.1, 0.075])
            btnPrev = Button(axPrev, "Previous")
            btnPrev.on_clicked(prev_response)
            plt.show()
        elif type == "heatmap":
            # Plot the responses as a heatmap
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            self.currentResponseIndex = 0
            from matplotlib.colorbar import Colorbar
            colourbar : Colorbar = None
            # function to update heatmap plot
            def update_plot(responseIndex: int) -> None:
                nonlocal colourbar
                # fig.clear()
                # ax = fig.add_subplot(111)
                response = self.responses[responseIndex].response
                x, y = self.responses[responseIndex].x

                c = ax.imshow(response, cmap=cmap, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                # colourbar = fig.colorbar(c, ax=ax)
                plt.draw()

            update_plot(self.currentResponseIndex)

            # function for when next button is clicked:
            def next_response(e) -> None:
                self.currentResponseIndex = (self.currentResponseIndex + 1) % len(
                    self.responses
                )
                update_plot(self.currentResponseIndex)

            # function for when previous button is clicked:
            def prev_response(e) -> None:
                self.currentResponseIndex = (self.currentResponseIndex - 1) % len(
                    self.responses
                )
                update_plot(self.currentResponseIndex)

            # add buttons for interactivity:
            axNext = plt.axes([0.8, 0, 0.1, 0.075])
            btnNext = Button(axNext, "Next")
            btnNext.on_clicked(next_response)
            axPrev = plt.axes([0.7, 0, 0.1, 0.075])
            btnPrev = Button(axPrev, "Previous")
            btnPrev.on_clicked(prev_response)
            plt.show()

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
        """
        Plots a 3D Multidimensional Scaling (MDS) scatter plot for the stored responses.

        This method computes a 3D MDS embedding from the dissimilarity matrix of the responses
        and visualises the responses in a 3D scatter plot. The color of each point in the scatter
        plot can be customized based on response parameters or a user-defined function.

        Args:
            dissimilarityMetric (Callable[ [npt.ArrayLike, npt.ArrayLike], npt.number ], optional):
                A function to compute the dissimilarity between two responses. Defaults to inverse_correlation.
            xlabel (str, optional): Label for the x-axis of the MDS plot. Defaults to "MDS Dimension 1".
            ylabel (str, optional): Label for the y-axis of the MDS plot. Defaults to "MDS Dimension 2".
            zlabel (str, optional): Label for the z-axis of the MDS plot. Defaults to "MDS Dimension 3".
            cmap (str, optional): The colourmap used for coloring the scatter points. Defaults to "viridis".
            cbarLabel (Union[str, None], optional): Label for the colour bar. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            figsize (Tuple[int], optional):  The size of the figure. Defaults to (7, 7).
            responseToColour (Union[Callable[[ResponseData], npt.number], str, None], optional):
                A function or string to define the color mapping for the scatter points. If
                a string is passed, it should be the name of a parameter in the responses.
                If a callable is passed, it should be a function that maps a `ResponseData`
                object to a numeric value for coloring the points. Defaults to None.
        """
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

    def plot_2D_mds(
        self,
        dissimilarityMetric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        xlabel: str = "MDS Dimension 1",
        ylabel: str = "MDS Dimension 2",
        cmap: str = "viridis",
        cbarLabel: Union[str, None] = None,
        title: str = None,
        figsize: Tuple[int] = (7, 7),
        responseToColour: Union[Callable[[ResponseData], npt.number], str, None] = None,
    ) -> None:
        """
        Plots a 2D Multidimensional Scaling (MDS) scatter plot for the stored responses.

        This method computes a 2D MDS embedding from the dissimilarity matrix of the responses
        and visualises the responses in a 2D scatter plot. The color of each point in the scatter
        plot can be customized based on response parameters or a user-defined function.

        Args:
            dissimilarityMetric (Callable[ [npt.ArrayLike, npt.ArrayLike], npt.number ], optional):
                A function to compute the dissimilarity between two responses. Defaults to inverse_correlation.
            xlabel (str, optional): Label for the x-axis of the MDS plot. Defaults to "MDS Dimension 1".
            ylabel (str, optional): Label for the y-axis of the MDS plot. Defaults to "MDS Dimension 2".
            cmap (str, optional): The colourmap used for coloring the scatter points. Defaults to "viridis".
            cbarLabel (Union[str, None], optional): Label for the colour bar. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            figsize (Tuple[int], optional):  The size of the figure. Defaults to (7, 7).
            responseToColour (Union[Callable[[ResponseData], npt.number], str, None], optional):
                A function or string to define the color mapping for the scatter points. If
                a string is passed, it should be the name of a parameter in the responses.
                If a callable is passed, it should be a function that maps a `ResponseData`
                object to a numeric value for coloring the points. Defaults to None.
        """
        # compute dissimilarity
        rdm = self.compute_rdm(dissimilarityMetric)

        # compute 3D MDS:
        mdsCoords = mds(rdm, nComponents=2)

        # display:
        fig = plt.figure(figsize=figsize)
        ax = fig.figure.subplots(1, 1)

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
            c=colours,
            cmap=cmap,
            marker="o",
            s=100,
        )

        # set labels and title:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # add colour bar:
        cb = fig.colorbar(scat, ax=ax, shrink=0.5, aspect=10)  # ?
        if cbarLabel:
            cb.set_label(cbarLabel)

        plt.show()

    def plot_PCA_variance_explained(
        self,
        dissimilarityMetric: Callable[
            [npt.ArrayLike, npt.ArrayLike], npt.number
        ] = inverse_correlation,
        xlabel: str = "Number of Principal Components",
        ylabel: str = "Variance Explained",
        title: str = "Variance Explained vs. Number of Principal Components",
        figsize: Tuple[int] = (7, 7),
        grid: bool = False,
    ) -> None:
        """
        Plots the variance explained by each principal component from PCA on the dissimilarity matrix.

        This method computes the Principal Component Analysis (PCA) of the dissimilarity matrix
        and visualises the variance explained by each principal component in a line plot. This
        provides insight into how much variance each component captures, helping to determine
        how many components are needed for sufficient variance representation.

        Args:
            dissimilarityMetric (Callable[ [npt.ArrayLike, npt.ArrayLike], npt.number ], optional):
                A function to compute the dissimilarity between two responses. Defaults to inverse_correlation.
            xlabel (str, optional): Label for the x-axis. Defaults to 'Number of Principal Components'.
            ylabel (str, optional): Label for the y-axis. Defaults to 'Variance Explained'.
            title (str, optional): Title of the plot. Defaults to 'Variance Explained vs. Number of Principal Components'.
            figsize (Tuple[int], optional): Size of the figure. Defaults to (7, 7).
            grid (bool, optional): Whether to show grid lines on the plot. Defaults to False.
        """
        # compute dissimilarity
        rdm = self.compute_rdm(dissimilarityMetric)

        # compute PCA variance explained:
        pcaVarianceExplained = pca_variance_explained(rdm)

        plt.figure(figsize=figsize)
        plt.plot(
            range(1, len(pcaVarianceExplained) + 1), pcaVarianceExplained, marker="o"
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(grid)
        plt.show()

    def plot_2D_PCA(
        self,
        xlabel: str = "PC 1",
        ylabel: str = "PC 2",
        title: str = "2D PCA of Embedding",
        figsize: Tuple[int] = (7, 7),
    ) -> None:
        curves = np.array([r.response for r in self.responses])
        pcs = pca(curves, nComponents=2)
        
        plt.figure(figsize=figsize)
        plt.scatter(pcs[:, 0], pcs[:, 1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()
        