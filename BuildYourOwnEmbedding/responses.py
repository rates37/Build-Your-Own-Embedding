from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Union, Dict, Tuple, Any, Type, Callable, List
from .parameters import Parameter
import inspect
import itertools

##! =====================================
##!             Responses:
##! =====================================


class ResponseFunction(ABC):
    """
    Base class for defining neural responses.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialises response class with response-specific parameters.

        Args:
            **kwargs (Any): Named arguments for parameter values.
        """
        self.params: Dict = kwargs # Store all parameters for general use

    @abstractmethod
    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the response of the neuron for a given input

        Args:
            x (npt.NDArray): The input data

        Returns:
            npt.NDArray: The response values
        """
        pass

    def __call__(self, x: npt.NDArray) -> npt.NDArray:
        return self.evaluate(x)

    def __add__(self, other: ResponseFunction) -> CompositeResponse:
        """
        Add another response to this response to create a composite response.

        Args:
            other (ResponseFunction): The other response to add.

        Returns:
            CompositeResponse: A new composite response representing the addition of this response and the other response.
        """
        return CompositeResponse(self, other, np.add)

    def __sub__(self, other: ResponseFunction) -> CompositeResponse:
        """
        Subtract another response from this response to create a composite response.

        Args:
            other (ResponseFunction): The other response to subtract.

        Returns:
            CompositeResponse: A new composite response representing the subtraction of the other response from this response.
        """
        return CompositeResponse(self, other, np.subtract)

    def __mul__(self, other: ResponseFunction) -> CompositeResponse:
        """
        Multiply by another response to create a composite response.

        Args:
            other (ResponseFunction): The other response to multiply with.

        Returns:
            CompositeResponse: A new composite response representing the multiplication of this response and the other response.
        """
        return CompositeResponse(self, other, np.multiply)

    def __str__(self) -> str:
        return f"{type(self).__name__}, {self.params}"

    # todo: divide operation as well


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
        return list(inspect.signature(self.responseClasses.__init__).parameters.keys())[1:]
    
    def _get_combinations(self) -> List[Dict[str, Any]]:
        """
        Gets all combinations of parameter values based on the input parameters

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing all combinations of parameter values
        """
        # paramValuesDict = {str(i): param.get_values() for i, param in enumerate(self.parameters)} #!
        paramValuesDict = self.parameters
        paramNames = self._get_response_parameters()
        paramCombinations = itertools.product(*[p.get_values() for p in [*paramValuesDict.values()]])
        return [dict(zip(paramNames, paramCombo)) for paramCombo in paramCombinations]

    def _generate_response(self, x: npt.NDArray, responseParams: Dict[str, Any]) -> ResponseData:
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
        return ResponseData(params=responseParams, response=responseValues, responseFunction=self.responseClasses.__name__)
    
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
        params (Dict[str, np.ndarray]): The parameters used for generating the response.
        response (np.ndarray): The response values.
        responseFunction (str): The name of the Response Function that generated this Response
    """

    params: Dict[str, npt.NDArray]
    response: npt.NDArray
    responseFunction: str


if __name__ == "__main__":
    a = VonMisesResponse(1, 0)
    b = SigmoidResponse(1,1)
    x = np.linspace(-10, 10, 1000)
    import matplotlib.pyplot as plt
    from parameters import *
    
    # example of plotting response functions directly:
    # plt.plot(x, (a)(x), label="von-mises")
    # plt.plot(x, (b)(x), label="sigmoid")
    # plt.plot(x, (a+b)(x), label="von-mises + sigmoid")  # composite function support
    # plt.plot(x, (a-b)(x), label="von-mises - sigmoid")
    # plt.plot(x, (a*b)(x), label="von-mises * sigmoid")
    # plt.legend()
    # plt.show()
    
    # example of generating large sets of data by varying parameters:
    meanParam: Parameter = UniformRangeParameter(-2, 2, numSamples=4)
    stdParam: Parameter = FixedParameterSet([0.5, 1.0, 1.5])
    gaussianResponseManager = ResponseManager(GaussianResponse, mean=meanParam, std=stdParam) 
    gaussianResponses = gaussianResponseManager.generate_responses(x)   
    for response in gaussianResponses:
        plt.plot(x, response.response)
    plt.show()