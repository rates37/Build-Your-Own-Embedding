"""
Base classes for defining neural response functions.

This module provides the abstract base classes for creating neural response
functions that can be evaluated based on input stimuli.

Classes:
    CompositeResponse: Combines two response functions with an operation
    ResponseFunction: Abstract base class for all response functions
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt


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
        params: A dictionary containing response-specific parameters used
            to define the behavior of the response function.

    Methods:
        evaluate: Abstract method to evaluate the neural response for a given input.
        __call__: Calls the `evaluate` method to compute the response.
        __add__: Combines this response with another using addition.
        __sub__: Combines this response with another using subtraction.
        __mul__: Combines this response with another using multiplication.
        __str__: Returns a string representation of the response function.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialises the response function with specific parameters.

        This constructor accepts named arguments for the parameters that define
        the behavior of the neural response function. These parameters are stored
        in a dictionary for later use by the response function.

        Args:
            **kwargs: Named arguments representing the parameters of the response function.

        Notes:
            Subclasses should provide specific parameter requirements, which will be
            documented in their respective class definitions.
        """
        self.params: dict[str, Any] = kwargs

    @abstractmethod
    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the response of the neuron for a given input.

        This abstract method must be implemented by subclasses to define how the
        response function computes output values based on input data.

        Args:
            x: The input data for which the response is evaluated. This should be
                a Numpy array representing the stimulus or feature input to the neuron.

        Returns:
            A Numpy array representing the evaluated response values corresponding
            to the input data.

        Raises:
            NotImplementedError: This method must be overridden by subclasses.
        """
        pass

    def __call__(
        self, x: npt.NDArray, noiseLevel: npt.number = 0, noiseType: str = "Gaussian"
    ) -> npt.NDArray:
        """
        Compute the response by calling the evaluate method.

        This method allows the object to be used as a callable function, directly
        invoking the `evaluate` method when the instance is called with input data.

        Args:
            x: The input data for which the response is evaluated.
            noiseLevel: The standard deviation of noise to be added to the response.
                Defaults to 0 (no noise).
            noiseType: The type of noise to add. Currently supports 'Gaussian'.
                Defaults to 'Gaussian'.

        Returns:
            The response values computed by the `evaluate` method, optionally with
            added noise.
        """
        response = self.evaluate(x)

        if noiseLevel == 0:
            return response

        if noiseType == "Gaussian":
            return response + np.random.normal(0, noiseLevel, response.shape)

        # Unknown noise type
        warnings.warn(f"Unknown noise type {noiseType}. Defaulting to Gaussian noise.", stacklevel=2)
        return response + np.random.normal(0, noiseLevel, response.shape)

    def __add__(self, other: ResponseFunction) -> CompositeResponse:
        """
        Add another response function to this response function.

        This method allows the creation of a composite response by adding the outputs
        of two different response functions.

        Args:
            other: The other response function to be added to this response.

        Returns:
            A new `CompositeResponse` instance representing the combined response.
        """
        return CompositeResponse(self, other, np.add)

    def __sub__(self, other: ResponseFunction) -> CompositeResponse:
        """
        Subtract another response function from this response function.

        Args:
            other: The other response function to be subtracted from this response.

        Returns:
            A new `CompositeResponse` instance representing the difference.
        """
        return CompositeResponse(self, other, np.subtract)

    def __mul__(self, other: ResponseFunction) -> CompositeResponse:
        """
        Multiply this response function by another response function.

        Args:
            other: The other response function to multiply with this response.

        Returns:
            A new `CompositeResponse` instance representing the product.
        """
        return CompositeResponse(self, other, np.multiply)

    def __str__(self) -> str:
        """
        Return a string representation of the response function.

        Returns:
            A string representation including the class name and parameters.
        """
        return f"{type(self).__name__}, {self.params}"


class CompositeResponse(ResponseFunction):
    """
    Composite response class that combines two responses using a specified operation.

    This class allows creating complex response functions by combining simpler ones
    using mathematical operations like addition, subtraction, or multiplication.
    """

    def __init__(
        self,
        response1: ResponseFunction,
        response2: ResponseFunction,
        operation: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    ) -> None:
        """
        Initialises a composite response function.

        Args:
            response1: First response function.
            response2: Second response function.
            operation: A callable that takes two arrays and returns their combination
                (e.g., np.add, np.subtract, np.multiply).
        """
        super().__init__()
        self.response1: ResponseFunction = response1
        self.response2: ResponseFunction = response2
        self.operation = operation

    def evaluate(self, x: npt.NDArray) -> npt.NDArray:
        """
        Evaluates the CompositeResponse for a given input stimulus.

        Args:
            x: The input stimuli for which to generate a response.

        Returns:
            The response of the CompositeResponse function to the stimuli `x`.
        """
        return self.operation(self.response1(x), self.response2(x))

    def __str__(self) -> str:
        """Return a string representation of the composite response."""
        return f"{self.operation.__name__}({str(self.response1)}, {str(self.response2)})"
