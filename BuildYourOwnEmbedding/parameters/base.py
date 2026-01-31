"""
Parameter classes for defining neural response parameter strategies.

This module provides various parameter classes that define how parameters
vary across a set of neural responses. Each parameter class implements a
specific strategy for generating parameter values.

Classes:
    Parameter: Abstract Base class for all parameter classes
    UniformRangeParameter: Generates uniformly spaced values
    RandomRangeParameter: Generates random values within a range
    FixedParameterSet: Uses a fixed set of values
    LogRangeParameter: Generates logarithmically spaced values
    ConstantParameter: Always outputs a single constant value
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Parameter(ABC):
    """
    Abstract base class for defining a parameter with a specific strategy to vary its value.
    """

    def __init__(self, numSamples: int) -> None:
        """
        Initialises the Parameter with the number of samples to generate.

        Args:
            numSamples: The number of samples to generate for this parameter.
        """
        self.numSamples = numSamples

    @abstractmethod
    def get_values(self) -> npt.NDArray:
        """
        Generates a list of values for this parameter.

        This method should be overridden in derived class to generate the output values.

        Returns:
            A numpy array of parameter values.
        """
        pass

    def __len__(self) -> int:
        """
        Gets the number of values that this parameter can take on.

        Returns:
            The number of values that the Parameter will take.
        """
        return self.numSamples


class UniformRangeParameter(Parameter):
    """
    Parameter class for generating a uniform set of values within a range.
    """

    def __init__(self, minValue: np.number, maxValue: np.number, numSamples: int) -> None:
        """
        Constructor for UniformRangeParameter class.

        This Parameter takes on a uniform range of values between `minValue` and
        `maxValue`, with `numSamples` evenly spaced points.

        Args:
            minValue: The minimum value that the parameter can take on.
            maxValue: The maximum value that the parameter can take on.
            numSamples: The number of different values that the parameter can take on.
        """
        super().__init__(numSamples)
        self.minValue: np.number = minValue
        self.maxValue: np.number = maxValue

    def get_values(self, dtype: npt.DTypeLike = np.float64) -> npt.NDArray:
        """
        Generates a linearly spaced array of values for the parameter.

        Args:
            dtype: The data type of the values in the output array. Defaults to np.float64.

        Returns:
            A linearly spaced numpy array of values.
        """
        return np.linspace(
            start=self.minValue, stop=self.maxValue, num=self.numSamples, dtype=dtype
        )


class RandomRangeParameter(Parameter):
    """
    Parameter class for generating a random set of values within a range.
    """

    def __init__(self, minValue: np.number, maxValue: np.number, numSamples: int) -> None:
        """
        Constructor for the RandomRangeParameter class.

        This Parameter takes on a random sample of `numSamples` values between
        `minValue` and `maxValue`.

        Args:
            minValue: The minimum value that the parameter can take on.
            maxValue: The maximum value that the parameter can take on.
            numSamples: The number of different values that the parameter can take on.
        """
        super().__init__(numSamples)
        self.minValue: np.number = minValue
        self.maxValue: np.number = maxValue

    def get_values(self, dtype: npt.DTypeLike = np.float64) -> npt.NDArray:
        """
        Generates a random array of values for the parameter.

        Args:
            dtype: The data type of the values in the output array. Defaults to np.float64.

        Returns:
            A numpy array of random values.
        """
        return np.random.uniform(self.minValue, self.maxValue, self.numSamples).astype(dtype)


class FixedParameterSet(Parameter):
    """
    Parameter class for generating a fixed set of values.
    """

    def __init__(
        self,
        values: npt.NDArray | list[np.number],
        dtype: npt.DTypeLike = np.float64,
    ) -> None:
        """
        Constructor for the FixedParameterSet class.

        This parameter takes on a specific sequence of values.

        Args:
            values: A list or array containing a sequence of values that the
                parameter should take on.
            dtype: The data type that the values should be returned as.
                Defaults to np.float64.
        """
        super().__init__(len(values))
        self.values = np.array(values, dtype=dtype)

    def get_values(self) -> npt.NDArray:
        """
        Generates an array of specified values for the parameter.

        Returns:
            An array of specified parameter values.
        """
        return self.values


class LogRangeParameter(Parameter):
    """
    Parameter class for generating a logarithmically scaling set of values within a range.
    """

    def __init__(self, minValue: np.number, maxValue: np.number, numSamples: int) -> None:
        """
        Constructor for the LogRangeParameter class.

        This Parameter takes on a range of `numSamples` values logarithmically
        spaced between `minValue` and `maxValue`.

        Args:
            minValue: The minimum value that the parameter can take on.
            maxValue: The maximum value that the parameter can take on.
            numSamples: The number of different values that the parameter can take on.
        """
        super().__init__(numSamples)
        self.minValue: np.number = minValue
        self.maxValue: np.number = maxValue

    def get_values(self, dtype: npt.DTypeLike = np.float64) -> npt.NDArray:
        """
        Generates a logarithmically spaced array of values for the parameter.

        Args:
            dtype: The data type of the values in the output array. Defaults to np.float64.

        Returns:
            A logarithmically spaced numpy array of values.
        """
        return np.geomspace(
            start=self.minValue, stop=self.maxValue, num=self.numSamples, dtype=dtype
        )


class ConstantParameter(Parameter):
    """
    Parameter class that always outputs a constant value.
    """

    def __init__(self, value: np.number) -> None:
        """
        Constructor for the ConstantParameter class.

        This Parameter always only ever takes on a single value.

        Args:
            value: The value that this parameter will take on.
        """
        super().__init__(1)  # only ever 1 value
        self.value: np.number = value

    def get_values(self, dtype: npt.DTypeLike = np.float64) -> npt.NDArray:
        """
        Generates an array of the specified value for the parameter.

        Args:
            dtype: The data type of the values in the output array. Defaults to np.float64.

        Returns:
            An array of size numSamples where all elements are the same value.
        """
        return np.full(dtype=dtype, shape=(1, 1), fill_value=self.value)
