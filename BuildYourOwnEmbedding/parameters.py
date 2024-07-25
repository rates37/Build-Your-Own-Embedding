from __future__ import annotations
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import List, Any, Union


class Parameter(ABC):
    """
    Abstract base class for defining a parameter with a specific strategy to vary its value
    """
    
    def __init__(self, numSamples: int) -> None:
        """
        Initialises the Parameter with the number of samples to generate

        Args:
            numSamples (int): The number of samples to generate for this parameter.
        """
        self.numSamples = numSamples

    @abstractmethod
    def get_values(self) -> npt.NDArray:
        """
        Generates a list of values for this parameter.
        This method should be overriden in derived class to generate the output values.

        Returns:
            npt.NDArray: A list of parameter values
        """
        pass

    def __len__(self) -> int:
        return self.numSamples

class UniformRangeParameter(Parameter):
    """
    Parameter class for generating a uniform set of values within a range.
    """

    def __init__(self, minValue: np.number, maxValue: np.number, numSamples: int) -> None:
        super().__init__(numSamples)
        self.minValue: np.number = minValue
        self.maxValue: np.number = maxValue

    def get_values(
        self, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        """
        Generates a linearly spaced array of values for the parameter

        Args:
            dtype (np.DTypeLike, optional): The data type of the values in the output array. Defaults to np.float64.

        Returns:
            npt.NDArray: A linearly spaced numpy array of values
        """
        return np.linspace(
            start=self.minValue, stop=self.maxValue, num=self.numSamples, dtype=dtype
        )


class RandomRangeParameter(Parameter):
    """
    Parameter class for generating a random set of values within a range.
    """

    def __init__(self, minValue: np.number, maxValue: np.number, numSamples: int) -> None:
        super().__init__(numSamples)
        self.minValue: np.number = minValue
        self.maxValue: np.number = maxValue

    def get_values(
        self, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        """
        Generates a random array of values for the parameter

        Args:
            dtype (np.DTypeLike, optional): The data type of the values in the output array. Defaults to np.float64.

        Returns:
            npt.NDArray: A numpy array of random values
        """
        return np.random.uniform(self.minValue, self.maxValue, self.numSamples).astype(dtype)


class FixedParameterSet(Parameter):
    """
    Parameter class for generating a fixed set of values.
    """

    def __init__(
        self,
        values: Union[npt.NDArray, List[np.number]],
        dtype: npt.DTypeLike = np.float64,
    ) -> None:
        super().__init__(len(values))
        self.values = np.array(values, dtype=dtype)

    def get_values(self) -> npt.NDArray:
        """
        Generates an array of specified values for the parameter

        Returns:
            npt.NDArray: An array of specified parameter values
        """
        return self.values


class LogRangeParameter(Parameter):
    """
    Parameter class for generating a logarithmically scaling set of values within a range.
    """

    def __init__(self, minValue: np.number, maxValue: np.number, numSamples: int) -> None:
        super().__init__(numSamples)
        self.minValue: np.number = minValue
        self.maxValue: np.number = maxValue

    def get_values(
        self, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        """
        Generates a logarithmically spaced array of values for the parameter

        Args:
            dtype (np.DTypeLike, optional): The data type of the values in the output array. Defaults to np.float64.

        Returns:
            npt.NDArray: A logarithmically spaced numpy array of values
        """
        return np.geomspace(
            start=self.minValue, stop=self.maxValue, num=self.numSamples, dtype=dtype
        )



class ConstantParameter(Parameter):
    """
    Parameter class that always outputs a constant value.
    """
    def __init__(self, value: np.number) -> None:
        super().__init__(1)  # only ever 1 value
        self.value: np.number = value

    def get_values(
        self, dtype: npt.DTypeLike = np.float64
    ) -> npt.NDArray:
        """
        Generates an array of the specified value for the parameter

        Args:
            dtype (np.DTypeLike, optional): The data type of the values in the output array. Defaults to np.float64.

        Returns:
            npt.NDArray: An array of size numSamples where all elements are the same value.
        """
        return np.full(
            dtype=dtype, shape=(1,1), fill_value=self.value
        )
    
    
    pass

if __name__ == "__main__":
    # testing
    u = LogRangeParameter(1, 1000, 4)
    print(u.get_values())
    u = FixedParameterSet([12.12, 112, 1, 1, 432, 4346, 62])
    print(u.get_values())
    u = RandomRangeParameter(1, 1000, 5)
    print(u.get_values())
