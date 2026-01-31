"""
ResponseManager for batch generation of neural responses

This module provides the ResponseManager class which generates sets of responses
based on parameter combinations.

Classes:
    ResponseManager: Generates ResponseSets from parameter combinations
"""

from __future__ import annotations

import inspect
import itertools
from typing import Any

import numpy.typing as npt

from BuildYourOwnEmbedding.parameters.base import Parameter
from BuildYourOwnEmbedding.responses.base import ResponseFunction
from BuildYourOwnEmbedding.responses.data import ResponseData, ResponseSet


class ResponseManager:
    """
    Manager class for generating sets of neural responses.

    The ResponseManager takes a response class and parameter strategies,
    then generates ResponseSets containing responses for all combinations
    of parameter values.
    """

    def __init__(
        self,
        response_class: type[ResponseFunction] | tuple[type[ResponseFunction], ...],
        **parameters: Parameter,
    ) -> None:
        """
        Initialises response manager with a response type and parameters.

        Args:
            response_class: A single response class or a tuple of response classes
                to generate responses for.
            **parameters: Parameter strategies for each response parameter.
                The parameter names should match the response class constructor arguments.
        """
        self.response_class: type[ResponseFunction] = response_class
        self.parameters: dict[str, Parameter] = parameters

    def _get_response_parameters(self) -> list[str]:
        """
        Gets the parameter names from the response class' __init__ method.

        Returns:
            A list of parameter names.
        """
        return list(inspect.signature(self.response_class.__init__).parameters.keys())[1:]

    def _get_combinations(self) -> list[dict[str, Any]]:
        """
        Gets all combinations of parameter values.

        Returns:
            A list of dictionaries containing all combinations of parameter values.
        """
        param_values_dict = self.parameters
        param_names = self._get_response_parameters()
        param_combinations = itertools.product(
            *[p.get_values() for p in param_values_dict.values()]
        )
        return [dict(zip(param_names, combo, strict=False)) for combo in param_combinations]

    def _generate_response(
        self,
        x: npt.NDArray,
        response_params: dict[str, Any],
        noise_level: npt.number = 0,
    ) -> ResponseData:
        """
        Generates a single response based on the provided parameter values.

        Args:
            x: The input data for which the response is generated.
            response_params: Parameter values for the response.
            noise_level: The standard deviation of Gaussian noise to add.
                Defaults to 0 (no noise).

        Returns:
            A ResponseData object containing the response and parameters.
        """
        # Import here to avoid circular imports
        from BuildYourOwnEmbedding.responses.data import ResponseData

        response = self.response_class(**response_params)
        response_values = response(x, noise_level)
        return ResponseData(
            params=response_params,
            response=response_values,
            response_function=self.response_class.__name__,
            x=x,
        )

    def generate_responses(
        self,
        x: npt.NDArray,
        noise_level: npt.number = 0,
        num_samples: int = 1,
    ) -> ResponseSet:
        """
        Generates responses for each combination of parameter values.

        Args:
            x: The input data for which responses are generated.
            noise_level: The standard deviation of Gaussian noise to add.
                Defaults to 0 (no noise).
            num_samples: The number of times to sample each parameter combination.
                Defaults to 1.

        Returns:
            A ResponseSet object containing all generated responses.
        """
        # Import here to avoid circular imports
        from BuildYourOwnEmbedding.responses.data import ResponseSet

        param_combinations = self._get_combinations()
        responses = []
        for params in param_combinations:
            for _ in range(num_samples):
                responses.append(self._generate_response(x, params, noise_level))
        return ResponseSet(responses=responses)
