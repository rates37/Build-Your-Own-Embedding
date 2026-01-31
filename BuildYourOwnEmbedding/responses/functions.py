"""
Concrete neural response function implementations

This module provides various neural response function classes that model different
types of neuronal tuning curves commonly used in neuroscience.

Classes:
    GaussianResponse: 1D Gaussian tuning curve
    SigmoidResponse: 1D Sigmoidal response curve
    VonMisesResponse: 1D Von Mises (circular Gaussian) response
    GaussianResponse2D: 2D Gaussian response
"""

import numpy as np
import numpy.typing as npt

from BuildYourOwnEmbedding.responses.base import ResponseFunction


class GaussianResponse(ResponseFunction):
    """
    GaussianResponse class for a 1-dimensional Gaussian response model.

    This models neurons with bell-shaped tuning curves, commonly seen in
    sensory neurons that respond preferentially to specific stimulus values.
    """

    def __init__(self, mean: np.number, std: np.number) -> None:
        """
        Constructor for the GaussianResponse response function.

        Args:
            mean: The preferred stimulus value (peak of the Gaussian).
            std: The tuning width (standard deviation of the Gaussian).
        """
        super().__init__(mean=mean, std=std)

    def evaluate(self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64) -> npt.NDArray:
        """
        Evaluates the GaussianResponse for a given stimulus.

        Args:
            x: The stimuli for which to generate a response.
            dtype: The desired data type of the output. Defaults to np.float64.

        Returns:
            A numpy array containing the response values.
        """
        mean = self.params["mean"]
        std = self.params["std"]
        return dtype(np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi)))


class SigmoidResponse(ResponseFunction):
    """
    SigmoidResponse class for a 1-dimensional sigmoid response model.

    This models neurons with monotonic tuning curves, commonly seen in
    neurons that encode stimulus intensity or direction.
    """

    def __init__(self, alpha: np.number, beta: np.number) -> None:
        """
        Constructor for SigmoidResponse response function.

        Args:
            alpha: The steepness of the sigmoid slope.
            beta: The pivot point (stimulus value at half-maximum response).
        """
        super().__init__(alpha=alpha, beta=beta)

    def evaluate(self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64) -> npt.NDArray:
        """
        Evaluates the SigmoidResponse for a given stimulus.

        Args:
            x: The stimuli for which to generate a response.
            dtype: The desired data type of the output. Defaults to np.float64.

        Returns:
            A numpy array containing the response values.
        """
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        return dtype(1 / (1 + np.exp(-alpha * (x - beta))))


class VonMisesResponse(ResponseFunction):
    """
    VonMisesResponse class for a 1-dimensional Von Mises response model.

    This models neurons with circular/periodic tuning, commonly seen in
    orientation-selective neurons in visual cortex.
    """

    def __init__(self, kappa: np.number, theta: np.number) -> None:
        """
        Constructor for the VonMisesResponse response function.

        Args:
            kappa: The concentration parameter (higher = narrower tuning).
            theta: The preferred orientation/direction.
        """
        super().__init__(kappa=kappa, theta=theta)

    def _I0(self, kappa: np.number, num_terms: int = 50) -> np.number:
        """
        Modified Bessel function of the first kind of order 0.

        Calculated using a Maclaurin series expansion.

        Args:
            kappa: The argument to the Bessel function.
            num_terms: Number of terms in the series expansion.

        Returns:
            The value of I_0(kappa).
        """
        result = 0
        cumulative_factorial = 1

        for i in range(1, num_terms + 1):
            cumulative_factorial *= i
            result += (kappa / 2) ** (2 * i) / (cumulative_factorial**2)
        return result

    def evaluate(self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64) -> npt.NDArray:
        """
        Evaluates a Von Mises response for a given stimulus.

        Args:
            x: The stimuli (angles) for which to generate a response.
            dtype: The desired data type of the output. Defaults to np.float64.

        Returns:
            A numpy array containing the response values.
        """
        kappa = self.params["kappa"]
        theta = self.params["theta"]
        return np.exp(kappa * np.cos(x - theta)) / (2 * np.pi * self._I0(kappa))


class GaussianResponse2D(ResponseFunction):
    """
    GaussianResponse2D class for a 2-dimensional Gaussian receptive field.

    This models neurons with 2D Gaussian receptive fields, commonly seen in
    visual neurons with localised spatial receptive fields.
    """

    def __init__(
        self,
        x_mean: np.number,
        y_mean: np.number,
        x_std: np.number,
        y_std: np.number,
    ) -> None:
        """
        Constructor for 2D Gaussian response function.

        Args:
            x_mean: The mean (center) of the Gaussian in the x-axis.
            y_mean: The mean (center) of the Gaussian in the y-axis.
            x_std: The standard deviation in the x-axis.
            y_std: The standard deviation in the y-axis.
        """
        super().__init__(x_mean=x_mean, y_mean=y_mean, x_std=x_std, y_std=y_std)

    def evaluate(self, x: npt.NDArray, dtype: npt.DTypeLike = np.float64) -> npt.NDArray:
        """
        Evaluates the GaussianResponse2D for a given 2D stimulus.

        Args:
            x: The stimuli, expected to be a 2D array where x[0] contains
                x-coordinates and x[1] contains y-coordinates.
            dtype: The desired data type of the output. Defaults to np.float64.

        Returns:
            A numpy array containing the response values.
        """
        x_vals, y_vals = x  # unpack
        return dtype(
            np.exp(
                -(
                    ((x_vals - self.params["x_mean"]) ** 2) / (2 * self.params["x_std"] ** 2)
                    + ((y_vals - self.params["y_mean"]) ** 2) / (2 * self.params["y_std"] ** 2)
                )
            )
        )
