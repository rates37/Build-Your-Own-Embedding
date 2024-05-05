"""
Implements Tuning Curve Generation Features of package

Author: Satya Jhaveri

"""

__author__ = "Satya Jhaveri"

##! =====================================
##!              Imports:
##! =====================================
# Mathematical / Numerical Libraries:
import numpy as np
import random

# Type hinting:
from typing import List, Union, Tuple, Callable, Iterable

# import parameters:
from parameters import Parameter



##! =====================================
##!     1-dimensional Tuning Curves:
##! =====================================

def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Creates a simple 1 dimensional Gaussian probability distribution

    Args:
        x (np.ndarray): A numpy array of points to evaluate the distribution at
        mu (float): The mean (centre point) of the distribution
        sigma (float): The stdev of the gaussian distribution

    Returns:
        np.ndarray: A numpy array representing the gaussian distribution evaluated at the input points x
    """
    return (np.exp(-((x - mu)**2) / (2 * sigma**2))) / (np.sqrt(2 *  np.pi) * sigma)


def gaussian_tuning_function(x: np.ndarray, mu: float, sigma: float, maxResponse: float=1, minResponse: float=0) -> np.ndarray:
    """Creates a shifted scaled Gaussian distribution

    Args:
        x (np.ndarray): A numpy array of points to evaluate at
        mu (float): The centre point of the distribution
        sigma (float): The stdev of the distribution
        maxResponse (float, optional): The maximum value of the response function. Defaults to 1.
        minResponse (float, optional): The minimum value of the response function. Defaults to 0.

    Returns:
        np.ndarray: A numpy array representing the gaussian distribution evaluated at the input points x
    """
    # compute gaussian PDF based on input:
    dist = gaussian(x, mu, sigma)
    
    # scale + shift to match min and max response range:
    dist = (dist/max(dist)) * (maxResponse - minResponse) + minResponse
    
    return dist


def sigmoid_tuning(x: np.ndarray, steepness: float, pivot: float, maxResponse: float=0, minResponse: float=1) -> np.ndarray:
    """Creates a shifted scaled Sigmoid distribution

    Args:
        x (np.ndarray): A numpy array of points to evaluate at
        steepness (float): The steepness of the step
        pivot (float): The pivot point (point at which the distribution is centred at)
        maxResponse (float, optional): The maximum value of the response function. Defaults to 1.
        minResponse (float, optional): The minimum value of the response function. Defaults to 0.

    Returns:
        np.ndarray: A numpy array representing the sigmoid distribution evaluated at the input points x
    """
    # create initial distribution:
    dist: np.ndarray =  1 / (1 + np.exp(-steepness * (x - pivot)))
    
    # scale + shift to match min and max response range:
    dist = (dist/max(dist)) * (maxResponse - minResponse) + minResponse
    
    return dist


def von_misses_tuning_function(x: np.ndarray, kappa: float, theta: float, maxResponse: float=1, minResponse: float=0) -> np.ndarray:
    """Creates a shifted scaled von-mises distribution

    Args:
        x (np.ndarray): A numpy array of points to evaluate at
        kappa (float): The concentration parameter
        theta (float): The ideal angle (with maximum response)
        maxResponse (float, optional): The maximum value of the response function. Defaults to 1.
        minResponse (float, optional): The minimum value of the response function. Defaults to 0.

    Returns:
        np.ndarray: A numpy array representing the von-mises distribution evaluated at the input points x
    """
    # compute von-mises PDF based on input:
    dist = np.exp(kappa * np.cos(x - theta)) / (2 * np.pi * np.sinh(kappa))
    
    # scale + shift to match min and max response range:
    dist = (dist/max(dist)) * (maxResponse - minResponse) + minResponse
    
    return dist



##! =====================================
##!     Random Generation for 1D:
##! =====================================


def generate_random_curves(
        tuning_function: Callable[[np.ndarray, float], np.ndarray],  # the tuning function to use
        x: np.ndarray,  # a 1D numpy array of input values
        nCurves: int,  # the number of output curves to randomly generate
        parameters: List[Union[float, Tuple[float, float]]],  # a list of floats or 2-tuples of floats (single float = parameter held constant, tuple = range of values for that parameter)
    ) -> List[np.ndarray]:
    
    generatedCurves = []
    
    for _ in range(nCurves):
        curveParams = [n if type(n) in [float, int] else random.uniform(*n) for n in parameters]
        try:
            generatedCurves.append(tuning_function(x, *curveParams))
        except TypeError:  # will occur if the incorrect amount of parameters have been provided for the chosen tuning_function
            raise TypeError(f"Incorrect number of parameters provided for {tuning_function.__name__}!")
        except Exception as e:
            raise ValueError(f"Using parameters {str(curveParams)} with {tuning_function.__name__} resulted in the following exception:\n{str(e)}\nCheck your parameters are all valid for the chosen tuning function!")
    
    return generatedCurves


def generate_random_sigmoid_curves_old(x: np.ndarray, nCurves: int, parameters: List[Union[float, Tuple[float, float]]]) -> List[np.ndarray]:
    return generate_random_curves(sigmoid_tuning, x, nCurves, parameters)


def generate_random_gaussian_curves_old(x: np.ndarray, nCurves: int, parameters: List[Union[float, Tuple[float, float]]]) -> List[np.ndarray]:
    return generate_random_curves(gaussian_tuning_function, x, nCurves, parameters)


def generate_random_von_mises_curves(x: np.ndarray, nCurves: int, parameters: List[Union[float, Tuple[float, float]]]) -> List[np.ndarray]:
    return generate_random_curves(von_misses_tuning_function, x, nCurves, parameters)




##! =====================================
##!     New Random Generation for 1D:
##!      Using Parameter Class
##! =====================================
def conv_to_iter(obj) -> Iterable:
    try:
        iter(obj)
        return obj
    except:
        return [obj]

def generate_random_sigmoid_curves(
        x: np.ndarray,
        steepness: Union[float, Parameter],
        pivot:  Union[float, Parameter],
        maxResponse:  Union[float, Parameter],
        minResponse:  Union[float, Parameter]
    ) -> List[np.ndarray]:
    # convert all parameters to iterables:
    steepness = conv_to_iter(steepness)
    pivot = conv_to_iter(pivot)
    maxResponse = conv_to_iter(maxResponse)
    minResponse = conv_to_iter(minResponse)
    
    # generate outputs:
    outputs = []
    for s in steepness:
        for p in pivot:
            for mi in minResponse:
                for ma in maxResponse:
                    outputs.append(sigmoid_tuning(x, s, p ,ma, mi))
    return outputs

def generate_random_gaussian_curves(
        x: np.ndarray,
        mu: Union[float, Parameter],
        sigma:  Union[float, Parameter],
        maxResponse:  Union[float, Parameter],
        minResponse:  Union[float, Parameter]
    ) -> List[np.ndarray]:
    # convert all parameters to iterables:
    mu = conv_to_iter(mu)
    sigma = conv_to_iter(sigma)
    maxResponse = conv_to_iter(maxResponse)
    minResponse = conv_to_iter(minResponse)
    
    # generate outputs:
    outputs = []
    for m in mu:
        for s in sigma:
            for mi in minResponse:
                for ma in maxResponse:
                    outputs.append(gaussian_tuning_function(x, m, s ,ma, mi))
    return outputs

if __name__ == "__main__":
    from parameters import *
    x = np.linspace(-5, 5, 100)  # Sample input values
    steepness = UniformRange(1, 3, 3)
    pivot = UniformRange(-2, 2, 2)
    min_response = UniformRange(0.1, 0.3, 2)
    max_response = UniformRange(0.7, 0.9, 2)

    outputs = generate_random_sigmoid_curves(x, steepness, pivot, min_response, max_response)
    
    
    mu = UniformRange(-2, 2, 2)
    sigma = UniformRange(0.5, 3, 3)
    outputs = generate_random_gaussian_curves(x, mu, sigma, max_response, min_response)
    
    import matplotlib.pyplot as plt
    fig: plt.Figure = plt.figure(figsize=(10,5))
    for curve in outputs:
        plt.plot(x, curve, '-', color="gray")
    plt.xlabel(r"Orientation")
    plt.ylabel(r"Response")
    plt.title(r"Plot of multiple tuning curves")
    plt.legend()
    plt.show()