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
from typing import List, Union, Tuple, Callable



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
