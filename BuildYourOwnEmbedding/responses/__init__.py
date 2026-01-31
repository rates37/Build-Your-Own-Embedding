"""
Neural response functions and data structures

This module provides classes for defining, generating, and analysing neural
response functions.

Classes:
    ResponseManager: Generates ResponseSets from parameter combinations
    GaussianResponse: 1D Gaussian tuning curve
    SigmoidResponse: 1D Sigmoidal response curve
    VonMisesResponse: 1D Von Mises (circular Gaussian) response
    GaussianResponse2D: 2D Gaussian response
    ResponseData: Stores a single response with its parameters
    ResponseSet: Stores and analyses a collections of responses
    CompositeResponse: Combines two response functions with an operation
    ResponseFunction: Abstract base class for all response functions
"""

from BuildYourOwnEmbedding.responses.base import (
    CompositeResponse,
    ResponseFunction,
)
from BuildYourOwnEmbedding.responses.data import (
    ResponseData,
    ResponseSet,
)
from BuildYourOwnEmbedding.responses.functions import (
    GaussianResponse,
    GaussianResponse2D,
    SigmoidResponse,
    VonMisesResponse,
)
from BuildYourOwnEmbedding.responses.manager import ResponseManager

__all__ = [
    # Base classes
    "ResponseFunction",
    "CompositeResponse",
    # Response functions
    "GaussianResponse",
    "SigmoidResponse",
    "VonMisesResponse",
    "GaussianResponse2D",
    # Manager and data
    "ResponseManager",
    "ResponseData",
    "ResponseSet",
]
