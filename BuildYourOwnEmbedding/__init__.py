"""
BuildYourOwnEmbedding: A library for generating and analysing synthetic neural responses

This library provides tools for:
    * Modelling various neural response functions
    * Generating sets of synthetic neural responses with configurable parameters
    * Adding realistic noise to responses
    * Computing and visualising RDMs
    * Dimensionality reduction techniques
    * Information-theoretic measures
"""

__version__ = "2.0.0"

from BuildYourOwnEmbedding import analysis, metrics, parameters, responses

__all__ = [
    "parameters",
    "responses",
    "metrics",
    "analysis",
]
