"""
Information-theoretic metrics for neural response analysis

This module provides functions for computing information-theoretic measures
such as Fisher Information and Mutual Information for neural responses

Functions:
    fisher_information: Computes Fisher Information for tuning curves
    mutual_information: Computes Mutual Information between response and stimulus
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from BuildYourOwnEmbedding.responses.data import ResponseData, ResponseSet


def fisher_information(curves: npt.NDArray | ResponseSet) -> npt.NDArray:
    """
    Calculates the Fisher Information for a set of tuning curves.

    This function supports both 1-D and arbitrary-dimensional tuning curves.

    Args:
        curves: A numpy array of shape (numCurves, dim1, dim2, ..., dimN) representing
            the tuning curves, where `numCurves` is the number of curves and
            `dim1, dim2, ..., dimN` are the dimensions of each response.
            Alternatively, can be a ResponseSet object.

    Returns:
        A numpy array representing the FI values, with shape (dim1, dim2, ..., dimN)
        for the input curves.

    Raises:
        ValueError: If the dimensionality of `curves` is not at least 2-dimensional.
    """
    if "ResponseSet" in str(curves.__class__):
        curves = np.stack([r.response for r in curves.responses], axis=0)

    if len(curves.shape) < 2:
        raise ValueError("curves must be at least two dimensions")

    if len(curves.shape) == 2:
        fi_values = np.zeros(curves[0].shape)
        for i in range(curves.shape[0]):
            derivative = np.gradient(curves[i])
            fi = derivative**2
            fi_values += fi
        return fi_values
    else:
        fi_values = np.zeros(curves.shape[1:])
        for i in range(curves.shape[0]):
            derivatives = np.gradient(curves[i])
            fi = sum(derivative**2 for derivative in derivatives)
            fi_values += fi
        return fi_values


def mutual_information(
    response: npt.NDArray | ResponseData,
    stimulus: npt.NDArray | None = None,
) -> np.number:
    """
    Computes the mutual information between a single neural response and its stimulus.

    Args:
        response: Neural response with shape (dim1, dim2, ..., dimN).
            Can also be a ResponseData object.
        stimulus: Corresponding stimulus with shape (dim1, dim2, ..., dimN).
            Can be None if response is a ResponseData object.

    Returns:
        The mutual information between the response and the stimulus.

    Raises:
        ValueError: If the stimulus and response are different dimensions, or if
            stimulus is None when response is not a ResponseData object.
    """
    if "ResponseData" in str(response.__class__):
        response, stimulus = response.response, response.x
    else:
        if stimulus is None:
            raise ValueError("Stimulus cannot be None")

    if response.shape != stimulus.shape:
        raise ValueError("Response and Stimulus must have the same dimensions")

    response_flat = response.flatten()
    stimulus_flat = stimulus.flatten()

    # Compute joint distribution
    joint_values = np.stack((response_flat, stimulus_flat), axis=-1)
    unique_joint, counts_joint = np.unique(joint_values, axis=0, return_counts=True)
    joint_probabilities = counts_joint / len(response_flat)

    # Compute marginal distributions
    unique_response, counts_response = np.unique(response_flat, return_counts=True)
    prob_response = counts_response / len(response_flat)

    unique_stimulus, counts_stimulus = np.unique(stimulus_flat, return_counts=True)
    prob_stimulus = counts_stimulus / len(stimulus_flat)

    response_p_dict = {val: prob_response[i] for i, val in enumerate(unique_response)}
    stimulus_p_dict = {val: prob_stimulus[i] for i, val in enumerate(unique_stimulus)}

    mi = 0.0
    for i, joint_val in enumerate(unique_joint):
        resp_val, stim_val = joint_val
        joint_prob = joint_probabilities[i]
        if joint_prob > 0 and response_p_dict[resp_val] > 0 and stimulus_p_dict[stim_val] > 0:
            mi += joint_prob * np.log(
                joint_prob / (response_p_dict[resp_val] * stimulus_p_dict[stim_val])
            )

    return mi
