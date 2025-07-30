#!/usr/bin/env python3
"""
Validation helpers for experiment analysis

Provides comprehensive input validation to catch errors early
and provide helpful feedback to users.
"""

from typing import Any

import numpy as np

from ..exceptions import DataQualityError, InsufficientDataError, InvalidParameterError


def validate_data_length(
    data: list[Any], min_length: int, experiment_type: str = ""
) -> None:
    """
    Validate data has sufficient length for analysis

    Args:
        data: Data list to validate
        min_length: Minimum required length
        experiment_type: Type of experiment for error context

    Raises:
        InsufficientDataError: If data length is insufficient
    """
    if len(data) < min_length:
        raise InsufficientDataError(len(data), min_length, experiment_type)


def validate_probability_values(
    probabilities: list[float], allow_zero: bool = True, allow_one: bool = True
) -> None:
    """
    Validate probability values are in valid range

    Args:
        probabilities: List of probability values to validate
        allow_zero: Whether to allow exactly 0.0
        allow_one: Whether to allow exactly 1.0

    Raises:
        InvalidParameterError: If probabilities are out of range
        DataQualityError: If probabilities contain NaN or inf
    """
    # Check for NaN or infinite values
    if any(not np.isfinite(p) for p in probabilities):
        nan_count = sum(1 for p in probabilities if np.isnan(p))
        inf_count = sum(1 for p in probabilities if np.isinf(p))
        raise DataQualityError(
            "Invalid probability values detected",
            f"{nan_count} NaN, {inf_count} infinite values",
        )

    # Check range
    min_val = 0.0 if allow_zero else 1e-10
    max_val = 1.0 if allow_one else 1.0 - 1e-10

    out_of_range = [p for p in probabilities if not (min_val <= p <= max_val)]
    if out_of_range:
        raise InvalidParameterError(
            "probabilities",
            f"values outside [{min_val}, {max_val}]: {out_of_range[:3]}",
            f"values between {min_val} and {max_val}",
        )


def validate_positive_values(values: list[float], parameter_name: str) -> None:
    """
    Validate values are positive

    Args:
        values: Values to validate
        parameter_name: Name of parameter for error messages

    Raises:
        InvalidParameterError: If values are not positive
        DataQualityError: If values contain NaN or inf
    """
    # Check for NaN or infinite values
    if any(not np.isfinite(v) for v in values):
        raise DataQualityError(
            f"Invalid {parameter_name} values",
            "contains NaN or infinite values",
        )

    # Check positivity
    non_positive = [v for v in values if v <= 0]
    if non_positive:
        raise InvalidParameterError(
            parameter_name,
            f"non-positive values: {non_positive[:3]}",
            "positive values > 0",
        )


def validate_sequence_lengths(lengths: list[int]) -> None:
    """
    Validate sequence lengths for experiments like Randomized Benchmarking

    Args:
        lengths: List of sequence lengths

    Raises:
        InvalidParameterError: If lengths are invalid
        InsufficientDataError: If not enough lengths provided
    """
    validate_data_length(lengths, 2, "Sequence length analysis")

    # Check for non-positive lengths
    if any(length <= 0 for length in lengths):
        invalid = [length for length in lengths if length <= 0]
        raise InvalidParameterError(
            "sequence_lengths",
            f"non-positive lengths: {invalid}",
            "positive integers",
        )

    # Check for reasonable range
    if max(lengths) > 10000:
        raise InvalidParameterError(
            "sequence_lengths",
            f"maximum length {max(lengths)} seems unusually large",
            "reasonable sequence lengths (< 10000)",
        )


def validate_fitting_data(
    x_data: list[float], y_data: list[float], experiment_type: str = ""
) -> None:
    """
    Validate data is suitable for curve fitting

    Args:
        x_data: Independent variable data
        y_data: Dependent variable data
        experiment_type: Type of experiment for context

    Raises:
        InsufficientDataError: If not enough data points
        DataQualityError: If data has quality issues
        InvalidParameterError: If data dimensions don't match
    """
    # Check lengths match
    if len(x_data) != len(y_data):
        raise InvalidParameterError(
            "data_dimensions",
            f"x_data length {len(x_data)} != y_data length {len(y_data)}",
            "matching dimensions",
        )

    # Check sufficient data for fitting
    validate_data_length(x_data, 3, f"{experiment_type} fitting")

    # Validate x_data
    validate_positive_values(x_data, "x_data")

    # Validate y_data (probabilities)
    validate_probability_values(y_data)

    # Check for duplicate x values
    unique_x = set(x_data)
    if len(unique_x) < len(x_data):
        raise DataQualityError(
            "Duplicate x values found",
            f"{len(x_data) - len(unique_x)} duplicates",
        )

    # Check for reasonable data spread
    x_range = max(x_data) - min(x_data)
    if x_range == 0:
        raise DataQualityError(
            "No variation in x data",
            "all x values are identical",
        )


def validate_measurement_counts(counts: dict[str, int]) -> None:
    """
    Validate measurement count dictionary

    Args:
        counts: Dictionary of measurement outcomes and counts

    Raises:
        DataQualityError: If counts have issues
        InvalidParameterError: If counts format is invalid
    """
    if not counts:
        raise DataQualityError("Empty measurement counts", "no data to analyze")

    # Check for negative counts
    negative_counts = {k: v for k, v in counts.items() if v < 0}
    if negative_counts:
        raise InvalidParameterError(
            "measurement_counts",
            f"negative counts: {negative_counts}",
            "non-negative integer counts",
        )

    # Check total counts
    total_counts = sum(counts.values())
    if total_counts == 0:
        raise DataQualityError("Zero total counts", "no measurements recorded")

    if total_counts > 1e9:  # Unreasonably large
        raise DataQualityError(
            f"Unusually large total counts: {total_counts}",
            "verify measurement setup",
        )
