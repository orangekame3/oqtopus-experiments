#!/usr/bin/env python3
"""
Custom exception classes for OQTOPUS Experiments

Provides specific exception types for better error handling and user feedback.
"""

from typing import Any


class OQTOPUSExperimentError(Exception):
    """Base exception for all OQTOPUS experiment errors"""

    def __init__(self, message: str, suggestions: list[str] | None = None):
        """
        Initialize experiment error

        Args:
            message: Error description
            suggestions: List of suggested solutions
        """
        super().__init__(message)
        self.suggestions = suggestions or []


class AnalysisError(OQTOPUSExperimentError):
    """Base class for analysis-related errors"""

    pass


class InsufficientDataError(AnalysisError):
    """Raised when there's not enough data for analysis"""

    def __init__(self, data_points: int, required: int, experiment_type: str = ""):
        message = (
            f"Insufficient data for analysis: {data_points} points (need ≥{required})"
        )
        if experiment_type:
            message = f"{experiment_type}: {message}"

        suggestions = [
            "Increase the number of measurement points",
            "Check if data collection completed successfully",
            "Verify backend execution didn't fail silently",
        ]
        super().__init__(message, suggestions)
        self.data_points = data_points
        self.required = required


class FittingError(AnalysisError):
    """Raised when curve fitting fails"""

    def __init__(self, fit_type: str, reason: str):
        message = f"{fit_type} fitting failed: {reason}"
        suggestions = [
            "Check if data quality is sufficient for fitting",
            "Try different initial parameter guesses",
            "Verify data doesn't contain NaN or infinite values",
            "Consider using more robust fitting algorithms",
        ]
        super().__init__(message, suggestions)
        self.fit_type = fit_type
        self.reason = reason


class InvalidParameterError(AnalysisError):
    """Raised when parameters are invalid"""

    def __init__(self, parameter: str, value: Any, expected: str):
        message = f"Invalid parameter '{parameter}': {value} (expected: {expected})"
        suggestions = [
            f"Check {parameter} value is within expected range",
            "Verify parameter types match requirements",
            "Review experiment setup and configuration",
        ]
        super().__init__(message, suggestions)
        self.parameter = parameter
        self.value = value
        self.expected = expected


class DataQualityError(AnalysisError):
    """Raised when data quality issues prevent analysis"""

    def __init__(self, issue: str, data_info: str = ""):
        message = f"Data quality issue: {issue}"
        if data_info:
            message += f" ({data_info})"

        suggestions = [
            "Check measurement data for anomalies",
            "Verify backend execution completed successfully",
            "Consider re-running the experiment",
            "Check for hardware or software issues",
        ]
        super().__init__(message, suggestions)
        self.issue = issue
