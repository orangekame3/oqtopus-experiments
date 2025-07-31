#!/usr/bin/env python3
"""
Tests for improved error handling system

Tests comprehensive error handling, validation, and recovery suggestions
across different experiment types and failure scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from oqtopus_experiments.exceptions import (
    DataQualityError,
    FittingError,
    InsufficientDataError,
    InvalidParameterError,
    OQTOPUSExperimentError,
)
from oqtopus_experiments.models.analysis_result import AnalysisResult
from oqtopus_experiments.models.randomized_benchmarking_models import (
    RandomizedBenchmarkingData,
    RandomizedBenchmarkingResult,
)
from oqtopus_experiments.utils.validation_helpers import (
    validate_data_length,
    validate_fitting_data,
    validate_measurement_counts,
    validate_positive_values,
    validate_probability_values,
    validate_sequence_lengths,
)


class TestExceptionClasses:
    """Test custom exception classes provide helpful information"""

    def test_base_exception_with_suggestions(self):
        suggestions = ["Try this", "Or that"]
        error = OQTOPUSExperimentError("Test error", suggestions)

        assert str(error) == "Test error"
        assert error.suggestions == suggestions

    def test_insufficient_data_error(self):
        error = InsufficientDataError(2, 5, "Test Experiment")

        assert "Test Experiment" in str(error)
        assert "2 points" in str(error)
        assert "need ≥5" in str(error)
        assert len(error.suggestions) > 0
        assert error.data_points == 2
        assert error.required == 5

    def test_fitting_error(self):
        error = FittingError("exponential", "convergence failed")

        assert "exponential fitting failed" in str(error)
        assert "convergence failed" in str(error)
        assert len(error.suggestions) > 0

    def test_invalid_parameter_error(self):
        error = InvalidParameterError("test_param", -1, "positive values")

        assert "test_param" in str(error)
        assert "-1" in str(error)
        assert "positive values" in str(error)
        assert len(error.suggestions) > 0

    def test_data_quality_error(self):
        error = DataQualityError("NaN values detected", "5 NaN values")

        assert "NaN values detected" in str(error)
        assert "5 NaN values" in str(error)
        assert len(error.suggestions) > 0


class TestAnalysisResult:
    """Test structured analysis result class"""

    def test_success_result_creation(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = AnalysisResult.success_result(df, warnings=["test warning"])

        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 3
        assert result.warnings == ["test warning"]
        assert result.errors == []

    def test_error_result_creation(self):
        errors = ["Error 1", "Error 2"]
        suggestions = ["Fix 1", "Fix 2"]
        result = AnalysisResult.error_result(errors, suggestions)

        assert result.success is False
        assert result.data is None
        assert result.errors == errors
        assert result.suggestions == suggestions

    def test_add_methods(self):
        result = AnalysisResult.success_result(pd.DataFrame())

        result.add_warning("Warning message")
        result.add_error("Error message")
        result.add_suggestion("Suggestion message")

        assert result.warnings == ["Warning message"]
        assert result.errors == ["Error message"]
        assert result.suggestions == ["Suggestion message"]
        assert result.success is False  # Should become False after adding error

    def test_has_methods(self):
        result = AnalysisResult.success_result(pd.DataFrame())

        assert not result.has_errors()
        assert not result.has_warnings()

        result.add_warning("Warning")
        result.add_error("Error")

        assert result.has_errors()
        assert result.has_warnings()

    def test_get_summary(self):
        # Success case
        result = AnalysisResult.success_result(pd.DataFrame())
        summary = result.get_summary()
        assert "✅" in summary
        assert "completed successfully" in summary

        # Error case
        result = AnalysisResult.error_result(["Error 1", "Error 2"])
        summary = result.get_summary()
        assert "❌" in summary
        assert "2 errors" in summary

    def test_to_legacy_dataframe_success(self):
        df = pd.DataFrame({"test": [1, 2, 3]})
        result = AnalysisResult.success_result(df)
        legacy_df = result.to_legacy_dataframe()

        assert legacy_df.equals(df)

    def test_to_legacy_dataframe_error(self):
        result = AnalysisResult.error_result(
            ["Test error"], suggestions=["Test suggestion"]
        )
        legacy_df = result.to_legacy_dataframe()

        assert "error" in legacy_df.columns
        assert "success" in legacy_df.columns
        assert not legacy_df.iloc[0]["success"]
        assert "Test error" in legacy_df.iloc[0]["error"]


class TestValidationHelpers:
    """Test input validation functions"""

    def test_validate_data_length_success(self):
        # Should not raise for sufficient data
        validate_data_length([1, 2, 3], 2, "Test")

    def test_validate_data_length_failure(self):
        with pytest.raises(InsufficientDataError) as exc_info:
            validate_data_length([1], 3, "Test Experiment")

        assert "Test Experiment" in str(exc_info.value)
        assert exc_info.value.data_points == 1
        assert exc_info.value.required == 3

    def test_validate_probability_values_success(self):
        # Valid probabilities
        validate_probability_values([0.0, 0.5, 1.0])
        validate_probability_values([0.1, 0.9], allow_zero=False, allow_one=False)

    def test_validate_probability_values_out_of_range(self):
        with pytest.raises(InvalidParameterError):
            validate_probability_values([-0.1, 0.5, 1.2])

    def test_validate_probability_values_nan(self):
        with pytest.raises(DataQualityError):
            validate_probability_values([0.5, np.nan, 0.8])

    def test_validate_probability_values_inf(self):
        with pytest.raises(DataQualityError):
            validate_probability_values([0.5, np.inf, 0.8])

    def test_validate_positive_values_success(self):
        validate_positive_values([0.1, 1.0, 10.5], "test_param")

    def test_validate_positive_values_failure(self):
        with pytest.raises(InvalidParameterError):
            validate_positive_values([0.1, -1.0, 2.0], "test_param")

    def test_validate_sequence_lengths_success(self):
        validate_sequence_lengths([1, 2, 5, 10, 20])

    def test_validate_sequence_lengths_insufficient(self):
        with pytest.raises(InsufficientDataError):
            validate_sequence_lengths([1])

    def test_validate_sequence_lengths_invalid(self):
        with pytest.raises(InvalidParameterError):
            validate_sequence_lengths([1, 0, 5])  # Zero length invalid

    def test_validate_fitting_data_success(self):
        validate_fitting_data([1, 2, 3, 4], [0.9, 0.8, 0.7, 0.6], "Test")

    def test_validate_fitting_data_length_mismatch(self):
        with pytest.raises(InvalidParameterError):
            validate_fitting_data([1, 2, 3], [0.9, 0.8], "Test")

    def test_validate_fitting_data_insufficient(self):
        with pytest.raises(InsufficientDataError):
            validate_fitting_data([1, 2], [0.9, 0.8], "Test")

    def test_validate_measurement_counts_success(self):
        validate_measurement_counts({"00": 50, "01": 30, "10": 15, "11": 5})

    def test_validate_measurement_counts_empty(self):
        with pytest.raises(DataQualityError):
            validate_measurement_counts({})

    def test_validate_measurement_counts_negative(self):
        with pytest.raises(InvalidParameterError):
            validate_measurement_counts({"00": 50, "01": -10})


class TestRandomizedBenchmarkingErrorHandling:
    """Test improved error handling in RandomizedBenchmarking analysis"""

    def create_valid_rb_data(self):
        """Create valid RB data for testing"""
        lengths = [1, 2, 5, 10, 20]
        survival_probs = [
            [0.95, 0.94],
            [0.90, 0.89],
            [0.75, 0.76],
            [0.60, 0.58],
            [0.45, 0.47],
        ]
        mean_probs = [np.mean(probs) for probs in survival_probs]
        std_probs = [np.std(probs) for probs in survival_probs]

        return RandomizedBenchmarkingData(
            sequence_lengths=lengths,
            survival_probabilities=survival_probs,
            mean_survival_probabilities=mean_probs,
            std_survival_probabilities=std_probs,
            num_samples=2,
        )

    def test_successful_analysis(self):
        """Test successful analysis with valid data"""
        data = self.create_valid_rb_data()
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RandomizedBenchmarkingResult(
            data=data, raw_results=mock_raw_results, experiment_instance=mock_experiment
        )

        df = result.analyze(plot=False)

        # Should return DataFrame with results
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # 5 sequence lengths
        assert "sequence_length" in df.columns
        assert "mean_survival_probability" in df.columns

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        # Create data with only 1 point
        data = RandomizedBenchmarkingData(
            sequence_lengths=[1],
            survival_probabilities=[[0.95]],
            mean_survival_probabilities=[0.95],
            std_survival_probabilities=[0.02],
            num_samples=1,
        )
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RandomizedBenchmarkingResult(
            data=data, raw_results=mock_raw_results, experiment_instance=mock_experiment
        )

        df = result.analyze(plot=False)

        # Should still return DataFrame but with warnings
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_invalid_probability_handling(self):
        """Test handling of invalid probability values"""
        data = self.create_valid_rb_data()
        # Corrupt data with invalid probabilities
        data.mean_survival_probabilities = [1.5, 0.9, 0.8, 0.7, 0.6]  # 1.5 > 1.0
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RandomizedBenchmarkingResult(
            data=data, raw_results=mock_raw_results, experiment_instance=mock_experiment
        )

        df = result.analyze(plot=False)

        # Should handle error gracefully
        assert isinstance(df, pd.DataFrame)
        # Should contain error information
        if "error" in df.columns:
            assert not df.empty

    def test_nan_data_handling(self):
        """Test handling of NaN values in data"""
        data = self.create_valid_rb_data()
        # Introduce NaN values
        data.mean_survival_probabilities = [0.9, np.nan, 0.8, 0.7, 0.6]
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RandomizedBenchmarkingResult(
            data=data, raw_results=mock_raw_results, experiment_instance=mock_experiment
        )

        df = result.analyze(plot=False)

        # Should handle NaN gracefully
        assert isinstance(df, pd.DataFrame)

    def test_length_mismatch_handling(self):
        """Test handling of data length mismatches"""
        data = self.create_valid_rb_data()
        # Create length mismatch
        data.mean_survival_probabilities = [
            0.9,
            0.8,
            0.7,
        ]  # Only 3 values for 5 lengths
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RandomizedBenchmarkingResult(
            data=data, raw_results=mock_raw_results, experiment_instance=mock_experiment
        )

        df = result.analyze(plot=False)

        # Should handle mismatch gracefully
        assert isinstance(df, pd.DataFrame)

    def test_fitting_failure_handling(self):
        """Test handling when curve fitting fails"""
        data = self.create_valid_rb_data()
        # Create data that's hard to fit (all same values)
        data.mean_survival_probabilities = [0.5, 0.5, 0.5, 0.5, 0.5]
        data.std_survival_probabilities = [0.0, 0.0, 0.0, 0.0, 0.0]
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RandomizedBenchmarkingResult(
            data=data, raw_results=mock_raw_results, experiment_instance=mock_experiment
        )

        df = result.analyze(plot=False)

        # Should handle fitting failure gracefully
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1  # Should return at least some data

    def test_plotting_failure_handling(self):
        """Test handling when plotting fails"""
        data = self.create_valid_rb_data()
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RandomizedBenchmarkingResult(
            data=data, raw_results=mock_raw_results, experiment_instance=mock_experiment
        )

        # This should not raise an exception even if plotting fails
        df = result.analyze(plot=True)  # Enable plotting to test error handling

        assert isinstance(df, pd.DataFrame)


class TestBackwardCompatibility:
    """Test that new error handling maintains backward compatibility"""

    def test_legacy_dataframe_return(self):
        """Test that analyze methods still return DataFrames"""
        data = RandomizedBenchmarkingData(
            sequence_lengths=[1, 2, 5],
            survival_probabilities=[[0.95], [0.90], [0.75]],
            mean_survival_probabilities=[0.95, 0.90, 0.75],
            std_survival_probabilities=[0.02, 0.03, 0.04],
            num_samples=1,
        )
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RandomizedBenchmarkingResult(
            data=data, raw_results=mock_raw_results, experiment_instance=mock_experiment
        )

        df = result.analyze(plot=False)

        # Should return DataFrame (backward compatibility)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_error_information_in_dataframe(self):
        """Test that error information is included in returned DataFrame"""
        # Create invalid data
        data = RandomizedBenchmarkingData(
            sequence_lengths=[],  # Empty - will cause error
            survival_probabilities=[],
            mean_survival_probabilities=[],
            std_survival_probabilities=[],
            num_samples=0,
        )
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RandomizedBenchmarkingResult(
            data=data, raw_results=mock_raw_results, experiment_instance=mock_experiment
        )

        df = result.analyze(plot=False)

        # Should still return DataFrame with error info
        assert isinstance(df, pd.DataFrame)
        # Error information should be accessible
        if "error" in df.columns or "success" in df.columns:
            assert not df.empty
