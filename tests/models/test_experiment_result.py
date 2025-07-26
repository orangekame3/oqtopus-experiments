#!/usr/bin/env python3
"""
Tests for ExperimentResult
"""

from unittest.mock import patch

import pandas as pd
import pytest

from oqtopus_experiments.models.experiment_result import ExperimentResult


class MockExperiment:
    """Mock experiment for testing"""

    def analyze(self, results, plot=False, save_data=False, save_image=False):
        return {"mock_analysis": "result", "fitted_params": {"amplitude": 1.0}}


class TestExperimentResult:
    """Test ExperimentResult functionality"""

    def test_init_basic(self):
        """Test basic initialization"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        assert result.raw_results == raw_results
        assert result.experiment == experiment
        assert result.experiment_type == "test"
        assert result.analysis_params == {}
        assert result._analyzed_results is None

    def test_init_with_kwargs(self):
        """Test initialization with additional parameters"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="rabi",
            shots=1000,
            amplitude_range=(0, 2.0)
        )

        assert result.experiment_type == "rabi"
        assert result.analysis_params["shots"] == 1000
        assert result.analysis_params["amplitude_range"] == (0, 2.0)

    def test_init_default_experiment_type(self):
        """Test initialization with default experiment type"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment
        )

        assert result.experiment_type == "generic"

    def test_analyze_calls_experiment_analyze(self):
        """Test that analyze method calls experiment's analyze method"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        with patch.object(experiment, 'analyze') as mock_analyze:
            mock_analyze.return_value = {"test": "analysis"}

            _ = result.analyze(plot=False, save_data=False, save_image=False)

            mock_analyze.assert_called_once_with(
                raw_results,
                plot=False,
                save_data=False,
                save_image=False
            )

    def test_analyze_returns_dataframe_when_dict(self):
        """Test that analyze returns DataFrame when experiment returns dict with DataFrames"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        # Mock experiment returns dict with DataFrame
        mock_df = pd.DataFrame({"amplitude": [0, 1, 2], "probability": [0.9, 0.5, 0.1]})

        with patch.object(experiment, 'analyze') as mock_analyze:
            mock_analyze.return_value = {
                "analysis_result": mock_df,
                "fitted_params": {"amplitude": 1.0}
            }

            analysis_result = result.analyze(plot=False, save_data=False, save_image=False)

            assert isinstance(analysis_result, pd.DataFrame)
            pd.testing.assert_frame_equal(analysis_result, mock_df)

    def test_analyze_returns_dataframe_when_direct_dataframe(self):
        """Test that analyze returns DataFrame when experiment returns DataFrame directly"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        # Mock experiment returns DataFrame directly
        mock_df = pd.DataFrame({"amplitude": [0, 1, 2], "probability": [0.9, 0.5, 0.1]})

        with patch.object(experiment, 'analyze') as mock_analyze:
            mock_analyze.return_value = mock_df

            analysis_result = result.analyze(plot=False, save_data=False, save_image=False)

            assert isinstance(analysis_result, pd.DataFrame)
            pd.testing.assert_frame_equal(analysis_result, mock_df)

    def test_analyze_returns_empty_dataframe_when_no_dataframe(self):
        """Test that analyze returns empty DataFrame when no DataFrame found"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        with patch.object(experiment, 'analyze') as mock_analyze:
            mock_analyze.return_value = {"fitted_params": {"amplitude": 1.0}}

            analysis_result = result.analyze(plot=False, save_data=False, save_image=False)

            assert isinstance(analysis_result, pd.DataFrame)
            assert len(analysis_result) == 0

    def test_analyze_caches_results(self):
        """Test that analyze results are cached"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        mock_df = pd.DataFrame({"test": [1, 2, 3]})

        with patch.object(experiment, 'analyze') as mock_analyze:
            mock_analyze.return_value = {"analysis_result": mock_df}

            # First call
            result1 = result.analyze(plot=False, save_data=False, save_image=False)
            # Second call
            result2 = result.analyze(plot=False, save_data=False, save_image=False)

            # analyze should only be called once due to caching
            assert mock_analyze.call_count == 1
            assert result._analyzed_results is not None
            pd.testing.assert_frame_equal(result1, result2)

    def test_analyze_with_default_parameters(self):
        """Test analyze with default parameters"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        with patch.object(experiment, 'analyze') as mock_analyze:
            mock_analyze.return_value = {"test": "analysis"}

            result.analyze(plot=False, save_data=False, save_image=False)

            mock_analyze.assert_called_once_with(
                raw_results,
                plot=False,
                save_data=False,
                save_image=False
            )

    def test_analyze_with_experiment_error(self):
        """Test analyze when experiment's analyze method raises error"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        with patch.object(experiment, 'analyze') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")

            # Should still return empty DataFrame rather than raise
            analysis_result = result.analyze(plot=False, save_data=False, save_image=False)

            assert isinstance(analysis_result, pd.DataFrame)
            assert len(analysis_result) == 0

    def test_analyze_multiple_dataframes_in_dict(self):
        """Test analyze when experiment returns dict with multiple DataFrames"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        # Mock experiment returns dict with multiple DataFrames
        df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        with patch.object(experiment, 'analyze') as mock_analyze:
            mock_analyze.return_value = {
                "main_result": df1,
                "secondary_result": df2,
                "fitted_params": {"amplitude": 1.0}
            }

            analysis_result = result.analyze(plot=False, save_data=False, save_image=False)

            # Should return the first DataFrame found
            assert isinstance(analysis_result, pd.DataFrame)
            pd.testing.assert_frame_equal(analysis_result, df1)

    def test_raw_results_access(self):
        """Test access to raw results"""
        raw_results = {
            "circuit_0": [{"counts": {"0": 800, "1": 200}, "success": True}],
            "circuit_1": [{"counts": {"0": 300, "1": 700}, "success": True}]
        }
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        assert result.raw_results == raw_results
        assert len(result.raw_results) == 2
        assert "circuit_0" in result.raw_results
        assert "circuit_1" in result.raw_results

    def test_experiment_access(self):
        """Test access to experiment instance"""
        raw_results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
        experiment = MockExperiment()

        result = ExperimentResult(
            raw_results=raw_results,
            experiment_instance=experiment,
            experiment_type="test"
        )

        assert result.experiment == experiment
        assert hasattr(result.experiment, 'analyze')


if __name__ == "__main__":
    pytest.main([__file__])

