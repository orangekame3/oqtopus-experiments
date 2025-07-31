#!/usr/bin/env python3
"""
Tests for error handling improvements across multiple experiment types

Tests comprehensive error handling for Rabi, T1, and other experiment models.
"""

import numpy as np
import pandas as pd

from oqtopus_experiments.models.rabi_models import RabiAnalysisResult, RabiData
from oqtopus_experiments.models.t1_models import T1AnalysisResult, T1Data


class TestRabiErrorHandling:
    """Test improved error handling in Rabi analysis"""

    def create_valid_rabi_data(self):
        """Create valid Rabi data for testing"""
        return RabiData(
            amplitudes=[0.1, 0.2, 0.3, 0.4, 0.5],
            probabilities=[0.1, 0.4, 0.7, 0.9, 0.8],
            probability_errors=[0.05, 0.06, 0.05, 0.04, 0.06],
            shots_per_point=1000,
        )

    def test_successful_rabi_analysis(self):
        """Test successful Rabi analysis with valid data"""
        data = self.create_valid_rabi_data()
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RabiAnalysisResult(
            data=data,
            raw_results=mock_raw_results,
            experiment_instance=mock_experiment,
        )

        df = result.analyze(plot=False, save_data=False, save_image=False)

        # Should return DataFrame with results
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # 5 amplitude points
        assert "amplitude" in df.columns
        assert "probability" in df.columns
        assert "fitted_probability" in df.columns

    def test_rabi_insufficient_data_handling(self):
        """Test handling of insufficient Rabi data"""
        data = RabiData(
            amplitudes=[0.1],
            probabilities=[0.3],
            probability_errors=[0.05],
            shots_per_point=1000,
        )
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RabiAnalysisResult(
            data=data,
            raw_results=mock_raw_results,
            experiment_instance=mock_experiment,
        )

        df = result.analyze(plot=False, save_data=False, save_image=False)

        # Should handle insufficient data gracefully
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_rabi_invalid_data_handling(self):
        """Test handling of invalid Rabi data"""
        data = RabiData(
            amplitudes=[0.1, 0.2, 0.3],
            probabilities=[1.5, 0.4, 0.7],  # Invalid probability > 1
            probability_errors=[0.05, 0.06, 0.05],
            shots_per_point=1000,
        )
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = RabiAnalysisResult(
            data=data,
            raw_results=mock_raw_results,
            experiment_instance=mock_experiment,
        )

        df = result.analyze(plot=False, save_data=False, save_image=False)

        # Should handle invalid data gracefully
        assert isinstance(df, pd.DataFrame)
        # Should return error information
        if "error" in df.columns:
            assert not df.empty


class TestT1ErrorHandling:
    """Test improved error handling in T1 analysis"""

    def create_valid_t1_data(self):
        """Create valid T1 data for testing"""
        return T1Data(
            delay_times=[0.0, 1000.0, 2000.0, 3000.0, 4000.0],
            probabilities=[1.0, 0.8, 0.6, 0.4, 0.3],
            probability_errors=[0.02, 0.03, 0.04, 0.05, 0.05],
            shots_per_point=1000,
        )

    def test_successful_t1_analysis(self):
        """Test successful T1 analysis with valid data"""
        data = self.create_valid_t1_data()
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = T1AnalysisResult(
            data=data,
            raw_results=mock_raw_results,
            experiment_instance=mock_experiment,
        )

        df = result.analyze(plot=False, save_data=False, save_image=False)

        # Should return DataFrame with results
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # 5 delay points
        assert "delay_time_ns" in df.columns
        assert "probability" in df.columns
        assert "fitted_probability" in df.columns

    def test_t1_insufficient_data_handling(self):
        """Test handling of insufficient T1 data"""
        data = T1Data(
            delay_times=[0.0],
            probabilities=[1.0],
            probability_errors=[0.02],
            shots_per_point=1000,
        )
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = T1AnalysisResult(
            data=data,
            raw_results=mock_raw_results,
            experiment_instance=mock_experiment,
        )

        df = result.analyze(plot=False, save_data=False, save_image=False)

        # Should handle insufficient data gracefully
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_t1_negative_delay_handling(self):
        """Test handling of negative delay times"""
        data = T1Data(
            delay_times=[-100.0, 1000.0, 2000.0],
            probabilities=[1.0, 0.8, 0.6],
            probability_errors=[0.02, 0.03, 0.04],
            shots_per_point=1000,
        )
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = T1AnalysisResult(
            data=data,
            raw_results=mock_raw_results,
            experiment_instance=mock_experiment,
        )

        df = result.analyze(plot=False, save_data=False, save_image=False)

        # Should handle negative delays gracefully
        assert isinstance(df, pd.DataFrame)
        # Should return error information
        if "error" in df.columns:
            assert "non-negative" in df.iloc[0]["error"]

    def test_t1_nan_data_handling(self):
        """Test handling of NaN values in T1 data"""
        data = T1Data(
            delay_times=[0.0, 1000.0, 2000.0],
            probabilities=[1.0, np.nan, 0.6],
            probability_errors=[0.02, 0.03, 0.04],
            shots_per_point=1000,
        )
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        result = T1AnalysisResult(
            data=data,
            raw_results=mock_raw_results,
            experiment_instance=mock_experiment,
        )

        df = result.analyze(plot=False, save_data=False, save_image=False)

        # Should handle NaN gracefully
        assert isinstance(df, pd.DataFrame)


class TestBackwardCompatibilityMultiExperiment:
    """Test backward compatibility across multiple experiment types"""

    def test_all_experiments_return_dataframes(self):
        """Test that all experiments still return DataFrames"""
        # Rabi test
        rabi_data = RabiData(
            amplitudes=[0.1, 0.2, 0.3],
            probabilities=[0.1, 0.4, 0.7],
            probability_errors=[0.05, 0.06, 0.05],
            shots_per_point=1000,
        )
        mock_raw_results = {"test": "data"}
        mock_experiment = type("MockExperiment", (), {})()
        rabi_result = RabiAnalysisResult(
            data=rabi_data,
            raw_results=mock_raw_results,
            experiment_instance=mock_experiment,
        )
        rabi_df = rabi_result.analyze(plot=False, save_data=False, save_image=False)
        assert isinstance(rabi_df, pd.DataFrame)

        # T1 test
        t1_data = T1Data(
            delay_times=[0.0, 1000.0, 2000.0],
            probabilities=[1.0, 0.8, 0.6],
            probability_errors=[0.02, 0.03, 0.04],
            shots_per_point=1000,
        )
        t1_result = T1AnalysisResult(
            data=t1_data,
            raw_results=mock_raw_results,
            experiment_instance=mock_experiment,
        )
        t1_df = t1_result.analyze(plot=False, save_data=False, save_image=False)
        assert isinstance(t1_df, pd.DataFrame)
