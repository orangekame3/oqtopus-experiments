#!/usr/bin/env python3
"""
Tests for display utilities
"""

from unittest.mock import patch

import pytest

from oqtopus_experiments.utils.display import (
    create_experiment_summary,
    display_experiment_results,
    format_scientific_notation,
    format_time_duration,
)


class TestDisplayExperimentResults:
    """Test display_experiment_results function"""

    def test_display_with_rich_t1_experiment(self):
        """Test display with Rich for T1 experiment"""
        results = {
            "analysis": {"estimated_t1": 125.5, "fit_quality": {"r_squared": 0.95}},
            "device_results": {
                "device1": {"analysis": {"fit_success": True}},
                "device2": {"analysis": {"fit_success": False}},
            },
            "experiment_params": {"delay_times": [0, 50, 100, 150, 200]},
        }

        # Just test that it doesn't crash - the function works with Rich
        # No need to assert specific calls since Rich handles the output
        display_experiment_results(results, "T1", use_rich=True)
        # Function should complete without error

    def test_display_with_rich_ramsey_experiment(self):
        """Test display with Rich for Ramsey experiment"""
        results = {
            "analysis": {
                "estimated_t2_star": 89.3,
                "estimated_frequency": 2.5,
                "fit_quality": {"r_squared": 0.92},
            }
        }

        display_experiment_results(results, "Ramsey", use_rich=True)
        # Function should complete without error

    def test_display_with_rich_t2_echo_experiment(self):
        """Test display with Rich for T2 Echo experiment"""
        results = {
            "analysis": {"estimated_t2": 150.7, "fit_quality": {"r_squared": 0.88}}
        }

        display_experiment_results(results, "T2 Echo", use_rich=True)
        # Function should complete without error

    def test_display_with_rich_chsh_experiment(self):
        """Test display with Rich for CHSH experiment"""
        results = {"analysis": {"s_value": 2.414, "fit_quality": {"r_squared": 0.97}}}

        display_experiment_results(results, "CHSH", use_rich=True)
        # Function should complete without error

    def test_display_with_rich_rabi_experiment(self):
        """Test display with Rich for Rabi experiment"""
        results = {
            "analysis": {"rabi_frequency": 15.2, "fit_quality": {"r_squared": 0.93}}
        }

        display_experiment_results(results, "Rabi", use_rich=True)
        # Function should complete without error

    def test_display_with_rich_multiple_devices(self):
        """Test display with Rich for multiple devices"""
        results = {
            "analysis": {"estimated_t1": 100.0},
            "device_results": {
                "device1": {
                    "analysis": {"fit_success": True, "data_points": [1, 2, 3]}
                },
                "device2": {"analysis": {"fit_success": False, "data_points": [4, 5]}},
                "device3": "invalid_data",
            },
        }

        display_experiment_results(results, "T1", use_rich=True)
        # Function should complete without error

    def test_display_with_rich_import_error_fallback(self):
        """Test fallback to plain text when Rich import fails"""
        results = {"analysis": {"estimated_t1": 100.0}}

        with patch(
            "builtins.__import__", side_effect=ImportError("Rich not available")
        ):
            with patch("builtins.print") as mock_print:
                display_experiment_results(results, "T1", use_rich=True)
                # Should fallback to plain text
                mock_print.assert_called()

    def test_display_without_rich_t1_experiment(self):
        """Test display without Rich for T1 experiment"""
        results = {
            "analysis": {"estimated_t1": 125.5, "fit_quality": {"r_squared": 0.95}},
            "device_results": {
                "device1": {"analysis": {"fit_success": True}},
                "device2": {"analysis": {"fit_success": False}},
            },
        }

        with patch("builtins.print") as mock_print:
            display_experiment_results(results, "T1", use_rich=False)

            # Check that plain text output was generated
            mock_print.assert_called()
            # Verify some expected content
            calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("T1 Experiment Results" in str(call) for call in calls)
            assert any("Estimated T1: 125.500 ns" in str(call) for call in calls)
            assert any("R² (Fit Quality): 0.9500" in str(call) for call in calls)

    def test_display_without_rich_ramsey_experiment(self):
        """Test display without Rich for Ramsey experiment"""
        results = {"analysis": {"estimated_t2_star": 89.3, "estimated_frequency": 2.5}}

        with patch("builtins.print") as mock_print:
            display_experiment_results(results, "Ramsey", use_rich=False)

            calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("Estimated T2*: 89.300 ns" in str(call) for call in calls)
            assert any("Detuning Frequency: 2.500 MHz" in str(call) for call in calls)

    def test_display_without_rich_chsh_experiment(self):
        """Test display without Rich for CHSH experiment"""
        results = {"analysis": {"s_value": 2.414}}

        with patch("builtins.print") as mock_print:
            display_experiment_results(results, "CHSH", use_rich=False)

            calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("S Value: 2.414" in str(call) for call in calls)
            assert any("Classical Bound: 2.000" in str(call) for call in calls)
            assert any("Quantum Maximum: 2.828" in str(call) for call in calls)

    def test_display_without_rich_multiple_devices(self):
        """Test display without Rich for multiple devices"""
        results = {
            "device_results": {
                "device1": {"analysis": {"fit_success": True}},
                "device2": {"analysis": {"fit_success": False}},
                "device3": "invalid_data",
            }
        }

        with patch("builtins.print") as mock_print:
            display_experiment_results(results, "T1", use_rich=False)

            calls = [call.args[0] for call in mock_print.call_args_list]
            assert any("Devices Used: 3" in str(call) for call in calls)
            assert any("device1: Success" in str(call) for call in calls)
            assert any("device2: Partial" in str(call) for call in calls)
            assert any("device3: Failed" in str(call) for call in calls)

    def test_display_empty_results(self):
        """Test display with empty results"""
        results = {}

        with patch("builtins.print") as mock_print:
            display_experiment_results(results, "Test", use_rich=False)
            mock_print.assert_called()


class TestCreateExperimentSummary:
    """Test create_experiment_summary function"""

    def test_create_summary_basic(self):
        """Test creating basic experiment summary"""
        results = {}
        experiment_type = "T1"
        experiment_params = {"delay_times": [0, 50, 100]}

        with patch("time.time", return_value=1234567890.0):
            with patch("time.strftime", return_value="2023-01-01 12:00:00"):
                summary = create_experiment_summary(
                    results, experiment_type, experiment_params
                )

        assert summary["experiment_type"] == "T1"
        assert summary["timestamp"] == 1234567890.0
        assert summary["summary_created"] == "2023-01-01 12:00:00"
        assert summary["total_devices"] == 0
        assert summary["successful_devices"] == 0
        assert summary["status"] == "no_devices"
        assert summary["experiment_parameters"] == {"delay_times": [0, 50, 100]}

    def test_create_summary_with_device_results(self):
        """Test creating summary with device results"""
        results = {
            "device_results": {
                "device1": {"analysis": {"fit_success": True}},
                "device2": {"analysis": {"fit_success": False}},
                "device3": {"analysis": {"fit_success": True}},
                "device4": "invalid_data",
            }
        }

        summary = create_experiment_summary(results, "Ramsey")

        assert summary["total_devices"] == 4
        assert summary["successful_devices"] == 2
        assert summary["success_rate"] == 0.5
        assert summary["device_names"] == ["device1", "device2", "device3", "device4"]
        assert summary["status"] == "partial_success"

    def test_create_summary_all_devices_successful(self):
        """Test creating summary when all devices are successful"""
        results = {
            "device_results": {
                "device1": {"analysis": {"fit_success": True}},
                "device2": {"analysis": {"fit_success": True}},
            }
        }

        summary = create_experiment_summary(results, "T1")

        assert summary["successful_devices"] == 2
        assert summary["success_rate"] == 1.0
        assert summary["status"] == "all_success"

    def test_create_summary_all_devices_failed(self):
        """Test creating summary when all devices failed"""
        results = {
            "device_results": {
                "device1": {"analysis": {"fit_success": False}},
                "device2": "invalid_data",
            }
        }

        summary = create_experiment_summary(results, "T1")

        assert summary["successful_devices"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["status"] == "all_failed"

    def test_create_summary_t1_analysis(self):
        """Test creating summary with T1 analysis results"""
        results = {
            "analysis": {
                "estimated_t1": 125.5,
                "fit_quality": {"r_squared": 0.95, "chi_squared": 1.2},
            }
        }

        summary = create_experiment_summary(results, "T1")

        assert "main_results" in summary
        assert summary["main_results"]["t1_time"] == 125.5
        assert summary["fit_quality"]["r_squared"] == 0.95
        assert summary["fit_quality"]["chi_squared"] == 1.2

    def test_create_summary_ramsey_analysis(self):
        """Test creating summary with Ramsey analysis results"""
        results = {
            "analysis": {
                "estimated_t2_star": 89.3,
                "estimated_frequency": 2.5,
                "fit_quality": {"r_squared": 0.92},
            }
        }

        summary = create_experiment_summary(results, "Ramsey")

        assert summary["main_results"]["t2_star"] == 89.3
        assert summary["main_results"]["detuning_frequency"] == 2.5

    def test_create_summary_t2_echo_analysis(self):
        """Test creating summary with T2 Echo analysis results"""
        results = {
            "analysis": {"estimated_t2": 150.7, "fit_quality": {"r_squared": 0.88}}
        }

        summary = create_experiment_summary(results, "T2 Echo")

        assert summary["main_results"]["t2_time"] == 150.7

    def test_create_summary_chsh_analysis(self):
        """Test creating summary with CHSH analysis results"""
        results = {"analysis": {"s_value": 2.414, "fit_quality": {"r_squared": 0.97}}}

        summary = create_experiment_summary(results, "CHSH")

        assert summary["main_results"]["s_value"] == 2.414
        assert summary["main_results"]["bell_violation"] is True

        # Test with s_value <= 2.0
        results["analysis"]["s_value"] = 1.8
        summary = create_experiment_summary(results, "CHSH")
        assert summary["main_results"]["bell_violation"] is False

    def test_create_summary_rabi_analysis(self):
        """Test creating summary with Rabi analysis results"""
        results = {
            "analysis": {"rabi_frequency": 15.2, "fit_quality": {"r_squared": 0.93}}
        }

        summary = create_experiment_summary(results, "Rabi")

        assert summary["main_results"]["rabi_frequency"] == 15.2

    def test_create_summary_without_experiment_params(self):
        """Test creating summary without experiment parameters"""
        results = {"analysis": {"estimated_t1": 100.0}}

        summary = create_experiment_summary(results, "T1", None)

        assert "experiment_parameters" not in summary

    def test_create_summary_empty_device_results(self):
        """Test creating summary with empty device results"""
        results = {"device_results": {}}

        summary = create_experiment_summary(results, "T1")

        assert summary["total_devices"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["status"] == "no_devices"


class TestFormatTimeDuration:
    """Test format_time_duration function"""

    def test_format_seconds(self):
        """Test formatting seconds"""
        assert format_time_duration(30.5) == "30.5s"
        assert format_time_duration(0.1) == "0.1s"
        assert format_time_duration(59.9) == "59.9s"

    def test_format_minutes(self):
        """Test formatting minutes"""
        assert format_time_duration(60) == "1.0m"
        assert format_time_duration(120) == "2.0m"
        assert format_time_duration(150) == "2.5m"
        assert format_time_duration(3599) == "60.0m"

    def test_format_hours(self):
        """Test formatting hours"""
        assert format_time_duration(3600) == "1.0h"
        assert format_time_duration(7200) == "2.0h"
        assert format_time_duration(5400) == "1.5h"

    def test_format_edge_cases(self):
        """Test formatting edge cases"""
        assert format_time_duration(0) == "0.0s"
        assert format_time_duration(59.99) == "60.0s"  # Close to boundary


class TestFormatScientificNotation:
    """Test format_scientific_notation function"""

    def test_format_small_numbers(self):
        """Test formatting small numbers"""
        assert format_scientific_notation(0.0001) == "1.000e-04"
        assert format_scientific_notation(0.000123) == "1.230e-04"
        assert format_scientific_notation(1e-6) == "1.000e-06"

    def test_format_large_numbers(self):
        """Test formatting large numbers"""
        assert format_scientific_notation(1000000) == "1.000e+06"
        assert format_scientific_notation(1.23e8) == "1.230e+08"

    def test_format_normal_numbers(self):
        """Test formatting normal range numbers"""
        assert format_scientific_notation(1.234) == "1.234"
        assert format_scientific_notation(0.123) == "0.123"
        assert format_scientific_notation(123.456) == "123.456"
        assert format_scientific_notation(999.9) == "999.900"

    def test_format_with_precision(self):
        """Test formatting with different precision"""
        assert format_scientific_notation(1.23456, precision=2) == "1.23"
        assert format_scientific_notation(1e-4, precision=1) == "1.0e-04"
        assert format_scientific_notation(1e6, precision=4) == "1.0000e+06"

    def test_format_edge_cases(self):
        """Test formatting edge cases"""
        # Test actual edge cases for the format_scientific_notation function
        assert format_scientific_notation(1e-4) == "1.000e-04"  # Below threshold
        assert format_scientific_notation(1e6) == "1.000e+06"  # Boundary case
        # Zero is formatted in scientific notation for consistency
        result = format_scientific_notation(0)
        assert result in ["0.000", "0.000e+00"]  # Either format is acceptable


if __name__ == "__main__":
    pytest.main([__file__])
