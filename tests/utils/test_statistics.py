#!/usr/bin/env python3
"""
Tests for statistics utilities
"""

import warnings

import numpy as np
import pytest

from oqtopus_experiments.utils.statistics import (
    calculate_fidelity,
    calculate_p0_probability,
    calculate_p1_probability,
    calculate_probability,
    calculate_process_fidelity,
    calculate_z_expectation,
    damped_oscillation,
    echo_decay,
    estimate_parameters_with_quality,
    exponential_decay,
    rabi_oscillation,
)


class TestCalculateProbability:
    """Test calculate_probability function"""

    def test_single_qubit_state_0(self):
        """Test probability calculation for state |0⟩"""
        counts = {"0": 800, "1": 200}
        prob = calculate_probability(counts, "0")
        assert prob == 0.8

    def test_single_qubit_state_1(self):
        """Test probability calculation for state |1⟩"""
        counts = {"0": 300, "1": 700}
        prob = calculate_probability(counts, "1")
        assert prob == 0.7

    def test_two_qubit_states(self):
        """Test probability calculation for two-qubit states"""
        counts = {"00": 250, "01": 250, "10": 250, "11": 250}

        assert calculate_probability(counts, "00") == 0.25
        assert calculate_probability(counts, "01") == 0.25
        assert calculate_probability(counts, "10") == 0.25
        assert calculate_probability(counts, "11") == 0.25

    def test_missing_state(self):
        """Test probability calculation for missing state"""
        counts = {"0": 1000}
        prob = calculate_probability(counts, "1")
        assert prob == 0.0

    def test_empty_counts(self):
        """Test probability calculation with empty counts"""
        counts = {}
        prob = calculate_probability(counts, "0")
        assert prob == 0.0

    def test_zero_total_shots(self):
        """Test probability calculation with zero total shots"""
        counts = {"0": 0, "1": 0}
        prob = calculate_probability(counts, "0")
        assert prob == 0.0

    def test_all_shots_in_target_state(self):
        """Test probability calculation when all shots are in target state"""
        counts = {"0": 1000}
        prob = calculate_probability(counts, "0")
        assert prob == 1.0


class TestSingleQubitProbabilities:
    """Test single-qubit probability functions"""

    def test_calculate_p0_probability(self):
        """Test P(|0⟩) calculation"""
        counts = {"0": 600, "1": 400}
        prob = calculate_p0_probability(counts)
        assert prob == 0.6

    def test_calculate_p1_probability(self):
        """Test P(|1⟩) calculation"""
        counts = {"0": 300, "1": 700}
        prob = calculate_p1_probability(counts)
        assert prob == 0.7

    def test_probabilities_sum_to_one(self):
        """Test that P(|0⟩) + P(|1⟩) = 1"""
        counts = {"0": 456, "1": 544}
        p0 = calculate_p0_probability(counts)
        p1 = calculate_p1_probability(counts)
        assert abs(p0 + p1 - 1.0) < 1e-10


class TestTwoQubitProbabilities:
    """Test two-qubit probability functions"""

    def test_calculate_p00_probability(self):
        """Test P(|00⟩) calculation"""
        counts = {"00": 400, "01": 200, "10": 200, "11": 200}
        prob = calculate_probability(counts, "00")
        assert prob == 0.4

    def test_calculate_p01_probability(self):
        """Test P(|01⟩) calculation"""
        counts = {"00": 200, "01": 300, "10": 200, "11": 300}
        prob = calculate_probability(counts, "01")
        assert prob == 0.3

    def test_calculate_p10_probability(self):
        """Test P(|10⟩) calculation"""
        counts = {"00": 200, "01": 200, "10": 350, "11": 250}
        prob = calculate_probability(counts, "10")
        assert prob == 0.35

    def test_calculate_p11_probability(self):
        """Test P(|11⟩) calculation"""
        counts = {"00": 100, "01": 200, "10": 200, "11": 500}
        prob = calculate_probability(counts, "11")
        assert prob == 0.5

    def test_two_qubit_probabilities_sum_to_one(self):
        """Test that all two-qubit probabilities sum to 1"""
        counts = {"00": 123, "01": 234, "10": 345, "11": 298}

        p00 = calculate_probability(counts, "00")
        p01 = calculate_probability(counts, "01")
        p10 = calculate_probability(counts, "10")
        p11 = calculate_probability(counts, "11")

        total = p00 + p01 + p10 + p11
        assert abs(total - 1.0) < 1e-10


class TestExpectationValueZ:
    """Test Z expectation value calculation"""

    def test_all_zero_state(self):
        """Test expectation value when all measurements are |0⟩"""
        counts = {"0": 1000, "1": 0}
        expectation = calculate_z_expectation(counts)
        assert expectation == 1.0

    def test_all_one_state(self):
        """Test expectation value when all measurements are |1⟩"""
        counts = {"0": 0, "1": 1000}
        expectation = calculate_z_expectation(counts)
        assert expectation == -1.0

    def test_equal_superposition(self):
        """Test expectation value for equal superposition"""
        counts = {"0": 500, "1": 500}
        expectation = calculate_z_expectation(counts)
        assert expectation == 0.0

    def test_general_case(self):
        """Test expectation value for general case"""
        counts = {"0": 800, "1": 200}
        expectation = calculate_z_expectation(counts)
        # ⟨Z⟩ = P(0) - P(1) = 0.8 - 0.2 = 0.6
        assert expectation == pytest.approx(0.6)

    def test_empty_counts(self):
        """Test expectation value with empty counts"""
        counts = {}
        expectation = calculate_z_expectation(counts)
        assert expectation == 0.0


class TestEstimateParametersWithQuality:
    """Test estimate_parameters_with_quality function"""

    def test_exponential_decay_fitting(self):
        """Test fitting exponential decay"""
        # Generate test data
        t_data = np.linspace(0, 5, 20)
        A_true, T_true, offset_true = 1.0, 2.0, 0.1
        y_data = exponential_decay(t_data, A_true, T_true, offset_true)

        initial_params = [0.8, 1.5, 0.05]
        result = estimate_parameters_with_quality(
            t_data, y_data, exponential_decay, initial_params
        )

        assert result["success"] is True
        assert len(result["fitted_params"]) == 3
        assert result["fit_quality"]["r_squared"] > 0.99

        # Check parameter accuracy
        fitted_A, fitted_T, fitted_offset = result["fitted_params"]
        assert abs(fitted_A - A_true) < 0.1
        assert abs(fitted_T - T_true) < 0.1
        assert abs(fitted_offset - offset_true) < 0.1

    def test_rabi_oscillation_fitting(self):
        """Test fitting Rabi oscillation"""
        # Generate test data
        t_data = np.linspace(0, 2, 30)
        A_true, freq_true, phase_true, offset_true = 0.5, 2.0, 0.0, 0.5
        y_data = rabi_oscillation(t_data, A_true, freq_true, phase_true, offset_true)

        initial_params = [0.4, 1.8, 0.1, 0.45]

        # Suppress OptimizeWarning for covariance estimation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = estimate_parameters_with_quality(
                t_data, y_data, rabi_oscillation, initial_params
            )

        assert result["success"] is True
        assert len(result["fitted_params"]) == 4
        assert result["fit_quality"]["r_squared"] > 0.99

    def test_insufficient_data_points(self):
        """Test handling of insufficient data points"""
        x_data = np.array([0, 1])  # Only 2 points
        y_data = np.array([1, 0.5])
        initial_params = [1.0, 2.0, 0.0]  # 3 parameters

        result = estimate_parameters_with_quality(
            x_data, y_data, exponential_decay, initial_params
        )

        assert result["success"] is False
        assert "Insufficient data points" in result["error"]
        assert result["fitted_params"] == initial_params

    def test_invalid_data_handling(self):
        """Test handling of invalid data (NaN, inf)"""
        x_data = np.array([0, 1, 2, 3, 4])
        y_data = np.array(
            [1, np.nan, np.inf, 0.3, 0.1]
        )  # 3 valid points: (0,1), (3,0.3), (4,0.1)
        initial_params = [1.0, 2.0, 0.0]

        # Suppress OptimizeWarning for covariance estimation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = estimate_parameters_with_quality(
                x_data, y_data, exponential_decay, initial_params
            )

        # Should still succeed with sufficient valid data points (3 valid >= 3 params)
        assert result["success"] is True

    def test_all_invalid_data(self):
        """Test handling when all data is invalid"""
        x_data = np.array([0, 1, 2])
        y_data = np.array([np.nan, np.inf, np.nan])
        initial_params = [1.0, 2.0, 0.0]

        result = estimate_parameters_with_quality(
            x_data, y_data, exponential_decay, initial_params
        )

        assert result["success"] is False
        assert "No valid data points" in result["error"]

    def test_parameter_bounds(self):
        """Test fitting with parameter bounds"""
        t_data = np.linspace(0, 3, 15)
        y_data = exponential_decay(t_data, 1.0, 2.0, 0.1)

        initial_params = [0.8, 1.5, 0.05]
        bounds = ([0, 0, 0], [2, 5, 1])  # Lower and upper bounds

        result = estimate_parameters_with_quality(
            t_data, y_data, exponential_decay, initial_params, param_bounds=bounds
        )

        assert result["success"] is True
        # All parameters should be within bounds
        fitted_params = result["fitted_params"]
        for i, param in enumerate(fitted_params):
            assert bounds[0][i] <= param <= bounds[1][i]


class TestFittingFunctions:
    """Test individual fitting functions"""

    def test_exponential_decay(self):
        """Test exponential decay function"""
        t = np.array([0, 1, 2])
        result = exponential_decay(t, A=1.0, T=1.0, offset=0.0)
        expected = np.array([1.0, np.exp(-1), np.exp(-2)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_damped_oscillation(self):
        """Test damped oscillation function"""
        t = np.array([0, 0.25, 0.5])
        result = damped_oscillation(
            t, A=1.0, T2_star=1.0, frequency=1.0, phase=0.0, offset=0.0
        )
        # At t=0: 1*exp(0)*cos(0) = 1
        # At t=0.25: 1*exp(-0.25)*cos(π/2) ≈ 0
        # At t=0.5: 1*exp(-0.5)*cos(π) ≈ -exp(-0.5)
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1]) < 1e-10
        assert result[2] < 0

    def test_echo_decay(self):
        """Test echo decay function"""
        t = np.array([0, 1, 2])
        result = echo_decay(t, A=1.0, T2=1.0, offset=0.1)
        expected = np.array([1.1, 0.1 + np.exp(-1), 0.1 + np.exp(-2)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rabi_oscillation(self):
        """Test Rabi oscillation function"""
        t = np.array([0, 0.25, 0.5])
        result = rabi_oscillation(t, A=1.0, rabi_freq=1.0, phase=0.0, offset=0.5)
        # At t=0: 1*cos(0) + 0.5 = 1.5
        # At t=0.25: 1*cos(π/2) + 0.5 = 0.5
        # At t=0.5: 1*cos(π) + 0.5 = -0.5
        expected = np.array([1.5, 0.5, -0.5])
        np.testing.assert_array_almost_equal(result, expected)


class TestFidelityCalculations:
    """Test fidelity calculation functions"""

    def test_calculate_fidelity_perfect_match(self):
        """Test fidelity calculation with perfect match"""
        theoretical = {"0": 0.6, "1": 0.4}
        measured = {"0": 0.6, "1": 0.4}

        fidelity = calculate_fidelity(theoretical, measured)
        assert abs(fidelity - 1.0) < 1e-10

    def test_calculate_fidelity_orthogonal_states(self):
        """Test fidelity calculation with orthogonal states"""
        theoretical = {"0": 1.0, "1": 0.0}
        measured = {"0": 0.0, "1": 1.0}

        fidelity = calculate_fidelity(theoretical, measured)
        assert abs(fidelity - 0.0) < 1e-10

    def test_calculate_fidelity_partial_overlap(self):
        """Test fidelity calculation with partial overlap"""
        theoretical = {"0": 0.8, "1": 0.2}
        measured = {"0": 0.6, "1": 0.4}

        fidelity = calculate_fidelity(theoretical, measured)
        expected = np.sqrt(0.8 * 0.6) + np.sqrt(0.2 * 0.4)
        assert abs(fidelity - expected) < 1e-10

    def test_calculate_fidelity_missing_states(self):
        """Test fidelity calculation with missing states"""
        theoretical = {"0": 0.7, "1": 0.3}
        measured = {"0": 0.8}  # Missing state "1"

        fidelity = calculate_fidelity(theoretical, measured)
        expected = np.sqrt(0.7 * 0.8) + np.sqrt(0.3 * 0.0)
        assert abs(fidelity - expected) < 1e-10

    def test_calculate_fidelity_empty_distributions(self):
        """Test fidelity calculation with empty distributions"""
        fidelity = calculate_fidelity({}, {"0": 1.0})
        assert fidelity == 0.0

        fidelity = calculate_fidelity({"0": 1.0}, {})
        assert fidelity == 0.0

    def test_calculate_process_fidelity(self):
        """Test process fidelity calculation"""
        ideal = {"0": 0.9, "1": 0.1}
        actual = {"0": 0.8, "1": 0.2}

        process_fidelity = calculate_process_fidelity(ideal, actual)
        expected_fidelity = calculate_fidelity(ideal, actual)

        assert abs(process_fidelity - expected_fidelity) < 1e-10


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_none_input(self):
        """Test functions with None input"""
        assert calculate_probability(None, "0") == 0.0
        assert calculate_p0_probability(None) == 0.0
        assert calculate_z_expectation(None) == 0.0

    def test_string_counts(self):
        """Test with string count values (should be converted to int)"""
        counts = {"0": "800", "1": "200"}
        # This should not work as the function expects int values
        with pytest.raises(TypeError):
            calculate_probability(counts, "0")

    def test_negative_counts(self):
        """Test with negative count values"""
        counts = {"0": -100, "1": 200}
        # The function should still work but give unexpected results
        prob = calculate_probability(counts, "0")
        assert prob == -1.0  # -100 / 100 = -1.0

    def test_very_large_numbers(self):
        """Test with very large count numbers"""
        counts = {"0": 10**9, "1": 10**9}
        prob = calculate_probability(counts, "0")
        assert abs(prob - 0.5) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
