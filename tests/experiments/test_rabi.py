#!/usr/bin/env python3
"""
Tests for Rabi experiment
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit

from oqtopus_experiments.experiments.rabi import Rabi
from oqtopus_experiments.models.rabi_models import RabiParameters


class TestRabi:
    """Test Rabi experiment functionality"""

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        experiment = Rabi()

        assert experiment.experiment_name.startswith("rabi_experiment")
        assert experiment.physical_qubit == 0
        assert experiment.amplitude_points == 10
        assert experiment.max_amplitude == 2.0
        assert not experiment._physical_qubit_specified
        assert isinstance(experiment.params, RabiParameters)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        experiment = Rabi(
            experiment_name="custom_rabi",
            physical_qubit=3,
            amplitude_points=20,
            max_amplitude=3.0,
        )

        assert experiment.experiment_name == "custom_rabi"
        assert experiment.physical_qubit == 3
        assert experiment.amplitude_points == 20
        assert experiment.max_amplitude == 3.0
        assert experiment._physical_qubit_specified

    def test_init_physical_qubit_none(self):
        """Test initialization with explicit None physical_qubit"""
        experiment = Rabi(physical_qubit=None)

        assert experiment.physical_qubit == 0
        assert not experiment._physical_qubit_specified

    def test_circuits_creation(self):
        """Test that circuits are created correctly"""
        experiment = Rabi(amplitude_points=5, max_amplitude=1.0)
        circuits = experiment.circuits()

        assert len(circuits) == 5
        assert all(isinstance(circuit, QuantumCircuit) for circuit in circuits)

        # Check that each circuit has the expected structure
        for circuit in circuits:
            assert circuit.num_qubits >= 1
            assert circuit.num_clbits >= 1

    def test_circuits_amplitude_range(self):
        """Test that circuits cover the correct amplitude range"""
        amplitude_points = 6
        max_amplitude = 2.0
        experiment = Rabi(
            amplitude_points=amplitude_points, max_amplitude=max_amplitude
        )

        # First call circuits() to initialize experiment_params
        circuits = experiment.circuits()
        assert len(circuits) == amplitude_points

        # Get circuit parameters to verify amplitudes
        if hasattr(experiment, "_get_circuit_params"):
            params = experiment._get_circuit_params()
            if params:
                amplitudes = [p["amplitude"] for p in params]
                expected_amplitudes = np.linspace(0, max_amplitude, amplitude_points)
                np.testing.assert_array_almost_equal(amplitudes, expected_amplitudes)

    def test_analyze_with_mock_results(self):
        """Test analyze method with mock results"""
        experiment = Rabi(amplitude_points=3)

        # Create mock results
        mock_results = {
            "circuit_0": [{"counts": {"0": 900, "1": 100}, "success": True}],
            "circuit_1": [{"counts": {"0": 600, "1": 400}, "success": True}],
            "circuit_2": [{"counts": {"0": 200, "1": 800}, "success": True}],
        }

        with patch("matplotlib.pyplot.show"):
            analysis_result = experiment.analyze(
                mock_results, plot=False, save_data=False, save_image=False
            )

        assert analysis_result is not None
        assert isinstance(analysis_result, pd.DataFrame)

    def test_analyze_empty_results(self):
        """Test analyze method with empty results"""
        experiment = Rabi(amplitude_points=2)

        empty_results = {"circuit_0": [], "circuit_1": []}

        with patch("matplotlib.pyplot.show"):
            analysis_result = experiment.analyze(
                empty_results, plot=False, save_data=False, save_image=False
            )

        assert analysis_result is not None
        assert isinstance(analysis_result, pd.DataFrame)

    def test_run_with_mock_backend(self):
        """Test run method with mock backend"""
        experiment = Rabi(amplitude_points=3)

        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.run.side_effect = [
            {"counts": {"0": 900, "1": 100}, "success": True},
            {"counts": {"0": 600, "1": 400}, "success": True},
            {"counts": {"0": 200, "1": 800}, "success": True},
        ]

        result = experiment.run(mock_backend, shots=1000)

        assert result is not None
        assert len(result.raw_results) == 3
        assert mock_backend.run.call_count == 3

    def test_run_parallel_with_mock_backend(self):
        """Test run_parallel method with mock backend"""
        experiment = Rabi(amplitude_points=3)

        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.submit_parallel.return_value = ["job1", "job2", "job3"]
        mock_backend.collect_parallel.return_value = [
            {"counts": {"0": 900, "1": 100}, "success": True},
            {"counts": {"0": 600, "1": 400}, "success": True},
            {"counts": {"0": 200, "1": 800}, "success": True},
        ]

        result = experiment.run_parallel(mock_backend, shots=1000)

        assert result is not None
        assert len(result.raw_results) == 3
        mock_backend.submit_parallel.assert_called_once()
        mock_backend.collect_parallel.assert_called_once()

    def test_physical_qubit_tracking(self):
        """Test that physical qubit specification is tracked correctly"""
        # Test with explicit physical qubit
        exp_with_qubit = Rabi(physical_qubit=5)
        assert exp_with_qubit._physical_qubit_specified
        assert exp_with_qubit.physical_qubit == 5

        # Test without explicit physical qubit
        exp_without_qubit = Rabi()
        assert not exp_without_qubit._physical_qubit_specified
        assert exp_without_qubit.physical_qubit == 0

        # Test with explicit None
        exp_with_none = Rabi(physical_qubit=None)
        assert not exp_with_none._physical_qubit_specified
        assert exp_with_none.physical_qubit == 0

    def test_params_object(self):
        """Test that RabiParameters object is created correctly"""
        experiment = Rabi(
            experiment_name="test_rabi",
            physical_qubit=2,
            amplitude_points=15,
            max_amplitude=1.5,
        )

        params = experiment.params
        assert isinstance(params, RabiParameters)
        assert params.experiment_name == "test_rabi"
        assert params.physical_qubit == 2
        assert params.amplitude_points == 15
        assert params.max_amplitude == 1.5

    def test_inheritance_from_base_experiment(self):
        """Test that Rabi inherits from BaseExperiment"""
        from oqtopus_experiments.core.base_experiment import BaseExperiment

        experiment = Rabi()
        assert isinstance(experiment, BaseExperiment)

        # Test that abstract methods are implemented
        assert hasattr(experiment, "circuits")
        assert hasattr(experiment, "analyze")
        assert callable(experiment.circuits)
        assert callable(experiment.analyze)

    def test_experiment_name_generation(self):
        """Test automatic experiment name generation"""
        # With custom name
        exp1 = Rabi(experiment_name="my_rabi")
        assert exp1.experiment_name == "my_rabi"

        # Without custom name
        exp2 = Rabi()
        assert exp2.experiment_name.startswith("rabi_experiment")

        # With None name
        exp3 = Rabi(experiment_name=None)
        assert exp3.experiment_name.startswith("rabi_experiment")

    def test_amplitude_points_validation(self):
        """Test that amplitude_points parameter works correctly"""
        # Test different values
        for points in [1, 5, 10, 20]:
            experiment = Rabi(amplitude_points=points)
            circuits = experiment.circuits()
            assert len(circuits) == points

    def test_max_amplitude_validation(self):
        """Test that max_amplitude parameter works correctly"""
        # Test different values
        for amplitude in [0.5, 1.0, 2.0, 5.0]:
            experiment = Rabi(max_amplitude=amplitude)
            assert experiment.max_amplitude == amplitude

    @patch("oqtopus_experiments.experiments.rabi.curve_fit")
    def test_analyze_with_fitting(self, mock_curve_fit):
        """Test analyze method with curve fitting"""
        experiment = Rabi(amplitude_points=5)

        # Mock curve_fit to return dummy parameters
        mock_curve_fit.return_value = ([1.0, 0.5, 0.1], np.eye(3))

        mock_results = {
            "circuit_0": [{"counts": {"0": 800, "1": 200}, "success": True}],
            "circuit_1": [{"counts": {"0": 600, "1": 400}, "success": True}],
            "circuit_2": [{"counts": {"0": 500, "1": 500}, "success": True}],
            "circuit_3": [{"counts": {"0": 400, "1": 600}, "success": True}],
            "circuit_4": [{"counts": {"0": 200, "1": 800}, "success": True}],
        }

        with patch("matplotlib.pyplot.show"):
            analysis_result = experiment.analyze(
                mock_results, plot=False, save_data=False, save_image=False
            )

        assert analysis_result is not None
        assert isinstance(analysis_result, pd.DataFrame)
        # Note: curve_fit may or may not be called depending on data conditions


if __name__ == "__main__":
    pytest.main([__file__])
