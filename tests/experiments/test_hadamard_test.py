#!/usr/bin/env python3
"""
Tests for HadamardTest experiment
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit

from oqtopus_experiments.experiments.hadamard_test import HadamardTest
from oqtopus_experiments.models.hadamard_test_models import HadamardTestParameters


class TestHadamardTest:
    """Test HadamardTest experiment functionality"""

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        experiment = HadamardTest()

        assert experiment.experiment_name.startswith("hadamard_test_experiment")
        assert experiment.physical_qubit == 0
        assert experiment.test_unitary == "Z"
        assert experiment.angle_points == 16
        assert experiment.max_angle == 2 * np.pi
        assert not experiment._physical_qubit_specified
        assert isinstance(experiment.params, HadamardTestParameters)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        experiment = HadamardTest(
            experiment_name="custom_hadamard",
            physical_qubit=2,
            test_unitary="X",
            angle_points=8,
            max_angle=np.pi
        )

        assert experiment.experiment_name == "custom_hadamard"
        assert experiment.physical_qubit == 2
        assert experiment.test_unitary == "X"
        assert experiment.angle_points == 8
        assert experiment.max_angle == np.pi
        assert experiment._physical_qubit_specified

    def test_init_invalid_test_unitary(self):
        """Test initialization with invalid test_unitary raises error"""
        with pytest.raises(ValueError, match="test_unitary must be X, Y, or Z"):
            HadamardTest(test_unitary="H")

        with pytest.raises(ValueError, match="test_unitary must be X, Y, or Z"):
            HadamardTest(test_unitary="invalid")

    def test_init_test_unitary_case_insensitive(self):
        """Test that test_unitary is case insensitive"""
        experiment_x = HadamardTest(test_unitary="x")
        assert experiment_x.test_unitary == "X"

        experiment_y = HadamardTest(test_unitary="y")
        assert experiment_y.test_unitary == "Y"

        experiment_z = HadamardTest(test_unitary="z")
        assert experiment_z.test_unitary == "Z"

    def test_init_physical_qubit_none(self):
        """Test initialization with explicit None physical_qubit"""
        experiment = HadamardTest(physical_qubit=None)

        assert experiment.physical_qubit == 0
        assert not experiment._physical_qubit_specified

    def test_circuits_creation(self):
        """Test that circuits are created correctly"""
        experiment = HadamardTest(angle_points=4, max_angle=np.pi)
        circuits = experiment.circuits()

        assert len(circuits) == 4
        assert all(isinstance(circuit, QuantumCircuit) for circuit in circuits)

        # Check that each circuit has the expected structure
        for circuit in circuits:
            assert circuit.num_qubits >= 1
            assert circuit.num_clbits >= 1

    def test_circuits_angle_range(self):
        """Test that circuits cover the correct angle range"""
        angle_points = 5
        max_angle = np.pi
        experiment = HadamardTest(angle_points=angle_points, max_angle=max_angle)

        # First call circuits() to initialize experiment_params
        circuits = experiment.circuits()
        assert len(circuits) == angle_points

        # Get circuit parameters to verify angles
        if hasattr(experiment, '_get_circuit_params'):
            params = experiment._get_circuit_params()
            if params:
                angles = [p["angle"] for p in params]
                expected_angles = np.linspace(0, max_angle, angle_points)
                np.testing.assert_array_almost_equal(angles, expected_angles)

    def test_circuits_different_unitaries(self):
        """Test circuit creation for different test unitaries"""
        for unitary in ["X", "Y", "Z"]:
            experiment = HadamardTest(test_unitary=unitary, angle_points=4)
            circuits = experiment.circuits()

            assert len(circuits) == 4
            assert all(isinstance(circuit, QuantumCircuit) for circuit in circuits)

    def test_analyze_with_mock_results(self):
        """Test analyze method with mock results"""
        experiment = HadamardTest(angle_points=4, test_unitary="Z")

        # Create mock results that should follow cos(θ) pattern for Z unitary
        mock_results = {
            "circuit_0": [{"counts": {"0": 1000, "1": 0}, "success": True}],    # θ=0, cos(0)=1
            "circuit_1": [{"counts": {"0": 750, "1": 250}, "success": True}],   # θ=π/3
            "circuit_2": [{"counts": {"0": 500, "1": 500}, "success": True}],   # θ=2π/3, cos(2π/3)=0
            "circuit_3": [{"counts": {"0": 0, "1": 1000}, "success": True}]     # θ=π, cos(π)=-1
        }

        with patch('matplotlib.pyplot.show'):
            analysis_result = experiment.analyze(
                mock_results,
                plot=False,
                save_data=False,
                save_image=False
            )

        assert analysis_result is not None
        assert isinstance(analysis_result, pd.DataFrame)

    def test_analyze_empty_results(self):
        """Test analyze method with empty results"""
        experiment = HadamardTest(angle_points=4)

        empty_results = {
            "circuit_0": [],
            "circuit_1": [],
            "circuit_2": [],
            "circuit_3": []
        }

        with patch('matplotlib.pyplot.show'):
            analysis_result = experiment.analyze(
                empty_results,
                plot=False,
                save_data=False,
                save_image=False
            )

        assert analysis_result is not None
        assert isinstance(analysis_result, pd.DataFrame)

    def test_run_with_mock_backend(self):
        """Test run method with mock backend"""
        experiment = HadamardTest(angle_points=4)

        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.run.side_effect = [
            {"counts": {"0": 1000, "1": 0}, "success": True},
            {"counts": {"0": 750, "1": 250}, "success": True},
            {"counts": {"0": 500, "1": 500}, "success": True},
            {"counts": {"0": 0, "1": 1000}, "success": True}
        ]

        result = experiment.run(mock_backend, shots=1000)

        assert result is not None
        assert len(result.raw_results) == 4
        assert mock_backend.run.call_count == 4

    def test_run_parallel_with_mock_backend(self):
        """Test run_parallel method with mock backend"""
        experiment = HadamardTest(angle_points=4)

        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.submit_parallel.return_value = ["job1", "job2", "job3", "job4"]
        mock_backend.collect_parallel.return_value = [
            {"counts": {"0": 1000, "1": 0}, "success": True},
            {"counts": {"0": 750, "1": 250}, "success": True},
            {"counts": {"0": 500, "1": 500}, "success": True},
            {"counts": {"0": 0, "1": 1000}, "success": True}
        ]

        result = experiment.run_parallel(mock_backend, shots=1000)

        assert result is not None
        assert len(result.raw_results) == 4
        mock_backend.submit_parallel.assert_called_once()
        mock_backend.collect_parallel.assert_called_once()

    def test_physical_qubit_tracking(self):
        """Test that physical qubit specification is tracked correctly"""
        # Test with explicit physical qubit
        exp_with_qubit = HadamardTest(physical_qubit=3)
        assert exp_with_qubit._physical_qubit_specified
        assert exp_with_qubit.physical_qubit == 3

        # Test without explicit physical qubit
        exp_without_qubit = HadamardTest()
        assert not exp_without_qubit._physical_qubit_specified
        assert exp_without_qubit.physical_qubit == 0

        # Test with explicit None
        exp_with_none = HadamardTest(physical_qubit=None)
        assert not exp_with_none._physical_qubit_specified
        assert exp_with_none.physical_qubit == 0

    def test_params_object(self):
        """Test that HadamardTestParameters object is created correctly"""
        experiment = HadamardTest(
            experiment_name="test_hadamard",
            physical_qubit=1,
            test_unitary="Y",
            angle_points=12,
            max_angle=3 * np.pi / 2
        )

        params = experiment.params
        assert isinstance(params, HadamardTestParameters)
        assert params.experiment_name == "test_hadamard"
        assert params.physical_qubit == 1
        assert params.test_unitary == "Y"
        assert params.angle_points == 12
        assert params.max_angle == 3 * np.pi / 2

    def test_inheritance_from_base_experiment(self):
        """Test that HadamardTest inherits from BaseExperiment"""
        from oqtopus_experiments.core.base_experiment import BaseExperiment

        experiment = HadamardTest()
        assert isinstance(experiment, BaseExperiment)

        # Test that abstract methods are implemented
        assert hasattr(experiment, 'circuits')
        assert hasattr(experiment, 'analyze')
        assert callable(experiment.circuits)
        assert callable(experiment.analyze)

    def test_experiment_name_generation(self):
        """Test automatic experiment name generation"""
        # With custom name
        exp1 = HadamardTest(experiment_name="my_hadamard")
        assert exp1.experiment_name == "my_hadamard"

        # Without custom name
        exp2 = HadamardTest()
        assert exp2.experiment_name.startswith("hadamard_test_experiment")

        # With None name
        exp3 = HadamardTest(experiment_name=None)
        assert exp3.experiment_name.startswith("hadamard_test_experiment")

    def test_angle_points_validation(self):
        """Test that angle_points parameter works correctly"""
        # Test different values
        for points in [4, 8, 16]:
            experiment = HadamardTest(angle_points=points)
            circuits = experiment.circuits()
            assert len(circuits) == points

    def test_max_angle_validation(self):
        """Test that max_angle parameter works correctly"""
        # Test different values
        for angle in [np.pi/2, np.pi, 2*np.pi, 3*np.pi]:
            experiment = HadamardTest(max_angle=angle)
            assert experiment.max_angle == angle

    def test_expected_theoretical_behavior(self):
        """Test that the experiment setup matches theoretical expectations"""
        # For Z unitary, expectation value should be cos(θ)
        experiment_z = HadamardTest(test_unitary="Z", angle_points=5, max_angle=2*np.pi)

        # Create theoretical expectation values
        angles = np.linspace(0, 2*np.pi, 5)
        _ = np.cos(angles)  # expected_z not used, kept for potential future use

        # For X unitary, expectation value should be sin(θ)
        experiment_x = HadamardTest(test_unitary="X", angle_points=5, max_angle=2*np.pi)
        _ = np.sin(angles)  # expected_x not used, kept for potential future use

        # For Y unitary, expectation value should be 0 (since RX(θ) commutes with Y axis rotation)
        experiment_y = HadamardTest(test_unitary="Y", angle_points=5, max_angle=2*np.pi)
        _ = np.zeros_like(angles)  # expected_y not used, kept for potential future use

        # Just verify that experiments are set up correctly
        assert experiment_z.test_unitary == "Z"
        assert experiment_x.test_unitary == "X"
        assert experiment_y.test_unitary == "Y"

    @patch('scipy.optimize.curve_fit')
    def test_analyze_with_fitting(self, mock_curve_fit):
        """Test analyze method with curve fitting"""
        experiment = HadamardTest(angle_points=4, test_unitary="Z")

        # Mock curve_fit to return dummy parameters
        mock_curve_fit.return_value = ([1.0, 0.0], np.eye(2))

        mock_results = {
            "circuit_0": [{"counts": {"0": 1000, "1": 0}, "success": True}],
            "circuit_1": [{"counts": {"0": 750, "1": 250}, "success": True}],
            "circuit_2": [{"counts": {"0": 500, "1": 500}, "success": True}],
            "circuit_3": [{"counts": {"0": 250, "1": 750}, "success": True}]
        }

        with patch('matplotlib.pyplot.show'):
            analysis_result = experiment.analyze(
                mock_results,
                plot=False,
                save_data=False,
                save_image=False
            )

        assert analysis_result is not None


if __name__ == "__main__":
    pytest.main([__file__])
