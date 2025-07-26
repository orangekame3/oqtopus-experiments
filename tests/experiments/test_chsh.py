#!/usr/bin/env python3
"""
Tests for CHSH experiment
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from qiskit import QuantumCircuit

from oqtopus_experiments.experiments.chsh import CHSH
from oqtopus_experiments.models.chsh_models import CHSHParameters


class TestCHSH:
    """Test CHSH experiment functionality"""

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        experiment = CHSH()

        assert experiment.experiment_name.startswith("chsh_experiment")
        assert experiment.params.physical_qubit_0 == 0
        assert experiment.params.physical_qubit_1 == 1
        assert experiment.params.shots_per_circuit == 1000
        assert experiment.params.theta == 0.0
        assert not experiment._physical_qubits_specified
        assert isinstance(experiment.params, CHSHParameters)

        # Check default measurement angles
        expected_angles = {
            "alice_0": 0.0,
            "alice_1": 45.0,
            "bob_0": 22.5,
            "bob_1": 67.5,
        }
        assert experiment.params.measurement_angles == expected_angles

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        custom_angles = {
            "alice_0": 10.0,
            "alice_1": 55.0,
            "bob_0": 32.5,
            "bob_1": 77.5,
        }

        experiment = CHSH(
            experiment_name="custom_chsh",
            physical_qubit_0=3,
            physical_qubit_1=5,
            shots_per_circuit=2048,
            measurement_angles=custom_angles,
            theta=0.5,
        )

        assert experiment.experiment_name == "custom_chsh"
        assert experiment.params.physical_qubit_0 == 3
        assert experiment.params.physical_qubit_1 == 5
        assert experiment.params.shots_per_circuit == 2048
        assert experiment.params.theta == 0.5
        assert experiment._physical_qubits_specified
        assert experiment.params.measurement_angles == custom_angles

    def test_init_partial_physical_qubits(self):
        """Test initialization with only one physical qubit specified"""
        experiment = CHSH(physical_qubit_0=2)

        assert experiment.params.physical_qubit_0 == 2
        assert experiment.params.physical_qubit_1 == 1  # Default
        assert not experiment._physical_qubits_specified

        experiment = CHSH(physical_qubit_1=3)

        assert experiment.params.physical_qubit_0 == 0  # Default
        assert experiment.params.physical_qubit_1 == 3
        assert not experiment._physical_qubits_specified

    def test_init_explicit_none_physical_qubits(self):
        """Test initialization with explicit None physical qubits"""
        experiment = CHSH(physical_qubit_0=None, physical_qubit_1=None)

        assert experiment.params.physical_qubit_0 == 0
        assert experiment.params.physical_qubit_1 == 1
        assert not experiment._physical_qubits_specified

    def test_circuits_creation(self):
        """Test that circuits are created correctly"""
        experiment = CHSH()
        circuits = experiment.circuits()

        # CHSH requires 4 measurement settings
        assert len(circuits) == 4
        assert all(isinstance(circuit, QuantumCircuit) for circuit in circuits)

        # Check that each circuit has the expected structure
        for circuit in circuits:
            assert circuit.num_qubits >= 2  # At least 2 qubits for Bell state
            assert circuit.num_clbits >= 2  # At least 2 classical bits for measurement

    def test_circuits_with_custom_angles(self):
        """Test circuits creation with custom measurement angles"""
        custom_angles = {
            "alice_0": 15.0,
            "alice_1": 60.0,
            "bob_0": 30.0,
            "bob_1": 75.0,
        }

        experiment = CHSH(measurement_angles=custom_angles)
        circuits = experiment.circuits()

        assert len(circuits) == 4
        assert all(isinstance(circuit, QuantumCircuit) for circuit in circuits)

    def test_analyze_with_mock_results(self):
        """Test analyze method with mock results"""
        experiment = CHSH()

        # Create mock results for 4 measurement settings
        mock_results = {
            "circuit_0": [{"counts": {"00": 400, "11": 600}, "success": True}],
            "circuit_1": [{"counts": {"00": 300, "11": 700}, "success": True}],
            "circuit_2": [{"counts": {"00": 350, "11": 650}, "success": True}],
            "circuit_3": [{"counts": {"00": 450, "11": 550}, "success": True}],
        }

        with patch("matplotlib.pyplot.show"):
            analysis_result = experiment.analyze(
                mock_results, plot=False, save_data=False, save_image=False
            )

        assert analysis_result is not None
        assert isinstance(analysis_result, pd.DataFrame)

    def test_analyze_empty_results(self):
        """Test analyze method with empty results"""
        experiment = CHSH()

        empty_results = {
            "circuit_0": [],
            "circuit_1": [],
            "circuit_2": [],
            "circuit_3": [],
        }

        with patch("matplotlib.pyplot.show"):
            analysis_result = experiment.analyze(
                empty_results, plot=False, save_data=False, save_image=False
            )

        assert analysis_result is not None
        assert isinstance(analysis_result, pd.DataFrame)

    def test_run_with_mock_backend(self):
        """Test run method with mock backend"""
        experiment = CHSH()

        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.run.side_effect = [
            {"counts": {"00": 400, "11": 600}, "success": True},
            {"counts": {"00": 300, "11": 700}, "success": True},
            {"counts": {"00": 350, "11": 650}, "success": True},
            {"counts": {"00": 450, "11": 550}, "success": True},
        ]

        result = experiment.run(mock_backend, shots=1000)

        assert result is not None
        assert len(result.raw_results) == 4
        assert mock_backend.run.call_count == 4

    def test_run_parallel_with_mock_backend(self):
        """Test run_parallel method with mock backend"""
        experiment = CHSH()

        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.submit_parallel.return_value = ["job1", "job2", "job3", "job4"]
        mock_backend.collect_parallel.return_value = [
            {"counts": {"00": 400, "11": 600}, "success": True},
            {"counts": {"00": 300, "11": 700}, "success": True},
            {"counts": {"00": 350, "11": 650}, "success": True},
            {"counts": {"00": 450, "11": 550}, "success": True},
        ]

        result = experiment.run_parallel(mock_backend, shots=1000)

        assert result is not None
        assert len(result.raw_results) == 4
        mock_backend.submit_parallel.assert_called_once()
        mock_backend.collect_parallel.assert_called_once()

    def test_physical_qubit_tracking(self):
        """Test that physical qubit specification is tracked correctly"""
        # Test with both explicit physical qubits
        exp_with_qubits = CHSH(physical_qubit_0=2, physical_qubit_1=4)
        assert exp_with_qubits._physical_qubits_specified
        assert exp_with_qubits.params.physical_qubit_0 == 2
        assert exp_with_qubits.params.physical_qubit_1 == 4

        # Test without explicit physical qubits
        exp_without_qubits = CHSH()
        assert not exp_without_qubits._physical_qubits_specified
        assert exp_without_qubits.params.physical_qubit_0 == 0
        assert exp_without_qubits.params.physical_qubit_1 == 1

        # Test with only one explicit qubit
        exp_partial = CHSH(physical_qubit_0=3)
        assert not exp_partial._physical_qubits_specified
        assert exp_partial.params.physical_qubit_0 == 3
        assert exp_partial.params.physical_qubit_1 == 1

    def test_params_object(self):
        """Test that CHSHParameters object is created correctly"""
        experiment = CHSH(
            experiment_name="test_chsh",
            physical_qubit_0=1,
            physical_qubit_1=2,
            shots_per_circuit=512,
            theta=0.25,
        )

        params = experiment.params
        assert isinstance(params, CHSHParameters)
        assert params.experiment_name == "test_chsh"
        assert params.physical_qubit_0 == 1
        assert params.physical_qubit_1 == 2
        assert params.shots_per_circuit == 512
        assert params.theta == 0.25

    def test_inheritance_from_base_experiment(self):
        """Test that CHSH inherits from BaseExperiment"""
        from oqtopus_experiments.core.base_experiment import BaseExperiment

        experiment = CHSH()
        assert isinstance(experiment, BaseExperiment)

        # Test that abstract methods are implemented
        assert hasattr(experiment, "circuits")
        assert hasattr(experiment, "analyze")
        assert callable(experiment.circuits)
        assert callable(experiment.analyze)

    def test_experiment_name_generation(self):
        """Test automatic experiment name generation"""
        # With custom name
        exp1 = CHSH(experiment_name="my_chsh")
        assert exp1.experiment_name == "my_chsh"

        # Without custom name
        exp2 = CHSH()
        assert exp2.experiment_name.startswith("chsh_experiment")

        # With None name
        exp3 = CHSH(experiment_name=None)
        assert exp3.experiment_name.startswith("chsh_experiment")

    def test_measurement_angles_validation(self):
        """Test measurement angles handling"""
        # Test with custom angles
        custom_angles = {
            "alice_0": 5.0,
            "alice_1": 50.0,
            "bob_0": 27.5,
            "bob_1": 72.5,
        }
        experiment = CHSH(measurement_angles=custom_angles)
        assert experiment.params.measurement_angles == custom_angles

        # Test with None (should use defaults)
        experiment = CHSH(measurement_angles=None)
        expected_defaults = {
            "alice_0": 0.0,
            "alice_1": 45.0,
            "bob_0": 22.5,
            "bob_1": 67.5,
        }
        assert experiment.params.measurement_angles == expected_defaults

    def test_shots_per_circuit_validation(self):
        """Test that shots_per_circuit parameter works correctly"""
        # Test different values
        for shots in [100, 500, 1024, 2048]:
            experiment = CHSH(shots_per_circuit=shots)
            assert experiment.params.shots_per_circuit == shots

    def test_theta_parameter_validation(self):
        """Test that theta parameter works correctly"""
        # Test different values (no validation constraints in the model)
        test_values = [0.0, 0.25, 0.5, 1.0, 0.7853981633974483]  # pi/4 as literal
        for theta in test_values:
            experiment = CHSH(theta=theta)
            assert experiment.params.theta == pytest.approx(theta, rel=1e-9)

    def test_analyze_with_realistic_bell_violation(self):
        """Test analyze method with realistic Bell violation data"""
        experiment = CHSH()

        # Create mock results that should violate Bell inequality
        # Using correlations that would give S ≈ 2.83 (near quantum maximum)
        mock_results = {
            "circuit_0": [
                {"counts": {"00": 425, "01": 75, "10": 75, "11": 425}, "success": True}
            ],
            "circuit_1": [
                {"counts": {"00": 425, "01": 75, "10": 75, "11": 425}, "success": True}
            ],
            "circuit_2": [
                {"counts": {"00": 425, "01": 75, "10": 75, "11": 425}, "success": True}
            ],
            "circuit_3": [
                {"counts": {"00": 75, "01": 425, "10": 425, "11": 75}, "success": True}
            ],
        }

        with patch("matplotlib.pyplot.show"):
            analysis_result = experiment.analyze(
                mock_results, plot=False, save_data=False, save_image=False
            )

        assert analysis_result is not None
        assert isinstance(analysis_result, pd.DataFrame)

    def test_analyze_with_classical_correlation(self):
        """Test analyze method with classical correlation data"""
        experiment = CHSH()

        # Create mock results with classical correlation (S ≤ 2)
        mock_results = {
            "circuit_0": [
                {"counts": {"00": 500, "01": 0, "10": 0, "11": 500}, "success": True}
            ],
            "circuit_1": [
                {"counts": {"00": 500, "01": 0, "10": 0, "11": 500}, "success": True}
            ],
            "circuit_2": [
                {"counts": {"00": 500, "01": 0, "10": 0, "11": 500}, "success": True}
            ],
            "circuit_3": [
                {"counts": {"00": 500, "01": 0, "10": 0, "11": 500}, "success": True}
            ],
        }

        with patch("matplotlib.pyplot.show"):
            analysis_result = experiment.analyze(
                mock_results, plot=False, save_data=False, save_image=False
            )

        assert analysis_result is not None
        assert isinstance(analysis_result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])
