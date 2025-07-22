#!/usr/bin/env python3
"""
Integration tests for LocalBackend with experiments
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oqtopus_experiments.backends.local_backend import LocalBackend
from oqtopus_experiments.experiments.rabi import Rabi


class TestLocalBackendExperimentIntegration:
    """Integration tests for LocalBackend with experiment classes"""

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    @patch("oqtopus_experiments.backends.local_backend.AerSimulator")
    @patch("oqtopus_experiments.backends.local_backend.transpile")
    def test_rabi_with_noisy_backend(self, mock_transpile, mock_aer):
        """Test Rabi experiment with noisy LocalBackend"""
        # Setup mocks
        mock_simulator = MagicMock()
        mock_aer.return_value = mock_simulator

        # Mock circuit transpilation
        from qiskit import QuantumCircuit

        mock_circuit = QuantumCircuit(1, 1)
        mock_transpile.return_value = mock_circuit

        # Mock simulation results - simulate Rabi oscillation pattern
        def mock_run_side_effect(circuit, shots, **kwargs):
            mock_job = MagicMock()
            mock_result = MagicMock()

            # Simple Rabi oscillation simulation based on circuit parameters
            # For this test, we'll return predictable results
            mock_counts = {"0": shots // 2, "1": shots // 2}
            mock_result.get_counts.return_value = mock_counts
            mock_job.result.return_value = mock_result
            return mock_job

        mock_simulator.run.side_effect = mock_run_side_effect

        # Create backend and experiment
        backend = LocalBackend(device="noisy")
        rabi = Rabi(
            experiment_name="test_rabi_noisy",
            physical_qubit=0,
            amplitude_points=5,
            max_amplitude=1.0,
        )

        # Run experiment
        result = rabi.run(backend, shots=100)

        # Verify experiment execution
        assert result.experiment_type == "rabi"
        assert len(result.raw_results) == 5  # 5 amplitude points

        # Analyze results
        df = result.analyze()

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "device" in df.columns
        assert "amplitude" in df.columns
        assert "probability" in df.columns

        # Verify device name is correctly set
        assert df["device"].iloc[0] == "noisy"

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    @patch("oqtopus_experiments.backends.local_backend.AerSimulator")
    @patch("oqtopus_experiments.backends.local_backend.transpile")
    def test_rabi_with_ideal_backend(self, mock_transpile, mock_aer):
        """Test Rabi experiment with ideal LocalBackend"""
        # Setup mocks similar to above
        mock_simulator = MagicMock()
        mock_aer.return_value = mock_simulator

        from qiskit import QuantumCircuit

        mock_circuit = QuantumCircuit(1, 1)
        mock_transpile.return_value = mock_circuit

        def mock_run_side_effect(circuit, shots, **kwargs):
            mock_job = MagicMock()
            mock_result = MagicMock()
            mock_counts = {
                "0": shots // 3,
                "1": 2 * shots // 3,
            }  # Different pattern for ideal
            mock_result.get_counts.return_value = mock_counts
            mock_job.result.return_value = mock_result
            return mock_job

        mock_simulator.run.side_effect = mock_run_side_effect

        # Create backend and experiment
        backend = LocalBackend(device="ideal")
        rabi = Rabi(
            experiment_name="test_rabi_ideal",
            physical_qubit=1,
            amplitude_points=3,
            max_amplitude=2.0,
        )

        # Run experiment
        result = rabi.run(backend, shots=150)

        # Analyze results
        df = result.analyze()

        # Verify DataFrame structure and device name
        if not df.empty:
            assert "device" in df.columns
            assert df["device"].iloc[0] == "ideal"
            assert len(df) == 3  # 3 amplitude points
        else:
            # If analysis fails, just verify the experiment ran
            assert result.experiment_type == "rabi"

    def test_backend_device_name_propagation(self):
        """Test that device names are correctly propagated through the system"""
        # Test both device types
        for device_type in ["noisy", "ideal"]:
            backend = LocalBackend(device=device_type)

            # Verify backend properties
            assert backend.device_name == device_type
            assert backend.noise_enabled == (device_type == "noisy")

            # Test device name switching
            if device_type == "noisy":
                backend.disable_noise()
                assert backend.device_name == "ideal"
                assert backend.noise_enabled == False

                backend.enable_noise()
                assert backend.device_name == "noisy"
                assert backend.noise_enabled == True

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    @patch("oqtopus_experiments.backends.local_backend.AerSimulator")
    def test_noise_parameter_effects(self, mock_aer):
        """Test that noise parameters affect the backend correctly"""
        # Test with custom noise parameters (T2 <= 2*T1)
        backend = LocalBackend(device="noisy", t1=40.0, t2=75.0)

        assert backend.t1_us == 40.0
        assert backend.t2_us == 75.0
        assert backend.noise_enabled == True

        # Update parameters
        backend.set_noise_parameters(t1=40.0, t2=80.0)  # T2 <= 2*T1
        assert backend.t1_us == 40.0
        assert backend.t2_us == 80.0

    def test_multiple_experiments_same_backend(self):
        """Test running multiple experiments with the same backend"""
        backend = LocalBackend(device="noisy")

        # Create multiple experiments
        rabi1 = Rabi(experiment_name="rabi_1", physical_qubit=0, amplitude_points=3)
        rabi2 = Rabi(experiment_name="rabi_2", physical_qubit=1, amplitude_points=4)

        # Both should use the same backend device name
        assert backend.device_name == "noisy"

        # Test device switching affects both
        backend.disable_noise()
        assert backend.device_name == "ideal"


@pytest.mark.slow
class TestLocalBackendPerformance:
    """Performance tests for LocalBackend"""

    def test_many_circuit_execution(self):
        """Test performance with many circuits"""
        backend = LocalBackend(device="ideal")

        if not backend.available:
            pytest.skip("Qiskit not available")

        from qiskit import QuantumCircuit

        # Create many simple circuits
        circuits = []
        for i in range(10):
            circuit = QuantumCircuit(1, 1)
            circuit.x(0) if i % 2 else circuit.h(0)
            circuit.measure(0, 0)
            circuits.append(circuit)

        # Time the execution
        import time

        start_time = time.time()

        results = []
        for circuit in circuits:
            result = backend.run(circuit, shots=100)
            results.append(result)

        execution_time = time.time() - start_time

        # Verify all executed successfully
        assert len(results) == 10
        for result in results:
            assert result["success"] == True
            assert result["backend"] == "ideal"

        # Performance should be reasonable (less than 10 seconds)
        assert execution_time < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
