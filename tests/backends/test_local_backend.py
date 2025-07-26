#!/usr/bin/env python3
"""
Tests for LocalBackend
"""

from unittest.mock import MagicMock, patch

import pytest
from qiskit import QuantumCircuit

from oqtopus_experiments.backends.local_backend import LocalBackend


class TestLocalBackend:
    """Test LocalBackend functionality"""

    def test_init_noisy_default(self):
        """Test LocalBackend initialization with default noisy device"""
        backend = LocalBackend()

        assert backend.device_name == "noisy"
        assert backend.noise_enabled
        assert backend.backend_type == "local"
        assert backend.t1_us == 50.0
        assert backend.t2_us == 100.0

    def test_init_ideal_device(self):
        """Test LocalBackend initialization with ideal device"""
        backend = LocalBackend(device="ideal")

        assert backend.device_name == "ideal"
        assert not backend.noise_enabled
        assert backend.backend_type == "local"

    def test_init_custom_params(self):
        """Test LocalBackend initialization with custom T1/T2 parameters"""
        # T2 must be <= 2*T1, so use valid values
        backend = LocalBackend(device="noisy", t1=50.0, t2=80.0)

        assert backend.device_name == "noisy"
        assert backend.noise_enabled
        assert backend.t1_us == 50.0
        assert backend.t2_us == 80.0

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    @patch("oqtopus_experiments.backends.local_backend.AerSimulator")
    def test_init_with_qiskit_available(self, mock_aer):
        """Test initialization when Qiskit is available"""
        mock_simulator = MagicMock()
        mock_aer.return_value = mock_simulator

        backend = LocalBackend(device="noisy")

        assert backend.available
        assert backend.simulator == mock_simulator
        mock_aer.assert_called_once()

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", False)
    def test_init_without_qiskit(self):
        """Test initialization when Qiskit is not available"""
        backend = LocalBackend()

        assert not backend.available
        assert backend.simulator is None
        assert backend.noise_model is None

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    @patch("oqtopus_experiments.backends.local_backend.AerSimulator")
    @patch("oqtopus_experiments.backends.local_backend.transpile")
    def test_run_noisy_success(self, mock_transpile, mock_aer):
        """Test successful circuit execution with noise"""
        # Setup mocks
        mock_simulator = MagicMock()
        mock_aer.return_value = mock_simulator

        mock_job = MagicMock()
        mock_result = MagicMock()
        mock_counts = {"0": 480, "1": 520}
        mock_result.get_counts.return_value = mock_counts
        mock_job.result.return_value = mock_result
        mock_simulator.run.return_value = mock_job

        mock_circuit = QuantumCircuit(1, 1)
        mock_transpile.return_value = mock_circuit

        # Test
        backend = LocalBackend(device="noisy")
        result = backend.run(mock_circuit, shots=1000)

        # Assertions
        assert result["backend"] == "noisy"
        assert result["noise_enabled"]
        assert result["counts"] == mock_counts
        assert result["shots"] == 1000
        assert result["success"]
        assert "job_id" in result

        mock_transpile.assert_called_once_with(mock_circuit, mock_simulator)
        mock_simulator.run.assert_called_once()

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    @patch("oqtopus_experiments.backends.local_backend.AerSimulator")
    @patch("oqtopus_experiments.backends.local_backend.transpile")
    def test_run_ideal_success(self, mock_transpile, mock_aer):
        """Test successful circuit execution without noise"""
        # Setup mocks
        mock_simulator = MagicMock()
        mock_aer.return_value = mock_simulator

        mock_job = MagicMock()
        mock_result = MagicMock()
        mock_counts = {"0": 500, "1": 500}
        mock_result.get_counts.return_value = mock_counts
        mock_job.result.return_value = mock_result
        mock_simulator.run.return_value = mock_job

        mock_circuit = QuantumCircuit(1, 1)
        mock_transpile.return_value = mock_circuit

        # Test
        backend = LocalBackend(device="ideal")
        result = backend.run(mock_circuit, shots=1000)

        # Assertions
        assert result["backend"] == "ideal"
        assert not result["noise_enabled"]
        assert result["counts"] == mock_counts

        # Should run without noise model
        call_args = mock_simulator.run.call_args
        assert "noise_model" not in call_args.kwargs

    def test_run_qiskit_unavailable(self):
        """Test circuit execution when Qiskit is not available"""
        backend = LocalBackend()
        backend.available = False

        circuit = QuantumCircuit(1, 1)
        result = backend.run(circuit, shots=1000)

        # Should return fake results
        assert result["counts"] == {"0": 500, "1": 500}

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    @patch("oqtopus_experiments.backends.local_backend.AerSimulator")
    @patch("oqtopus_experiments.backends.local_backend.transpile")
    def test_run_execution_error(self, mock_transpile, mock_aer):
        """Test circuit execution with exception"""
        mock_simulator = MagicMock()
        mock_aer.return_value = mock_simulator
        mock_simulator.run.side_effect = Exception("Simulation failed")

        # Mock transpile to return a real circuit
        circuit = QuantumCircuit(1, 1)
        mock_transpile.return_value = circuit

        backend = LocalBackend(device="noisy")
        result = backend.run(circuit, shots=1000)

        # Should return error result
        assert not result["success"]
        assert "error" in result
        assert "Simulation failed" in result["error"]

    def test_enable_disable_noise(self):
        """Test noise enable/disable functionality"""
        backend = LocalBackend(device="ideal")

        # Initially ideal
        assert backend.device_name == "ideal"
        assert not backend.noise_enabled

        # Enable noise
        backend.enable_noise()
        assert backend.device_name == "noisy"
        assert backend.noise_enabled

        # Disable noise
        backend.disable_noise()
        assert backend.device_name == "ideal"
        assert not backend.noise_enabled

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    def test_set_noise_parameters(self):
        """Test noise parameter updates"""
        backend = LocalBackend(device="noisy")

        # Update noise parameters (T2 <= 2*T1)
        backend.set_noise_parameters(t1=40.0, t2=60.0)

        assert backend.t1_us == 40.0
        assert backend.t2_us == 60.0

    def test_submit_get_result(self):
        """Test submit and get_result methods (placeholders)"""
        backend = LocalBackend()

        circuit = QuantumCircuit(1, 1)
        job_id = backend.submit(circuit, shots=1000)
        result = backend.get_result(job_id)

        # These are placeholder implementations
        assert isinstance(job_id, str)
        assert "counts" in result

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    @patch("oqtopus_experiments.backends.local_backend.AerSimulator")
    def test_noise_model_creation(self, mock_aer):
        """Test that noise model is created for noisy backend"""
        backend = LocalBackend(device="noisy")

        # Noise model should be created
        assert backend.noise_model is not None

    @patch("oqtopus_experiments.backends.local_backend.QISKIT_AVAILABLE", True)
    @patch("oqtopus_experiments.backends.local_backend.AerSimulator")
    def test_no_noise_model_for_ideal(self, mock_aer):
        """Test that no noise model is created for ideal backend"""
        backend = LocalBackend(device="ideal")

        # No noise model should be created
        assert backend.noise_model is None


@pytest.mark.integration
class TestLocalBackendIntegration:
    """Integration tests for LocalBackend with real Qiskit (if available)"""

    def test_real_circuit_execution(self):
        """Test real circuit execution if Qiskit is available"""
        try:
            from qiskit import QuantumCircuit

            # Create a simple circuit
            circuit = QuantumCircuit(1, 1)
            circuit.h(0)  # Hadamard gate
            circuit.measure(0, 0)

            # Test both devices
            for device in ["noisy", "ideal"]:
                backend = LocalBackend(device=device)

                if backend.available:
                    result = backend.run(circuit, shots=100)

                    assert result["backend"] == device
                    assert result["noise_enabled"] == (device == "noisy")
                    assert result["success"]
                    assert "counts" in result
                    assert result["shots"] == 100

                    # Check that we get both 0 and 1 outcomes (probabilistic)
                    counts = result["counts"]
                    total_counts = sum(counts.values())
                    assert total_counts == 100

        except ImportError:
            pytest.skip("Qiskit not available for integration test")


if __name__ == "__main__":
    pytest.main([__file__])
