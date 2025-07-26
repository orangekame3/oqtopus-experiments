#!/usr/bin/env python3
"""
Tests for OqtopusBackend
"""


import pytest
from qiskit import QuantumCircuit

from oqtopus_experiments.backends.oqtopus_backend import OqtopusBackend


class TestOqtopusBackend:
    """Test OqtopusBackend functionality"""

    def test_init_success(self):
        """Test successful initialization - OQTOPUS is available in test environment"""
        backend = OqtopusBackend("test_device", 180)

        assert backend.backend_type == "oqtopus"
        assert backend.device_name == "test_device"
        assert backend.timeout_seconds == 180
        assert backend.available is True  # In test environment, OQTOPUS is available
        assert backend._device_info_loaded is False

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        backend = OqtopusBackend()

        assert backend.device_name == "anemone"
        assert backend.timeout_seconds == 120
        assert backend.available is True  # In test environment, OQTOPUS is available

    def test_run_single_circuit(self):
        """Test running a single circuit - expects fallback to simulated results"""
        backend = OqtopusBackend("test_device")

        # Create test circuit
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.measure(0, 0)

        result = backend.run(circuit, shots=1000)

        # In test environment, OQTOPUS submission will fail and fallback to simulation
        assert "counts" in result
        assert result["counts"]["0"] + result["counts"]["1"] == 1000

    def test_run_backend_unavailable(self):
        """Test running when backend is manually set as unavailable"""
        backend = OqtopusBackend("test_device")
        backend.available = False  # Manually override for testing

        circuit = QuantumCircuit(1, 1)
        result = backend.run(circuit, shots=1000)

        assert result["counts"] == {"0": 500, "1": 500}

    def test_run_with_error_fallback(self):
        """Test running with error - should fallback to simulation"""
        backend = OqtopusBackend("test_device")

        circuit = QuantumCircuit(1, 1)
        # The actual OQTOPUS submission will fail in test environment
        result = backend.run(circuit, shots=1000)

        # Should fallback to simulated results
        assert result["counts"] == {"0": 500, "1": 500}

    def test_run_with_circuit_params(self):
        """Test running with circuit parameters"""
        backend = OqtopusBackend("test_device")

        circuit = QuantumCircuit(1, 1)
        circuit_params = {"amplitude": 1.5}

        result = backend.run(circuit, shots=1000, circuit_params=circuit_params)

        # Should get simulated results
        assert "counts" in result

    def test_submit_method(self):
        """Test submit method returns a job ID"""
        backend = OqtopusBackend("test_device")

        circuit = QuantumCircuit(1, 1)
        job_id = backend.submit(circuit, shots=2000)

        # Should return some job ID string
        assert isinstance(job_id, str)

    def test_submit_backend_unavailable(self):
        """Test submit when backend is unavailable"""
        backend = OqtopusBackend("test_device")
        backend.available = False

        circuit = QuantumCircuit(1, 1)
        job_id = backend.submit(circuit, shots=1000)

        # Submit method generates job ID regardless of availability
        assert isinstance(job_id, str)
        assert job_id.startswith("job_")

    def test_get_result_method(self):
        """Test get_result method"""
        backend = OqtopusBackend("test_device")

        result = backend.get_result("job_789")

        # Should return some result
        assert isinstance(result, dict)

    def test_get_result_backend_unavailable(self):
        """Test get_result when backend is unavailable"""
        backend = OqtopusBackend("test_device")
        backend.available = False

        result = backend.get_result("job_123")

        # When backend is unavailable, get_result returns empty result
        assert isinstance(result, dict)

    def test_backend_attributes(self):
        """Test that backend has required attributes"""
        backend = OqtopusBackend("test_device")

        assert hasattr(backend, 'backend_type')
        assert hasattr(backend, 'device_name')
        assert hasattr(backend, 'timeout_seconds')
        assert hasattr(backend, 'available')

        assert backend.backend_type == "oqtopus"
        assert backend.device_name == "test_device"
        assert isinstance(backend.timeout_seconds, int)
        assert isinstance(backend.available, bool)


if __name__ == "__main__":
    pytest.main([__file__])
