#!/usr/bin/env python3
"""
Tests for BaseExperiment
"""

from unittest.mock import MagicMock, patch

import pytest
from qiskit import QuantumCircuit

from oqtopus_experiments.core.base_experiment import BaseExperiment
from oqtopus_experiments.models.experiment_result import ExperimentResult


class MockExperiment(BaseExperiment):
    """Mock experiment for testing"""

    def __init__(self, experiment_name=None):
        super().__init__(experiment_name)
        self._test_circuits = None

    def circuits(self, **kwargs):
        """Return test circuits"""
        if self._test_circuits is not None:
            return self._test_circuits

        # Create simple test circuits
        circuit1 = QuantumCircuit(1, 1)
        circuit1.h(0)
        circuit1.measure(0, 0)

        circuit2 = QuantumCircuit(1, 1)
        circuit2.x(0)
        circuit2.measure(0, 0)

        return [circuit1, circuit2]

    def analyze(self, results, plot=False, save_data=False, save_image=False):
        """Mock analysis"""
        return {"test_result": "mock_analysis"}


class TestBaseExperiment:
    """Test BaseExperiment functionality"""

    def test_init_default_name(self):
        """Test initialization with default experiment name"""
        experiment = MockExperiment()

        assert experiment.experiment_name.startswith("mockexperiment_")
        assert experiment.data_manager is not None
        assert experiment.transpiler_options == {}
        assert experiment.mitigation_options == {}
        assert experiment.anemone_basis_gates == []
        assert not experiment.oqtopus_available
        assert experiment.experiment_params == {}

    def test_init_custom_name(self):
        """Test initialization with custom experiment name"""
        custom_name = "test_experiment"
        experiment = MockExperiment(custom_name)

        assert experiment.experiment_name == custom_name
        assert experiment.data_manager is not None

    def test_circuits_method(self):
        """Test circuits method returns correct circuits"""
        experiment = MockExperiment()
        circuits = experiment.circuits()

        assert len(circuits) == 2
        assert all(isinstance(circuit, QuantumCircuit) for circuit in circuits)

    def test_save_experiment_data(self):
        """Test save_experiment_data method"""
        experiment = MockExperiment()

        with patch.object(experiment.data_manager, "save_results") as mock_save:
            mock_save.return_value = "test_path"

            results = {"test": "data"}
            metadata = {"version": "1.0"}

            path = experiment.save_experiment_data(results, metadata, "custom_type")

            mock_save.assert_called_once_with(
                results=results, metadata=metadata, experiment_type="custom_type"
            )
            assert path == "test_path"

    def test_save_job_ids(self):
        """Test save_job_ids method"""
        experiment = MockExperiment()

        with patch.object(experiment.data_manager, "save_data") as mock_save:
            mock_save.return_value = "job_ids_path"

            job_ids = {"backend1": ["job1", "job2"]}
            metadata = {"test": "meta"}

            path = experiment.save_job_ids(job_ids, metadata, "custom_filename")

            mock_save.assert_called_once()
            args, kwargs = mock_save.call_args
            save_data = args[0]
            filename = args[1]

            assert save_data["job_ids"] == job_ids
            assert save_data["metadata"] == metadata
            assert save_data["experiment_type"] == "MockExperiment"
            assert "submitted_at" in save_data
            assert filename == "custom_filename"
            assert path == "job_ids_path"

    def test_save_raw_results(self):
        """Test save_raw_results method"""
        experiment = MockExperiment()

        with patch.object(experiment.data_manager, "save_data") as mock_save:
            mock_save.return_value = "raw_results_path"

            results = {"circuit_0": [{"counts": {"0": 50, "1": 50}}]}
            metadata = {"shots": 100}

            path = experiment.save_raw_results(results, metadata, "custom_raw")

            mock_save.assert_called_once()
            args, kwargs = mock_save.call_args
            save_data = args[0]
            filename = args[1]

            assert save_data["results"] == results
            assert save_data["metadata"] == metadata
            assert save_data["experiment_type"] == "MockExperiment"
            assert save_data["oqtopus_available"] == experiment.oqtopus_available
            assert "saved_at" in save_data
            assert filename == "custom_raw"
            assert path == "raw_results_path"

    def test_save_experiment_summary(self):
        """Test save_experiment_summary method"""
        experiment = MockExperiment()

        with patch.object(experiment.data_manager, "summary") as mock_summary:
            mock_summary.return_value = "summary_path"

            path = experiment.save_experiment_summary()

            mock_summary.assert_called_once()
            assert path == "summary_path"

    def test_run_with_backend(self):
        """Test run method with mock backend"""
        experiment = MockExperiment()
        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.run.side_effect = [
            {"counts": {"0": 40, "1": 60}, "success": True},
            {"counts": {"0": 80, "1": 20}, "success": True},
        ]

        result = experiment.run(mock_backend, shots=100)

        assert isinstance(result, ExperimentResult)
        assert len(result.raw_results) == 2
        assert "circuit_0" in result.raw_results
        assert "circuit_1" in result.raw_results
        assert result.experiment_type == "mock"
        assert mock_backend.run.call_count == 2

    def test_run_without_backend_raises_error(self):
        """Test run method raises error when backend is None"""
        experiment = MockExperiment()

        with pytest.raises(ValueError, match="Backend is required for execution"):
            experiment.run(None)

    def test_run_with_backend_error(self):
        """Test run method handles backend errors"""
        experiment = MockExperiment()
        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.run.side_effect = Exception("Backend error")

        result = experiment.run(mock_backend, shots=100)

        assert isinstance(result, ExperimentResult)
        assert len(result.raw_results) == 2
        assert result.raw_results["circuit_0"] == []
        assert result.raw_results["circuit_1"] == []

    def test_run_with_pre_created_circuits(self):
        """Test run method with pre-created circuits"""
        experiment = MockExperiment()

        # Set pre-created circuits
        custom_circuit = QuantumCircuit(1, 1)
        custom_circuit.h(0)
        custom_circuit.measure(0, 0)
        experiment._circuits = [custom_circuit]

        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.run.return_value = {"counts": {"0": 50, "1": 50}, "success": True}

        result = experiment.run(mock_backend, shots=100)

        assert len(result.raw_results) == 1
        assert "circuit_0" in result.raw_results
        mock_backend.run.assert_called_once()

    def test_auto_transpile_if_needed_no_physical_qubits(self):
        """Test auto-transpile when no physical qubits specified"""
        experiment = MockExperiment()
        circuits = [QuantumCircuit(1, 1)]
        mock_backend = MagicMock()

        result_circuits, was_transpiled = experiment._auto_transpile_if_needed(
            circuits, mock_backend
        )

        assert result_circuits == circuits
        assert not was_transpiled

    def test_auto_transpile_if_needed_with_physical_qubits(self):
        """Test auto-transpile with physical qubits specified"""
        experiment = MockExperiment()
        experiment._physical_qubits_specified = True
        experiment.experiment_params = {
            "physical_qubit_0": 2,
            "physical_qubit_1": 3,
            "logical_qubit_0": 0,
            "logical_qubit_1": 1,
        }

        circuits = [QuantumCircuit(2, 2)]
        mock_backend = MagicMock()
        mock_backend.transpile.return_value = "transpiled_circuits"

        result_circuits, was_transpiled = experiment._auto_transpile_if_needed(
            circuits, mock_backend
        )

        assert result_circuits == "transpiled_circuits"
        assert was_transpiled
        mock_backend.transpile.assert_called_once_with(circuits, physical_qubits=[2, 3])

    def test_auto_transpile_if_needed_transpile_error(self):
        """Test auto-transpile handles transpilation errors"""
        experiment = MockExperiment()
        experiment._physical_qubits_specified = True
        experiment.experiment_params = {"physical_qubit_0": 2, "physical_qubit_1": 3}

        circuits = [QuantumCircuit(2, 2)]
        mock_backend = MagicMock()
        mock_backend.transpile.side_effect = Exception("Transpile error")

        result_circuits, was_transpiled = experiment._auto_transpile_if_needed(
            circuits, mock_backend
        )

        assert result_circuits == circuits
        assert not was_transpiled

    def test_auto_transpile_if_needed_no_transpile_method(self):
        """Test auto-transpile when backend doesn't support transpilation"""
        experiment = MockExperiment()
        experiment._physical_qubits_specified = True
        experiment.experiment_params = {"physical_qubit_0": 2, "physical_qubit_1": 3}

        circuits = [QuantumCircuit(2, 2)]
        mock_backend = MagicMock()
        del mock_backend.transpile  # Remove transpile method

        result_circuits, was_transpiled = experiment._auto_transpile_if_needed(
            circuits, mock_backend
        )

        assert result_circuits == circuits
        assert not was_transpiled

    def test_should_disable_transpilation_two_qubits(self):
        """Test _should_disable_transpilation for two-qubit case"""
        experiment = MockExperiment()
        experiment._physical_qubits_specified = True

        assert experiment._should_disable_transpilation()

    def test_should_disable_transpilation_single_qubit(self):
        """Test _should_disable_transpilation for single-qubit case"""
        experiment = MockExperiment()
        experiment._physical_qubit_specified = True

        assert experiment._should_disable_transpilation()

    def test_should_disable_transpilation_legacy_params(self):
        """Test _should_disable_transpilation with legacy params"""
        experiment = MockExperiment()
        experiment.experiment_params = {"physical_qubit_0": 1, "physical_qubit_1": 2}

        assert experiment._should_disable_transpilation()

    def test_should_disable_transpilation_no_physical_qubits(self):
        """Test _should_disable_transpilation when no physical qubits"""
        experiment = MockExperiment()

        assert not experiment._should_disable_transpilation()

    def test_run_parallel_with_parallel_backend(self):
        """Test run_parallel with backend that supports parallel execution"""
        experiment = MockExperiment()

        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        mock_backend.submit_parallel.return_value = ["job1", "job2"]
        mock_backend.collect_parallel.return_value = [
            {"counts": {"0": 40, "1": 60}, "success": True},
            {"counts": {"0": 80, "1": 20}, "success": True},
        ]

        result = experiment.run_parallel(mock_backend, shots=100, workers=2)

        assert isinstance(result, ExperimentResult)
        assert len(result.raw_results) == 2
        mock_backend.submit_parallel.assert_called_once()
        mock_backend.collect_parallel.assert_called_once_with(["job1", "job2"])

    def test_run_parallel_without_parallel_backend(self):
        """Test run_parallel with backend that doesn't support parallel execution"""
        experiment = MockExperiment()

        mock_backend = MagicMock()
        mock_backend.device_name = "test_device"
        # Remove parallel methods
        del mock_backend.submit_parallel
        del mock_backend.collect_parallel
        mock_backend.run.side_effect = [
            {"counts": {"0": 40, "1": 60}, "success": True},
            {"counts": {"0": 80, "1": 20}, "success": True},
        ]

        result = experiment.run_parallel(mock_backend, shots=100, workers=2)

        assert isinstance(result, ExperimentResult)
        assert len(result.raw_results) == 2
        assert mock_backend.run.call_count == 2

    def test_run_parallel_without_backend_raises_error(self):
        """Test run_parallel raises error when backend is None"""
        experiment = MockExperiment()

        with pytest.raises(
            ValueError, match="Backend is required for parallel execution"
        ):
            experiment.run_parallel(None)


if __name__ == "__main__":
    pytest.main([__file__])
