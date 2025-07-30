#!/usr/bin/env python3
"""
Test cases for Deutsch-Jozsa algorithm experiment
"""

import pandas as pd
import pytest
from qiskit import QuantumCircuit

from oqtopus_experiments.experiments.deutsch_jozsa import DeutschJozsa
from oqtopus_experiments.models.deutsch_jozsa_models import (
    DeutschJozsaResult,
)


class TestDeutschJozsa:
    """Test cases for DeutschJozsa experiment class"""

    def test_initialization_default(self):
        """Test default initialization"""
        exp = DeutschJozsa()
        assert exp.n_qubits == 3
        assert exp.oracle_type == "balanced_random"
        assert exp.experiment_name.startswith("deutsch_jozsa_experiment")

    def test_initialization_custom(self):
        """Test custom initialization"""
        exp = DeutschJozsa(
            n_qubits=4,
            oracle_type="constant_0",
            experiment_name="test_dj",
        )
        assert exp.n_qubits == 4
        assert exp.oracle_type == "constant_0"
        assert exp.experiment_name == "test_dj"
        assert exp.is_constant is True

    def test_invalid_oracle_type(self):
        """Test invalid oracle type raises error"""
        with pytest.raises(ValueError, match="Oracle type must be one of"):
            DeutschJozsa(oracle_type="invalid_type")

    def test_oracle_generation(self):
        """Test oracle function generation"""
        # Test constant_0
        exp = DeutschJozsa(n_qubits=3, oracle_type="constant_0")
        assert exp.is_constant is True
        assert exp.oracle_function(0) == 0
        assert exp.oracle_function(7) == 0

        # Test constant_1
        exp = DeutschJozsa(n_qubits=3, oracle_type="constant_1")
        assert exp.is_constant is True
        assert exp.oracle_function(0) == 1
        assert exp.oracle_function(7) == 1

        # Test balanced_alternating (XOR)
        exp = DeutschJozsa(n_qubits=3, oracle_type="balanced_alternating")
        assert exp.is_constant is False
        assert exp.oracle_function(0) == 0  # 000 -> even parity
        assert exp.oracle_function(1) == 1  # 001 -> odd parity
        assert exp.oracle_function(3) == 0  # 011 -> even parity
        assert exp.oracle_function(7) == 1  # 111 -> odd parity

        # Test balanced_random
        exp = DeutschJozsa(n_qubits=3, oracle_type="balanced_random")
        assert exp.is_constant is False
        # Check that it's balanced
        values = [exp.oracle_function(i) for i in range(8)]
        assert sum(values) == 4  # Half zeros, half ones

    def test_circuits_generation(self):
        """Test quantum circuit generation"""
        exp = DeutschJozsa(n_qubits=3, oracle_type="constant_0")
        circuits = exp.circuits()

        assert len(circuits) == 1
        qc = circuits[0]
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4  # 3 input + 1 ancilla
        assert qc.num_clbits == 3  # Only measure input qubits

    def test_circuit_structure(self):
        """Test the structure of generated circuits"""
        exp = DeutschJozsa(n_qubits=2, oracle_type="constant_0")
        qc = exp.circuits()[0]

        # Should have initial X gate on ancilla
        x_gates = [inst for inst in qc.data if inst.operation.name == "x"]
        assert len(x_gates) >= 1  # At least one X gate for ancilla initialization

        # Should have Hadamard gates
        # First H on all qubits (n_qubits + 1), then H on input qubits (n_qubits)
        h_gates = [inst for inst in qc.data if inst.operation.name == "h"]
        assert (
            len(h_gates) == 3 + 2
        )  # 3 H gates initially (2 input + 1 ancilla), then 2 H on input qubits

        # Should have measurements on input qubits
        measure_gates = [inst for inst in qc.data if inst.operation.name == "measure"]
        assert len(measure_gates) == 2  # Measure only input qubits

    def test_process_results(self):
        """Test result processing"""
        exp = DeutschJozsa(n_qubits=2, oracle_type="constant_0")

        # Mock results for constant function (should measure all zeros)
        mock_results = [
            {
                "counts": {"00": 900, "01": 50, "10": 30, "11": 20},
                "backend": "test_backend",
            }
        ]

        result = exp._process_results(mock_results)
        assert isinstance(result, DeutschJozsaResult)
        assert result.oracle_type == "constant_0"
        assert result.is_constant_actual is True
        assert result.is_constant_measured is True  # 900/1000 > 0.5
        assert result.all_zeros_probability == 0.9
        assert result.is_correct is True
        assert result.total_shots == 1000

    def test_process_results_balanced(self):
        """Test result processing for balanced function"""
        exp = DeutschJozsa(n_qubits=2, oracle_type="balanced_alternating")

        # Mock results for balanced function (should never measure all zeros)
        mock_results = [
            {
                "counts": {"00": 10, "01": 490, "10": 480, "11": 20},
                "backend": "test_backend",
            }
        ]

        result = exp._process_results(mock_results)
        assert result.is_constant_actual is False
        assert result.is_constant_measured is False  # 10/1000 < 0.5
        assert result.all_zeros_probability == 0.01
        assert result.is_correct is True

    def test_create_dataframe(self):
        """Test DataFrame creation"""
        exp = DeutschJozsa(n_qubits=2, oracle_type="constant_0")

        result = DeutschJozsaResult(
            oracle_type="constant_0",
            is_constant_actual=True,
            is_constant_measured=True,
            all_zeros_probability=0.9,
            is_correct=True,
            counts={"00": 900, "01": 100},
            distribution={"00": 0.9, "01": 0.1},
            total_shots=1000,
        )

        df = exp._create_dataframe(result, "test_device")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "outcome" in df.columns
        assert "probability" in df.columns
        assert "is_all_zeros" in df.columns
        assert df[df["outcome"] == "00"]["is_all_zeros"].iloc[0]
        assert not df[df["outcome"] == "01"]["is_all_zeros"].iloc[0]

    def test_analyze(self):
        """Test analyze method"""
        exp = DeutschJozsa(n_qubits=2, oracle_type="constant_0")

        # Mock results
        mock_results = {
            "device1": [
                {
                    "counts": {"00": 950, "01": 30, "10": 15, "11": 5},
                    "backend": "test_backend",
                }
            ]
        }

        df = exp.analyze(mock_results, plot=False, save_data=False)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "outcome" in df.columns
        assert "probability" in df.columns
        assert "is_correct" in df.columns
        assert df["is_correct"].iloc[0]

    def test_analyze_empty_results(self):
        """Test analyze with empty results"""
        exp = DeutschJozsa()
        df = exp.analyze({}, plot=False, save_data=False)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_different_qubit_sizes(self):
        """Test experiment with different qubit sizes"""
        for n in [1, 2, 3, 4, 5]:
            exp = DeutschJozsa(n_qubits=n)
            circuits = exp.circuits()
            assert circuits[0].num_qubits == n + 1
            assert circuits[0].num_clbits == n

    def test_oracle_application(self):
        """Test oracle application for different types"""
        # Test constant_0 oracle (identity)
        exp = DeutschJozsa(n_qubits=2, oracle_type="constant_0")
        qc = QuantumCircuit(3, 2)
        initial_gates = len(qc.data)
        exp._apply_oracle(qc)
        assert len(qc.data) == initial_gates  # No gates added

        # Test constant_1 oracle (X on ancilla)
        exp = DeutschJozsa(n_qubits=2, oracle_type="constant_1")
        qc = QuantumCircuit(3, 2)
        initial_gates = len(qc.data)
        exp._apply_oracle(qc)
        assert len(qc.data) == initial_gates + 1  # One X gate added

        # Test balanced_alternating oracle (CNOTs)
        exp = DeutschJozsa(n_qubits=2, oracle_type="balanced_alternating")
        qc = QuantumCircuit(3, 2)
        initial_gates = len(qc.data)
        exp._apply_oracle(qc)
        assert len(qc.data) == initial_gates + 2  # Two CNOT gates added
