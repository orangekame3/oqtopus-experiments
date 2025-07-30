#!/usr/bin/env python3
"""
Test cases for Grover's quantum search algorithm experiment
"""

from typing import Any

import pandas as pd
from qiskit import QuantumCircuit

from oqtopus_experiments.experiments import Grover
from oqtopus_experiments.models.grover_models import GroverParameters


class TestGrover:
    """Test cases for Grover experiment"""

    def test_grover_initialization(self):
        """Test Grover experiment initialization"""
        grover = Grover(
            experiment_name="test_grover",
            n_qubits=3,
            marked_states=[2, 5],
            num_iterations=2,
        )

        assert grover.experiment_name == "test_grover"
        assert grover.n_qubits == 3
        assert grover.marked_states == [2, 5]
        assert grover.num_iterations == 2
        assert grover.search_space_size == 8
        assert grover.optimal_iterations == 1  # π/4 * sqrt(8/2) ≈ 1.11 → 1

    def test_grover_initialization_with_defaults(self):
        """Test Grover experiment initialization with default parameters"""
        grover = Grover()

        assert grover.n_qubits == 2
        assert grover.search_space_size == 4
        assert len(grover.marked_states) >= 1
        assert all(0 <= state < 4 for state in grover.marked_states)

    def test_grover_random_marked_states(self):
        """Test Grover with random marked states"""
        grover = Grover(n_qubits=3, marked_states="random")

        assert isinstance(grover.marked_states, list)
        assert len(grover.marked_states) >= 1
        assert all(0 <= state < 8 for state in grover.marked_states)
        assert len(set(grover.marked_states)) == len(
            grover.marked_states
        )  # No duplicates

    def test_grover_invalid_marked_states(self):
        """Test Grover with invalid marked states"""
        try:
            Grover(n_qubits=2, marked_states=[5])  # 5 is out of range for 2 qubits
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Marked states must be in range" in str(e)

    def test_optimal_iterations_calculation(self):
        """Test optimal iterations calculation"""
        # 2 qubits, 1 marked state: π/4 * sqrt(4/1) = π/2 ≈ 1.57 → 1
        grover = Grover(n_qubits=2, marked_states=[1])
        assert grover.optimal_iterations == 1

        # 3 qubits, 1 marked state: π/4 * sqrt(8/1) ≈ 2.22 → 2
        grover = Grover(n_qubits=3, marked_states=[1])
        assert grover.optimal_iterations == 2

        # No marked states
        grover = Grover(n_qubits=2, marked_states=[])
        assert grover.optimal_iterations == 0

    def test_circuit_generation(self):
        """Test Grover circuit generation"""
        grover = Grover(
            n_qubits=2,
            marked_states=[1],
            num_iterations=1,
        )

        circuits = grover.circuits()

        # Should return exactly one circuit
        assert len(circuits) == 1

        # Should be a QuantumCircuit instance
        assert isinstance(circuits[0], QuantumCircuit)

        # Should have the correct number of qubits and classical bits
        qc = circuits[0]
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

        # Should have a name
        assert qc.name
        assert "grover" in qc.name

    def test_circuit_structure(self):
        """Test basic structure of generated circuit"""
        grover = Grover(
            n_qubits=2,
            marked_states=[1],
            num_iterations=1,
        )

        circuits = grover.circuits()
        qc = circuits[0]

        # Circuit should have some gates (initialization + Grover operators + measurement)
        assert len(qc.data) > 0

        # Should contain Hadamard gates for initialization
        gate_names = [instr.operation.name for instr in qc.data]
        assert "h" in gate_names

        # Should contain measurement
        assert "measure" in gate_names

    def test_analyze_with_mock_data(self):
        """Test analysis with mock measurement data"""
        grover = Grover(
            n_qubits=2,
            marked_states=[1],  # State |01⟩
            num_iterations=1,
        )

        # Create mock measurement results favoring the marked state
        mock_counts = {"00": 50, "01": 800, "10": 100, "11": 50}

        # Mock result data in expected format
        mock_results = {"device": [{"counts": mock_counts}]}

        # Run analysis
        df = grover.analyze(mock_results, plot=False, save_data=False)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "state" in df.columns
        assert "state_binary" in df.columns
        assert "probability" in df.columns
        assert "is_marked" in df.columns
        assert len(df) == 4  # 2^2 states

        # Check marked state identification
        marked_row = df[df["state"] == 1].iloc[0]
        assert marked_row["is_marked"]
        assert marked_row["state_binary"] == "01"

        # Check probabilities sum to 1
        assert abs(df["probability"].sum() - 1.0) < 1e-10

    def test_analyze_empty_results(self):
        """Test analysis with empty results"""
        grover = Grover(n_qubits=2, marked_states=[1])

        # Empty results
        mock_results: dict[str, list[dict[str, Any]]] = {}

        # Run analysis
        df = grover.analyze(mock_results, plot=False, save_data=False)

        # Should return error DataFrame
        assert isinstance(df, pd.DataFrame)
        assert "error" in df.columns

    def test_multiple_marked_states(self):
        """Test Grover with multiple marked states"""
        grover = Grover(
            n_qubits=3,
            marked_states=[1, 3, 6],
            num_iterations=1,
        )

        assert len(grover.marked_states) == 3
        assert grover.marked_states == [1, 3, 6]

        # Generate circuit
        circuits = grover.circuits()
        assert len(circuits) == 1

        # Test analysis with mock data
        mock_counts = {f"{i:03b}": 10 for i in range(8)}
        # Boost marked states
        mock_counts["001"] = 200  # State 1
        mock_counts["011"] = 200  # State 3
        mock_counts["110"] = 200  # State 6

        mock_results = {"device": [{"counts": mock_counts}]}
        df = grover.analyze(mock_results, plot=False, save_data=False)

        # Check marked states are correctly identified
        marked_states = df[df["is_marked"]]["state"].tolist()
        assert set(marked_states) == {1, 3, 6}

    def test_parameters_model(self):
        """Test Grover parameters model validation"""
        params = GroverParameters(
            experiment_name="test",
            n_qubits=4,
            marked_states=[2, 7, 12],
            num_iterations=3,
        )

        assert params.experiment_name == "test"
        assert params.n_qubits == 4
        assert params.marked_states == [2, 7, 12]
        assert params.num_iterations == 3

    def test_zero_iterations(self):
        """Test Grover with zero iterations (just initialization)"""
        grover = Grover(
            n_qubits=2,
            marked_states=[1],
            num_iterations=0,
        )

        circuits = grover.circuits()
        qc = circuits[0]

        # Should still be a valid circuit
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

        # With zero iterations, should see uniform distribution in ideal case
        mock_counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        mock_results = {"device": [{"counts": mock_counts}]}

        df = grover.analyze(mock_results, plot=False, save_data=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4

    def test_single_qubit_grover(self):
        """Test Grover on single qubit"""
        grover = Grover(
            n_qubits=1,
            marked_states=[1],
            num_iterations=1,
        )

        assert grover.search_space_size == 2
        assert grover.optimal_iterations == 1  # π/4 * sqrt(2/1) ≈ 1.11 → 1

        circuits = grover.circuits()
        qc = circuits[0]
        assert qc.num_qubits == 1
        assert qc.num_clbits == 1
