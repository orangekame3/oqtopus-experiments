#!/usr/bin/env python3
"""
Test cases for Randomized Benchmarking experiment
"""

import pandas as pd
from qiskit import QuantumCircuit

from oqtopus_experiments.experiments import RandomizedBenchmarking
from oqtopus_experiments.models.randomized_benchmarking_models import (
    RandomizedBenchmarkingParameters,
)


class TestRandomizedBenchmarking:
    """Test cases for RandomizedBenchmarking experiment"""

    def test_rb_initialization(self):
        """Test RB experiment initialization"""
        rb = RandomizedBenchmarking(
            experiment_name="test_rb",
            physical_qubit=0,
            max_sequence_length=50,
            num_lengths=5,
            num_samples=10,
        )

        assert rb.experiment_name == "test_rb"
        assert rb.physical_qubit == 0
        assert rb.max_sequence_length == 50
        assert rb.num_lengths == 5
        assert rb.num_samples == 10
        assert rb.rb_type == "standard"
        assert rb.interleaved_gate is None

    def test_rb_initialization_with_defaults(self):
        """Test RB experiment initialization with default parameters"""
        rb = RandomizedBenchmarking()

        assert rb.physical_qubit == 0
        assert rb.max_sequence_length == 100
        assert rb.num_lengths == 10
        assert rb.num_samples == 50
        assert rb.rb_type == "standard"

    def test_sequence_lengths_generation(self):
        """Test sequence length generation"""
        rb = RandomizedBenchmarking(
            max_sequence_length=100,
            num_lengths=5,
        )

        lengths = rb.sequence_lengths
        assert len(lengths) <= 5
        assert all(1 <= length <= 100 for length in lengths)
        assert lengths == sorted(lengths)  # Should be sorted

    def test_circuits_generation(self):
        """Test full circuits generation"""
        rb = RandomizedBenchmarking(
            max_sequence_length=20,
            num_lengths=3,
            num_samples=2,
        )

        circuits = rb.circuits()

        # Should have num_lengths * num_samples circuits
        expected_num_circuits = len(rb.sequence_lengths) * rb.num_samples
        assert len(circuits) == expected_num_circuits

        # All should be QuantumCircuit instances
        assert all(isinstance(qc, QuantumCircuit) for qc in circuits)

        # All should have names
        assert all(qc.name for qc in circuits)

    def test_analyze_with_mock_data(self):
        """Test analysis with mock measurement data"""
        rb = RandomizedBenchmarking(
            max_sequence_length=10,
            num_lengths=3,
            num_samples=2,
        )

        # Create mock measurement results
        mock_counts = []
        for _length in rb.sequence_lengths:
            for _sample in range(rb.num_samples):
                # Perfect qubit: all shots result in '0'
                mock_counts.append({"0": 1000, "1": 0})

        # Mock result data in expected format
        mock_results = {"device": [{"counts": counts} for counts in mock_counts]}

        # Run analysis
        df = rb.analyze(mock_results)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert "sequence_length" in df.columns
        assert len(df) == len(rb.sequence_lengths)

    def test_parameters_model(self):
        """Test RB parameters model validation"""
        params = RandomizedBenchmarkingParameters(
            experiment_name="test",
            physical_qubit=1,
            max_sequence_length=50,
            num_lengths=8,
            num_samples=30,
        )

        assert params.experiment_name == "test"
        assert params.physical_qubit == 1
        assert params.max_sequence_length == 50
        assert params.num_lengths == 8
        assert params.num_samples == 30
