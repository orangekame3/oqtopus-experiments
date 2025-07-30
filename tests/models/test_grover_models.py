#!/usr/bin/env python3
"""
Test cases for Grover models
"""

import pandas as pd

from oqtopus_experiments.models.grover_models import (
    GroverAnalysisResult,
    GroverData,
    GroverParameters,
    GroverResult,
)


class TestGroverModels:
    """Test cases for Grover Pydantic models"""

    def test_grover_parameters_model(self):
        """Test GroverParameters model"""
        params = GroverParameters(
            experiment_name="test_grover",
            n_qubits=3,
            marked_states=[1, 5],
            num_iterations=2,
        )

        assert params.experiment_name == "test_grover"
        assert params.n_qubits == 3
        assert params.marked_states == [1, 5]
        assert params.num_iterations == 2

    def test_grover_parameters_defaults(self):
        """Test GroverParameters model with defaults"""
        params = GroverParameters()

        assert params.experiment_name is None
        assert params.n_qubits == 2
        assert params.marked_states == "random"
        assert params.num_iterations is None

    def test_grover_analysis_result_model(self):
        """Test GroverAnalysisResult model"""
        analysis = GroverAnalysisResult(
            success_probability=0.85,
            theoretical_success_probability=0.90,
            marked_states=[1, 3],
            unmarked_states=[0, 2],
            measurement_counts={"00": 50, "01": 400, "10": 50, "11": 500},
            total_shots=1000,
            success_rate_error=0.05,
            optimal_iterations=1,
            actual_iterations=1,
        )

        assert analysis.success_probability == 0.85
        assert analysis.theoretical_success_probability == 0.90
        assert analysis.marked_states == [1, 3]
        assert analysis.unmarked_states == [0, 2]
        assert analysis.total_shots == 1000
        assert analysis.success_rate_error == 0.05
        assert analysis.optimal_iterations == 1
        assert analysis.actual_iterations == 1

    def test_grover_data_model(self):
        """Test GroverData model"""
        data = GroverData(
            n_qubits=2,
            marked_states=[1],
            num_iterations=1,
            measurement_counts={"00": 100, "01": 800, "10": 50, "11": 50},
            total_shots=1000,
            search_space_size=4,
        )

        assert data.n_qubits == 2
        assert data.marked_states == [1]
        assert data.num_iterations == 1
        assert data.total_shots == 1000
        assert data.search_space_size == 4
        assert data.analysis_result is None

    def test_grover_result_analyze(self):
        """Test GroverResult analyze method"""
        # Create test data
        data = GroverData(
            n_qubits=2,
            marked_states=[1],  # State |01⟩
            num_iterations=1,
            measurement_counts={"00": 50, "01": 700, "10": 150, "11": 100},
            total_shots=1000,
            search_space_size=4,
        )

        # Create result object
        result = GroverResult(
            raw_results={"device": [{"counts": data.measurement_counts}]},
            experiment_instance=None,
            data=data,
            backend="test_backend",
            device="test_device",
            shots=1000,
            metadata={"experiment_type": "grover"},
        )

        # Run analysis
        df = result.analyze(plot=False, save_data=False)

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2^2 states
        assert "state" in df.columns
        assert "state_binary" in df.columns
        assert "probability" in df.columns
        assert "is_marked" in df.columns
        assert "theoretical_probability" in df.columns

        # Check state ordering
        expected_states = [0, 1, 2, 3]
        assert df["state"].tolist() == expected_states

        # Check binary representations
        expected_binary = ["00", "01", "10", "11"]
        assert df["state_binary"].tolist() == expected_binary

        # Check marked state identification
        marked_mask = df["is_marked"]
        assert marked_mask.tolist() == [False, True, False, False]

        # Check probabilities
        assert abs(df["probability"].sum() - 1.0) < 1e-10

        # Check analysis result was stored
        assert data.analysis_result is not None
        assert isinstance(data.analysis_result, GroverAnalysisResult)
        assert data.analysis_result.marked_states == [1]
        assert data.analysis_result.total_shots == 1000

    def test_grover_result_analyze_multiple_marked(self):
        """Test GroverResult analyze with multiple marked states"""
        # Create test data with multiple marked states
        data = GroverData(
            n_qubits=3,
            marked_states=[1, 3, 6],
            num_iterations=2,
            measurement_counts={
                "000": 50,  # State 0
                "001": 200,  # State 1 (marked)
                "010": 30,  # State 2
                "011": 250,  # State 3 (marked)
                "100": 40,  # State 4
                "101": 20,  # State 5
                "110": 300,  # State 6 (marked)
                "111": 110,  # State 7
            },
            total_shots=1000,
            search_space_size=8,
        )

        result = GroverResult(
            raw_results={"device": [{"counts": data.measurement_counts}]},
            experiment_instance=None,
            data=data,
            backend="test_backend",
            device="test_device",
            shots=1000,
            metadata={"experiment_type": "grover"},
        )

        df = result.analyze(plot=False, save_data=False)

        # Check DataFrame structure
        assert len(df) == 8  # 2^3 states

        # Check marked states
        marked_states = df[df["is_marked"]]["state"].tolist()
        assert set(marked_states) == {1, 3, 6}

        # Check unmarked states
        unmarked_states = df[~df["is_marked"]]["state"].tolist()
        assert set(unmarked_states) == {0, 2, 4, 5, 7}

    def test_grover_result_analyze_no_marked_states(self):
        """Test GroverResult analyze with no marked states"""
        data = GroverData(
            n_qubits=2,
            marked_states=[],
            num_iterations=0,
            measurement_counts={"00": 250, "01": 250, "10": 250, "11": 250},
            total_shots=1000,
            search_space_size=4,
        )

        result = GroverResult(
            raw_results={"device": [{"counts": data.measurement_counts}]},
            experiment_instance=None,
            data=data,
            backend="test_backend",
            device="test_device",
            shots=1000,
            metadata={"experiment_type": "grover"},
        )

        df = result.analyze(plot=False, save_data=False)

        # Should still produce valid DataFrame
        assert len(df) == 4
        assert not df["is_marked"].any()  # No marked states

        # Check analysis result
        assert data.analysis_result is not None
        assert data.analysis_result.optimal_iterations == 0
        assert data.analysis_result.theoretical_success_probability == 0.0

    def test_grover_result_analyze_error_handling(self):
        """Test GroverResult analyze with invalid data"""
        # Create data with invalid measurement counts (empty)
        data = GroverData(
            n_qubits=2,
            marked_states=[1],
            num_iterations=1,
            measurement_counts={},  # Empty counts
            total_shots=0,
            search_space_size=4,
        )

        result = GroverResult(
            raw_results={"device": [{"counts": {}}]},
            experiment_instance=None,
            data=data,
            backend="test_backend",
            device="test_device",
            shots=0,
            metadata={"experiment_type": "grover"},
        )

        df = result.analyze(plot=False, save_data=False)

        # Should still return DataFrame, potentially with error info
        assert isinstance(df, pd.DataFrame)

    def test_grover_theoretical_probability_calculation(self):
        """Test theoretical probability calculations"""
        data = GroverData(
            n_qubits=2,
            marked_states=[1],
            num_iterations=1,
            measurement_counts={"00": 250, "01": 250, "10": 250, "11": 250},
            total_shots=1000,
            search_space_size=4,
        )

        result = GroverResult(
            raw_results={"device": [{"counts": data.measurement_counts}]},
            experiment_instance=None,
            data=data,
            backend="test_backend",
            device="test_device",
            shots=1000,
            metadata={"experiment_type": "grover"},
        )

        df = result.analyze(plot=False, save_data=False)

        # Check that theoretical probabilities are calculated
        assert "theoretical_probability" in df.columns

        # For marked states, theoretical probability should be higher
        marked_row = df[df["is_marked"]].iloc[0]
        unmarked_rows = df[~df["is_marked"]]

        # Theoretical probability for marked state should be higher than unmarked
        assert (
            marked_row["theoretical_probability"]
            > unmarked_rows["theoretical_probability"].iloc[0]
        )
