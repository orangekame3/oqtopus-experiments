#!/usr/bin/env python3
"""
Test cases for Randomized Benchmarking models
"""

from oqtopus_experiments.models.randomized_benchmarking_models import (
    RandomizedBenchmarkingData,
    RandomizedBenchmarkingFittingResult,
    RandomizedBenchmarkingParameters,
    RandomizedBenchmarkingResult,
)


class TestRandomizedBenchmarkingModels:
    """Test cases for RB Pydantic models"""

    def test_rb_parameters_creation(self):
        """Test RandomizedBenchmarkingParameters model"""
        params = RandomizedBenchmarkingParameters(
            experiment_name="test_rb",
            physical_qubit=1,
            max_sequence_length=50,
            num_lengths=8,
            num_samples=20,
            rb_type="standard",
        )

        assert params.experiment_name == "test_rb"
        assert params.physical_qubit == 1
        assert params.max_sequence_length == 50
        assert params.num_lengths == 8
        assert params.num_samples == 20
        assert params.rb_type == "standard"
        assert params.interleaved_gate is None

    def test_rb_fitting_result_creation(self):
        """Test RandomizedBenchmarkingFittingResult model"""
        fitting_result = RandomizedBenchmarkingFittingResult(
            error_per_clifford=0.001,
            decay_rate=0.998,
            initial_fidelity=0.95,
            offset=0.05,
            r_squared=0.99,
            sequence_lengths=[1, 5, 10],
            survival_probabilities=[0.95, 0.90, 0.85],
        )

        assert fitting_result.error_per_clifford == 0.001
        assert fitting_result.decay_rate == 0.998
        assert fitting_result.r_squared == 0.99
        assert fitting_result.error_info is None

    def test_rb_data_creation(self):
        """Test RandomizedBenchmarkingData model"""
        data = RandomizedBenchmarkingData(
            sequence_lengths=[1, 5, 10],
            survival_probabilities=[[0.95, 0.96], [0.90, 0.89], [0.85, 0.84]],
            mean_survival_probabilities=[0.955, 0.895, 0.845],
            std_survival_probabilities=[0.005, 0.005, 0.005],
            num_samples=2,
        )

        assert len(data.sequence_lengths) == 3
        assert len(data.survival_probabilities) == 3
        assert data.num_samples == 2
        assert data.fitting_result is None

    def test_rb_result_creation(self):
        """Test RandomizedBenchmarkingResult model"""
        data = RandomizedBenchmarkingData(
            sequence_lengths=[1, 5],
            survival_probabilities=[[0.95, 0.96], [0.90, 0.89]],
            mean_survival_probabilities=[0.955, 0.895],
            std_survival_probabilities=[0.005, 0.005],
            num_samples=2,
        )

        # Mock experiment instance for testing
        class MockExperiment:
            def analyze(self, *args, **kwargs):
                return data

        mock_experiment = MockExperiment()
        raw_results = {
            "device": [
                {"counts": {"0": 950, "1": 50}},
                {"counts": {"0": 890, "1": 110}},
            ]
        }

        result = RandomizedBenchmarkingResult(
            raw_results=raw_results,
            experiment_instance=mock_experiment,
            data=data,
            backend="test_backend",
            device="test_device",
            shots=1000,
            metadata={"test": "value"},
        )

        # RandomizedBenchmarkingResult inherits from ExperimentResult
        # Check that the data attribute is correctly set
        assert isinstance(result.data, RandomizedBenchmarkingData)
        assert result.data.sequence_lengths == [1, 5]
        assert result.data.num_samples == 2
