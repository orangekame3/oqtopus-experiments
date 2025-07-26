#!/usr/bin/env python3
"""
Tests for Parity Oscillation models
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from oqtopus_experiments.models.parity_oscillation_models import (
    ParityOscillationAnalysisResult,
    ParityOscillationCircuitParams,
    ParityOscillationExperimentResult,
    ParityOscillationFitting,
    ParityOscillationParameters,
    ParityOscillationPoint,
)


class TestParityOscillationParameters:
    """Test ParityOscillationParameters model"""

    def test_init_default_values(self):
        """Test initialization with default values"""
        params = ParityOscillationParameters()

        assert params.experiment_name is None
        assert params.num_qubits_list == [1, 2, 3, 4, 5]
        assert params.delays_us == [0, 1, 2, 4, 8, 16]
        assert params.phase_points is None
        assert params.no_delay is False
        assert params.shots_per_circuit == 1000

    def test_init_custom_values(self):
        """Test initialization with custom values"""
        params = ParityOscillationParameters(
            experiment_name="test_parity",
            num_qubits_list=[2, 4, 6],
            delays_us=[0, 2, 4],
            phase_points=10,
            no_delay=True,
            shots_per_circuit=500,
        )

        assert params.experiment_name == "test_parity"
        assert params.num_qubits_list == [2, 4, 6]
        assert params.delays_us == [0, 2, 4]
        assert params.phase_points == 10
        assert params.no_delay is True
        assert params.shots_per_circuit == 500

    def test_validation_empty_lists(self):
        """Test validation with empty lists"""
        # Empty lists should be allowed
        params = ParityOscillationParameters(num_qubits_list=[], delays_us=[])
        assert params.num_qubits_list == []
        assert params.delays_us == []

    def test_validation_negative_values(self):
        """Test validation with negative values"""
        # Test that negative values are accepted (no validation constraints in model)
        # This test verifies current behavior - models don't have ge constraints
        params = ParityOscillationParameters(shots_per_circuit=-1)
        assert params.shots_per_circuit == -1

        params = ParityOscillationParameters(phase_points=-5)
        assert params.phase_points == -5


class TestParityOscillationPoint:
    """Test ParityOscillationPoint model"""

    def test_init_valid_data(self):
        """Test initialization with valid data"""
        point = ParityOscillationPoint(
            num_qubits=3,
            delay_us=2.5,
            phase_radians=1.57,
            phase_degrees=90.0,
            parity=0.8,
            parity_error=0.05,
            total_shots=1000,
            counts={"000": 400, "111": 600},
        )

        assert point.num_qubits == 3
        assert point.delay_us == 2.5
        assert point.phase_radians == 1.57
        assert point.phase_degrees == 90.0
        assert point.parity == 0.8
        assert point.parity_error == 0.05
        assert point.total_shots == 1000
        assert point.counts == {"000": 400, "111": 600}

    def test_required_fields(self):
        """Test that all fields are required"""
        with pytest.raises(ValidationError) as exc_info:
            ParityOscillationPoint()

        error_str = str(exc_info.value)
        required_fields = [
            "num_qubits",
            "delay_us",
            "phase_radians",
            "phase_degrees",
            "parity",
            "parity_error",
            "total_shots",
            "counts",
        ]
        for field in required_fields:
            assert field in error_str

    def test_validation_negative_values(self):
        """Test validation with invalid values"""
        base_data = {
            "num_qubits": 2,
            "delay_us": 1.0,
            "phase_radians": 0.0,
            "phase_degrees": 0.0,
            "parity": 0.5,
            "parity_error": 0.01,
            "total_shots": 1000,
            "counts": {"00": 500, "11": 500},
        }

        # Test that negative values are accepted (no validation constraints)
        point = ParityOscillationPoint(**{**base_data, "num_qubits": -1})
        assert point.num_qubits == -1

        point = ParityOscillationPoint(**{**base_data, "total_shots": -1})
        assert point.total_shots == -1


class TestParityOscillationFitting:
    """Test ParityOscillationFitting model"""

    def test_init_valid_data(self):
        """Test initialization with valid data"""
        fitting = ParityOscillationFitting(
            amplitude=0.8,
            phase_offset=0.1,
            vertical_offset=0.05,
            frequency=1.0,
            r_squared=0.95,
            coherence=0.7,
        )

        assert fitting.amplitude == 0.8
        assert fitting.phase_offset == 0.1
        assert fitting.vertical_offset == 0.05
        assert fitting.frequency == 1.0
        assert fitting.r_squared == 0.95
        assert fitting.coherence == 0.7

    def test_required_fields(self):
        """Test that all fields are required"""
        with pytest.raises(ValidationError) as exc_info:
            ParityOscillationFitting()

        error_str = str(exc_info.value)
        required_fields = [
            "amplitude",
            "phase_offset",
            "vertical_offset",
            "frequency",
            "r_squared",
            "coherence",
        ]
        for field in required_fields:
            assert field in error_str


class TestParityOscillationAnalysisResult:
    """Test ParityOscillationAnalysisResult model"""

    def test_init_valid_data(self):
        """Test initialization with valid data"""
        points = [
            ParityOscillationPoint(
                num_qubits=2,
                delay_us=0.0,
                phase_radians=0.0,
                phase_degrees=0.0,
                parity=1.0,
                parity_error=0.01,
                total_shots=1000,
                counts={"00": 500, "11": 500},
            ),
            ParityOscillationPoint(
                num_qubits=3,
                delay_us=1.0,
                phase_radians=1.57,
                phase_degrees=90.0,
                parity=0.8,
                parity_error=0.02,
                total_shots=1000,
                counts={"000": 400, "111": 600},
            ),
        ]

        fitting_results = {
            "N2_τ0us": ParityOscillationFitting(
                amplitude=0.9,
                phase_offset=0.0,
                vertical_offset=0.0,
                frequency=1.0,
                r_squared=0.95,
                coherence=0.9,
            )
        }

        result = ParityOscillationAnalysisResult(
            measurement_points=points,
            fitting_results=fitting_results,
            coherence_matrix={"N2_τ0us": 0.9, "N3_τ1us": 0.8},
            max_coherence=0.9,
            decoherence_rates={2: 0.1, 3: 0.2},
        )

        assert len(result.measurement_points) == 2
        assert len(result.fitting_results) == 1
        assert result.coherence_matrix == {"N2_τ0us": 0.9, "N3_τ1us": 0.8}
        assert result.max_coherence == 0.9
        assert result.decoherence_rates == {2: 0.1, 3: 0.2}

    def test_get_num_qubits_list(self):
        """Test get_num_qubits_list method"""
        points = [
            ParityOscillationPoint(
                num_qubits=3,
                delay_us=0.0,
                phase_radians=0.0,
                phase_degrees=0.0,
                parity=1.0,
                parity_error=0.01,
                total_shots=1000,
                counts={"000": 500, "111": 500},
            ),
            ParityOscillationPoint(
                num_qubits=2,
                delay_us=1.0,
                phase_radians=1.57,
                phase_degrees=90.0,
                parity=0.8,
                parity_error=0.02,
                total_shots=1000,
                counts={"00": 400, "11": 600},
            ),
            ParityOscillationPoint(
                num_qubits=3,  # Duplicate
                delay_us=2.0,
                phase_radians=3.14,
                phase_degrees=180.0,
                parity=0.6,
                parity_error=0.03,
                total_shots=1000,
                counts={"000": 300, "111": 700},
            ),
        ]

        result = ParityOscillationAnalysisResult(
            measurement_points=points,
            fitting_results={},
            coherence_matrix={},
            max_coherence=0.9,
            decoherence_rates={},
        )

        qubits_list = result.get_num_qubits_list()
        assert qubits_list == [2, 3]  # Sorted and unique

    def test_get_delays_list(self):
        """Test get_delays_list method"""
        points = [
            ParityOscillationPoint(
                num_qubits=2,
                delay_us=2.0,
                phase_radians=0.0,
                phase_degrees=0.0,
                parity=1.0,
                parity_error=0.01,
                total_shots=1000,
                counts={"00": 500, "11": 500},
            ),
            ParityOscillationPoint(
                num_qubits=2,
                delay_us=0.0,
                phase_radians=1.57,
                phase_degrees=90.0,
                parity=0.8,
                parity_error=0.02,
                total_shots=1000,
                counts={"00": 400, "11": 600},
            ),
            ParityOscillationPoint(
                num_qubits=2,
                delay_us=2.0,  # Duplicate
                phase_radians=3.14,
                phase_degrees=180.0,
                parity=0.6,
                parity_error=0.03,
                total_shots=1000,
                counts={"00": 300, "11": 700},
            ),
        ]

        result = ParityOscillationAnalysisResult(
            measurement_points=points,
            fitting_results={},
            coherence_matrix={},
            max_coherence=0.9,
            decoherence_rates={},
        )

        delays_list = result.get_delays_list()
        assert delays_list == [0.0, 2.0]  # Sorted and unique

    def test_get_coherence_for_qubits(self):
        """Test get_coherence_for_qubits method"""
        result = ParityOscillationAnalysisResult(
            measurement_points=[],
            fitting_results={},
            coherence_matrix={
                "N2_τ0us": 0.9,
                "N2_τ1us": 0.8,
                "N3_τ0us": 0.7,
                "N3_τ2us": 0.6,
            },
            max_coherence=0.9,
            decoherence_rates={},
        )

        # Test for 2 qubits
        coherence_2q = result.get_coherence_for_qubits(2)
        assert coherence_2q == {0.0: 0.9, 1.0: 0.8}

        # Test for 3 qubits
        coherence_3q = result.get_coherence_for_qubits(3)
        assert coherence_3q == {0.0: 0.7, 2.0: 0.6}

        # Test for non-existent qubit count
        coherence_4q = result.get_coherence_for_qubits(4)
        assert coherence_4q == {}


class TestParityOscillationCircuitParams:
    """Test ParityOscillationCircuitParams model"""

    def test_init_valid_data(self):
        """Test initialization with valid data"""
        params = ParityOscillationCircuitParams(
            num_qubits=3, delay_us=2.5, phase_radians=1.57
        )

        assert params.num_qubits == 3
        assert params.delay_us == 2.5
        assert params.phase_radians == 1.57
        assert params.no_delay is False  # Default value

    def test_init_with_no_delay(self):
        """Test initialization with no_delay option"""
        params = ParityOscillationCircuitParams(
            num_qubits=2, delay_us=0.0, phase_radians=0.0, no_delay=True
        )

        assert params.no_delay is True

    def test_required_fields(self):
        """Test that required fields are validated"""
        with pytest.raises(ValidationError) as exc_info:
            ParityOscillationCircuitParams()

        error_str = str(exc_info.value)
        required_fields = ["num_qubits", "delay_us", "phase_radians"]
        for field in required_fields:
            assert field in error_str


class TestParityOscillationExperimentResult:
    """Test ParityOscillationExperimentResult model"""

    def test_init_valid_data(self):
        """Test initialization with valid data"""
        analysis_result = ParityOscillationAnalysisResult(
            measurement_points=[],
            fitting_results={},
            coherence_matrix={},
            max_coherence=0.9,
            decoherence_rates={},
        )

        df = pd.DataFrame({"num_qubits": [2, 3], "coherence": [0.9, 0.8]})

        result = ParityOscillationExperimentResult(
            analysis_result=analysis_result,
            dataframe=df,
            metadata={"shots": 1000, "experiment_type": "parity_oscillation"},
        )

        assert result.analysis_result == analysis_result
        pd.testing.assert_frame_equal(result.dataframe, df)
        assert result.metadata == {
            "shots": 1000,
            "experiment_type": "parity_oscillation",
        }

    def test_init_default_metadata(self):
        """Test initialization with default metadata"""
        analysis_result = ParityOscillationAnalysisResult(
            measurement_points=[],
            fitting_results={},
            coherence_matrix={},
            max_coherence=0.9,
            decoherence_rates={},
        )

        df = pd.DataFrame()

        result = ParityOscillationExperimentResult(
            analysis_result=analysis_result, dataframe=df
        )

        assert result.metadata == {}

    def test_arbitrary_types_allowed(self):
        """Test that arbitrary types are allowed"""
        import numpy as np

        analysis_result = ParityOscillationAnalysisResult(
            measurement_points=[],
            fitting_results={},
            coherence_matrix={},
            max_coherence=0.9,
            decoherence_rates={},
        )

        # Custom numpy array
        custom_array = np.array([1, 2, 3])

        result = ParityOscillationExperimentResult(
            analysis_result=analysis_result,
            dataframe=custom_array,  # Using numpy array instead of DataFrame
            metadata={"custom_data": custom_array},
        )

        assert isinstance(result.dataframe, np.ndarray)
        assert isinstance(result.metadata["custom_data"], np.ndarray)
        np.testing.assert_array_equal(result.dataframe, custom_array)

    def test_required_fields(self):
        """Test that required fields are validated"""
        with pytest.raises(ValidationError) as exc_info:
            ParityOscillationExperimentResult()

        error_str = str(exc_info.value)
        required_fields = ["analysis_result", "dataframe"]
        for field in required_fields:
            assert field in error_str


if __name__ == "__main__":
    pytest.main([__file__])
