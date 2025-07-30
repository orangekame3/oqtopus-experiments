#!/usr/bin/env python3
"""
Test cases for Deutsch-Jozsa models
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from oqtopus_experiments.models.deutsch_jozsa_models import (
    DeutschJozsaAnalysisResult,
    DeutschJozsaCircuitParams,
    DeutschJozsaParameters,
    DeutschJozsaResult,
)


class TestDeutschJozsaParameters:
    """Test cases for DeutschJozsaParameters model"""

    def test_valid_parameters(self):
        """Test valid parameter creation"""
        params = DeutschJozsaParameters(
            experiment_name="test_dj",
            n_qubits=3,
            oracle_type="constant_0",
        )
        assert params.experiment_name == "test_dj"
        assert params.n_qubits == 3
        assert params.oracle_type == "constant_0"

    def test_default_parameters(self):
        """Test default parameter values"""
        params = DeutschJozsaParameters()
        assert params.experiment_name is None
        assert params.n_qubits == 3
        assert params.oracle_type == "balanced_random"

    def test_invalid_oracle_type(self):
        """Test invalid oracle type validation"""
        with pytest.raises(ValidationError) as excinfo:
            DeutschJozsaParameters(oracle_type="invalid_oracle")
        assert "Oracle type must be one of" in str(excinfo.value)

    def test_n_qubits_validation(self):
        """Test n_qubits validation"""
        # Valid range
        for n in [1, 5, 10]:
            params = DeutschJozsaParameters(n_qubits=n)
            assert params.n_qubits == n

        # Invalid: too small
        with pytest.raises(ValidationError):
            DeutschJozsaParameters(n_qubits=0)

        # Invalid: too large
        with pytest.raises(ValidationError):
            DeutschJozsaParameters(n_qubits=11)

    def test_all_oracle_types(self):
        """Test all valid oracle types"""
        valid_types = [
            "constant_0",
            "constant_1",
            "balanced_random",
            "balanced_alternating",
        ]
        for oracle_type in valid_types:
            params = DeutschJozsaParameters(oracle_type=oracle_type)
            assert params.oracle_type == oracle_type


class TestDeutschJozsaCircuitParams:
    """Test cases for DeutschJozsaCircuitParams model"""

    def test_valid_circuit_params(self):
        """Test valid circuit parameter creation"""
        params = DeutschJozsaCircuitParams(
            n_qubits=4,
            oracle_type="constant_1",
        )
        assert params.n_qubits == 4
        assert params.oracle_type == "constant_1"


class TestDeutschJozsaResult:
    """Test cases for DeutschJozsaResult model"""

    def test_valid_result(self):
        """Test valid result creation"""
        result = DeutschJozsaResult(
            oracle_type="constant_0",
            is_constant_actual=True,
            is_constant_measured=True,
            all_zeros_probability=0.95,
            is_correct=True,
            counts={"00": 950, "01": 30, "10": 15, "11": 5},
            distribution={"00": 0.95, "01": 0.03, "10": 0.015, "11": 0.005},
            total_shots=1000,
        )
        assert result.oracle_type == "constant_0"
        assert result.is_constant_actual is True
        assert result.is_constant_measured is True
        assert result.all_zeros_probability == 0.95
        assert result.is_correct is True
        assert result.total_shots == 1000

    def test_probability_validation(self):
        """Test probability validation"""
        # Valid: probabilities sum to 1
        result = DeutschJozsaResult(
            oracle_type="balanced_random",
            is_constant_actual=False,
            is_constant_measured=False,
            all_zeros_probability=0.1,
            is_correct=True,
            counts={"00": 100, "01": 300, "10": 400, "11": 200},
            distribution={"00": 0.1, "01": 0.3, "10": 0.4, "11": 0.2},
            total_shots=1000,
        )
        assert sum(result.distribution.values()) == 1.0

        # Invalid: probabilities don't sum to 1
        with pytest.raises(ValidationError) as excinfo:
            DeutschJozsaResult(
                oracle_type="constant_0",
                is_constant_actual=True,
                is_constant_measured=True,
                all_zeros_probability=0.5,
                is_correct=True,
                counts={"00": 500, "01": 300},
                distribution={"00": 0.5, "01": 0.3},  # Sum = 0.8
                total_shots=800,
            )
        assert "Probabilities must sum to 1" in str(excinfo.value)

    def test_all_zeros_probability_bounds(self):
        """Test all_zeros_probability bounds"""
        # Valid: 0 <= p <= 1
        for p in [0.0, 0.5, 1.0]:
            result = DeutschJozsaResult(
                oracle_type="constant_0",
                is_constant_actual=True,
                is_constant_measured=True,
                all_zeros_probability=p,
                is_correct=True,
                counts={"00": 100},
                distribution={"00": 1.0},
                total_shots=100,
            )
            assert result.all_zeros_probability == p

        # Invalid: negative probability
        with pytest.raises(ValidationError):
            DeutschJozsaResult(
                oracle_type="constant_0",
                is_constant_actual=True,
                is_constant_measured=True,
                all_zeros_probability=-0.1,
                is_correct=True,
                counts={"00": 100},
                distribution={"00": 1.0},
                total_shots=100,
            )

        # Invalid: probability > 1
        with pytest.raises(ValidationError):
            DeutschJozsaResult(
                oracle_type="constant_0",
                is_constant_actual=True,
                is_constant_measured=True,
                all_zeros_probability=1.1,
                is_correct=True,
                counts={"00": 100},
                distribution={"00": 1.0},
                total_shots=100,
            )

    def test_total_shots_validation(self):
        """Test total_shots validation"""
        # Invalid: zero shots
        with pytest.raises(ValidationError):
            DeutschJozsaResult(
                oracle_type="constant_0",
                is_constant_actual=True,
                is_constant_measured=True,
                all_zeros_probability=1.0,
                is_correct=True,
                counts={},
                distribution={},
                total_shots=0,
            )


class TestDeutschJozsaAnalysisResult:
    """Test cases for DeutschJozsaAnalysisResult model"""

    def test_valid_analysis_result(self):
        """Test valid analysis result creation"""
        # Create result
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

        # Create DataFrame
        df = pd.DataFrame(
            {
                "outcome": ["00", "01"],
                "probability": [0.9, 0.1],
                "counts": [900, 100],
            }
        )

        # Create analysis result
        analysis = DeutschJozsaAnalysisResult(
            result=result,
            dataframe=df,
            metadata={"experiment_type": "deutsch_jozsa", "n_qubits": 2},
        )

        assert analysis.result == result
        assert analysis.dataframe.equals(df)
        assert analysis.metadata["experiment_type"] == "deutsch_jozsa"
        assert analysis.metadata["n_qubits"] == 2

    def test_model_config(self):
        """Test model configuration allows arbitrary types"""
        # Should allow pandas DataFrame
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = DeutschJozsaResult(
            oracle_type="constant_0",
            is_constant_actual=True,
            is_constant_measured=True,
            all_zeros_probability=1.0,
            is_correct=True,
            counts={"0": 100},
            distribution={"0": 1.0},
            total_shots=100,
        )

        analysis = DeutschJozsaAnalysisResult(
            result=result,
            dataframe=df,
        )
        assert isinstance(analysis.dataframe, pd.DataFrame)
