#!/usr/bin/env python3
"""
Tests for BernsteinVazirani models
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from oqtopus_experiments.models.bernstein_vazirani_models import (
    BernsteinVaziraniAnalysisResult,
    BernsteinVaziraniCircuitParams,
    BernsteinVaziraniParameters,
    BernsteinVaziraniResult,
)


class TestBernsteinVaziraniParameters:
    """Test BernsteinVaziraniParameters model"""

    def test_valid_parameters(self):
        """Test creating valid parameters"""
        params = BernsteinVaziraniParameters(
            experiment_name="test_bv",
            secret_string="1011",
            n_bits=4,
        )

        assert params.experiment_name == "test_bv"
        assert params.secret_string == "1011"
        assert params.n_bits == 4

    def test_default_experiment_name(self):
        """Test default experiment name is None"""
        params = BernsteinVaziraniParameters(
            secret_string="101",
            n_bits=3,
        )

        assert params.experiment_name is None
        assert params.secret_string == "101"
        assert params.n_bits == 3

    def test_invalid_secret_string(self):
        """Test validation of secret string"""
        # Empty string
        with pytest.raises(ValidationError):
            BernsteinVaziraniParameters(
                secret_string="",
                n_bits=0,
            )

        # Invalid characters
        with pytest.raises(ValidationError):
            BernsteinVaziraniParameters(
                secret_string="102",
                n_bits=3,
            )

        with pytest.raises(ValidationError):
            BernsteinVaziraniParameters(
                secret_string="abc",
                n_bits=3,
            )

    def test_n_bits_mismatch(self):
        """Test validation of n_bits matching secret_string length"""
        with pytest.raises(ValidationError):
            BernsteinVaziraniParameters(
                secret_string="1011",
                n_bits=3,  # Should be 4
            )

        with pytest.raises(ValidationError):
            BernsteinVaziraniParameters(
                secret_string="10",
                n_bits=5,  # Should be 2
            )

    def test_n_bits_range(self):
        """Test n_bits range validation"""
        # Valid range
        params = BernsteinVaziraniParameters(
            secret_string="1",
            n_bits=1,
        )
        assert params.n_bits == 1

        params = BernsteinVaziraniParameters(
            secret_string="1" * 20,
            n_bits=20,
        )
        assert params.n_bits == 20

        # Too large
        with pytest.raises(ValidationError):
            BernsteinVaziraniParameters(
                secret_string="1" * 21,
                n_bits=21,
            )


class TestBernsteinVaziraniCircuitParams:
    """Test BernsteinVaziraniCircuitParams model"""

    def test_valid_circuit_params(self):
        """Test creating valid circuit parameters"""
        params = BernsteinVaziraniCircuitParams(
            secret_string="1101",
            n_bits=4,
        )

        assert params.secret_string == "1101"
        assert params.n_bits == 4

    def test_model_dump(self):
        """Test model serialization"""
        params = BernsteinVaziraniCircuitParams(
            secret_string="111",
            n_bits=3,
        )

        dumped = params.model_dump()
        assert dumped == {
            "secret_string": "111",
            "n_bits": 3,
        }


class TestBernsteinVaziraniResult:
    """Test BernsteinVaziraniResult model"""

    def test_valid_result(self):
        """Test creating valid result"""
        result = BernsteinVaziraniResult(
            secret_string="101",
            measured_string="101",
            success_probability=0.95,
            is_correct=True,
            counts={"101": 950, "111": 30, "001": 20},
            distribution={"101": 0.95, "111": 0.03, "001": 0.02},
            total_shots=1000,
        )

        assert result.secret_string == "101"
        assert result.measured_string == "101"
        assert result.success_probability == 0.95
        assert result.is_correct is True
        assert result.total_shots == 1000

    def test_probability_validation(self):
        """Test success_probability range validation"""
        # Valid range
        result = BernsteinVaziraniResult(
            secret_string="11",
            measured_string="11",
            success_probability=1.0,
            is_correct=True,
            counts={"11": 1000},
            distribution={"11": 1.0},
            total_shots=1000,
        )
        assert result.success_probability == 1.0

        # Invalid range
        with pytest.raises(ValidationError):
            BernsteinVaziraniResult(
                secret_string="11",
                measured_string="11",
                success_probability=1.5,  # > 1
                is_correct=True,
                counts={"11": 1000},
                distribution={"11": 1.0},
                total_shots=1000,
            )

    def test_distribution_validation(self):
        """Test distribution probabilities sum validation"""
        # Valid distribution
        result = BernsteinVaziraniResult(
            secret_string="10",
            measured_string="10",
            success_probability=0.8,
            is_correct=True,
            counts={"10": 800, "00": 150, "11": 50},
            distribution={"10": 0.8, "00": 0.15, "11": 0.05},
            total_shots=1000,
        )
        assert sum(result.distribution.values()) == 1.0

        # Invalid distribution (doesn't sum to 1)
        with pytest.raises(ValidationError):
            BernsteinVaziraniResult(
                secret_string="10",
                measured_string="10",
                success_probability=0.8,
                is_correct=True,
                counts={"10": 800, "00": 150},
                distribution={"10": 0.8, "00": 0.1},  # Sums to 0.9
                total_shots=1000,
            )

    def test_total_shots_validation(self):
        """Test total_shots must be positive"""
        with pytest.raises(ValidationError):
            BernsteinVaziraniResult(
                secret_string="1",
                measured_string="1",
                success_probability=1.0,
                is_correct=True,
                counts={"1": 0},
                distribution={"1": 1.0},
                total_shots=0,  # Invalid
            )


class TestBernsteinVaziraniAnalysisResult:
    """Test BernsteinVaziraniAnalysisResult model"""

    def test_valid_analysis_result(self):
        """Test creating valid analysis result"""
        bv_result = BernsteinVaziraniResult(
            secret_string="110",
            measured_string="110",
            success_probability=0.99,
            is_correct=True,
            counts={"110": 990, "111": 10},
            distribution={"110": 0.99, "111": 0.01},
            total_shots=1000,
        )

        df = pd.DataFrame(
            {
                "outcome": ["110", "111"],
                "probability": [0.99, 0.01],
            }
        )

        analysis = BernsteinVaziraniAnalysisResult(
            result=bv_result,
            dataframe=df,
            metadata={"test": "metadata"},
        )

        assert analysis.result == bv_result
        assert analysis.dataframe.equals(df)
        assert analysis.metadata == {"test": "metadata"}

    def test_empty_metadata(self):
        """Test default empty metadata"""
        bv_result = BernsteinVaziraniResult(
            secret_string="1",
            measured_string="1",
            success_probability=1.0,
            is_correct=True,
            counts={"1": 100},
            distribution={"1": 1.0},
            total_shots=100,
        )

        df = pd.DataFrame({"outcome": ["1"]})

        analysis = BernsteinVaziraniAnalysisResult(
            result=bv_result,
            dataframe=df,
        )

        assert analysis.metadata == {}
