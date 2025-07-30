#!/usr/bin/env python3
"""
Deutsch-Jozsa algorithm experiment models for parameter validation and result storage.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class DeutschJozsaParameters(BaseModel):
    """Parameters for Deutsch-Jozsa algorithm experiment

    The DJ algorithm determines whether a Boolean function is constant or balanced
    with a single quantum query to an oracle.
    """

    experiment_name: str | None = Field(
        default=None, description="Name of the experiment"
    )
    n_qubits: int = Field(default=3, ge=1, le=10, description="Number of input qubits")
    oracle_type: str = Field(
        default="balanced_random",
        description="Type of oracle function: constant_0, constant_1, balanced_random, balanced_alternating",
    )

    @field_validator("oracle_type")
    @classmethod
    def validate_oracle_type(cls, v: str) -> str:
        """Validate oracle type"""
        valid_types = [
            "constant_0",
            "constant_1",
            "balanced_random",
            "balanced_alternating",
        ]
        if v not in valid_types:
            raise ValueError(f"Oracle type must be one of {valid_types}, got: {v}")
        return v


class DeutschJozsaCircuitParams(BaseModel):
    """Circuit parameters for Deutsch-Jozsa algorithm"""

    n_qubits: int = Field(description="Number of input qubits")
    oracle_type: str = Field(description="Type of oracle function")


class DeutschJozsaResult(BaseModel):
    """Results from Deutsch-Jozsa algorithm execution"""

    oracle_type: str = Field(description="Type of oracle used")
    is_constant_actual: bool = Field(
        description="Whether the oracle function is actually constant"
    )
    is_constant_measured: bool = Field(
        description="Whether the algorithm measured the function as constant"
    )
    all_zeros_probability: float = Field(
        ge=0, le=1, description="Probability of measuring all zeros"
    )
    is_correct: bool = Field(
        description="Whether the measurement correctly identified the function type"
    )
    counts: dict[str, int] = Field(
        description="Raw measurement counts for each outcome"
    )
    distribution: dict[str, float] = Field(
        description="Probability distribution of measurement outcomes"
    )
    total_shots: int = Field(ge=1, description="Total number of measurement shots")

    @field_validator("distribution")
    @classmethod
    def validate_distribution(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate that probabilities sum to 1"""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:  # Allow small numerical errors
            raise ValueError(f"Probabilities must sum to 1, got {total}")
        return v


class DeutschJozsaAnalysisResult(BaseModel):
    """Complete analysis result for Deutsch-Jozsa algorithm"""

    result: DeutschJozsaResult = Field(description="Algorithm execution results")
    dataframe: pd.DataFrame = Field(
        description="Processed experiment data with all measurement outcomes"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional experiment metadata"
    )

    model_config = {"arbitrary_types_allowed": True}
