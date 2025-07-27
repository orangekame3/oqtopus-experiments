#!/usr/bin/env python3
"""
Bernstein-Vazirani algorithm experiment models for parameter validation and result storage.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class BernsteinVaziraniParameters(BaseModel):
    """Parameters for Bernstein-Vazirani algorithm experiment

    The BV algorithm finds an n-bit secret string s with a single quantum query
    to an oracle that computes f_s(x) = s·x (dot product mod 2).
    """

    experiment_name: str | None = Field(
        default=None, description="Name of the experiment"
    )
    secret_string: str = Field(description="Binary string to find (e.g., '1011')")
    n_bits: int = Field(ge=1, le=20, description="Number of bits in the secret string")

    @field_validator("secret_string")
    @classmethod
    def validate_secret_string(cls, v: str) -> str:
        """Validate that secret string contains only 0s and 1s"""
        if not v:
            raise ValueError("Secret string cannot be empty")
        if not all(bit in "01" for bit in v):
            raise ValueError(f"Secret string must contain only 0s and 1s, got: {v}")
        return v

    @field_validator("n_bits")
    @classmethod
    def validate_n_bits(cls, v: int, info: ValidationInfo) -> int:
        """Validate that n_bits matches secret_string length"""
        if "secret_string" in info.data:
            secret_string = info.data["secret_string"]
            if len(secret_string) != v:
                raise ValueError(
                    f"n_bits ({v}) must match secret_string length ({len(secret_string)})"
                )
        return v


class BernsteinVaziraniCircuitParams(BaseModel):
    """Circuit parameters for Bernstein-Vazirani algorithm"""

    secret_string: str = Field(description="The secret binary string")
    n_bits: int = Field(description="Number of bits in the secret")


class BernsteinVaziraniResult(BaseModel):
    """Results from Bernstein-Vazirani algorithm execution"""

    secret_string: str = Field(description="The actual secret string")
    measured_string: str = Field(description="The string measured by the algorithm")
    success_probability: float = Field(
        ge=0, le=1, description="Probability of measuring the correct secret"
    )
    is_correct: bool = Field(
        description="Whether the measured string matches the secret"
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


class BernsteinVaziraniAnalysisResult(BaseModel):
    """Complete analysis result for Bernstein-Vazirani algorithm"""

    result: BernsteinVaziraniResult = Field(description="Algorithm execution results")
    dataframe: pd.DataFrame = Field(
        description="Processed experiment data with all measurement outcomes"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional experiment metadata"
    )

    model_config = {"arbitrary_types_allowed": True}
