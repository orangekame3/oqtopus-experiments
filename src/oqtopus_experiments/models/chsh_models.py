#!/usr/bin/env python3
"""
Pydantic models for CHSH (Bell inequality) experiment
Provides structured data validation and serialization
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class CHSHAnalysisResult(BaseModel):
    """CHSH analysis results"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chsh_value: float = Field(description="CHSH value (S)")
    chsh_std_error: float = Field(description="Standard error of CHSH value")
    bell_violation: bool = Field(
        description="Whether Bell inequality is violated (S > 2)"
    )
    quantum_theoretical_max: float = Field(
        default=2.828, description="Quantum theoretical maximum (2√2)"
    )
    significance: float = Field(
        description="Statistical significance of violation (in σ)"
    )
    correlations: dict[str, float] = Field(description="Individual correlation values")
    correlation_errors: dict[str, float] = Field(
        description="Standard errors of correlations"
    )
    measurement_counts: dict[str, dict[str, int]] = Field(
        description="Raw measurement counts"
    )
    total_shots: int = Field(description="Total number of shots")


class CHSHParameters(BaseModel):
    """CHSH experiment parameters"""

    experiment_name: str | None = Field(default=None, description="Experiment name")
    physical_qubit_0: int = Field(default=0, description="First physical qubit")
    physical_qubit_1: int = Field(default=1, description="Second physical qubit")
    shots_per_circuit: int = Field(
        default=1000, description="Number of shots per circuit"
    )
    theta: float = Field(
        default=0.0, description="Rotation angle parameter for Bell state"
    )
    measurement_angles: dict[str, float] = Field(
        default={
            "alice_0": 0.0,  # θ_A0 = 0°
            "alice_1": 90.0,  # θ_A1 = 90°
            "bob_0": 45.0,  # θ_B0 = 45°
            "bob_1": 135.0,  # θ_B1 = 135°
        },
        description="Measurement angles in degrees for Alice and Bob",
    )


class CHSHExperimentResult(BaseModel):
    """Complete CHSH experiment results"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    analysis_result: CHSHAnalysisResult = Field(description="Analysis results")
    dataframe: pd.DataFrame = Field(description="Detailed results as DataFrame")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class CHSHCircuitParams(BaseModel):
    """Parameters for individual CHSH circuits"""

    experiment: str = Field(default="chsh", description="Experiment type")
    measurement_setting: str = Field(description="Measurement setting (e.g., 'A0B0')")
    alice_angle: float = Field(description="Alice's measurement angle in degrees")
    bob_angle: float = Field(description="Bob's measurement angle in degrees")
    logical_qubit_0: int = Field(description="First logical qubit index")
    logical_qubit_1: int = Field(description="Second logical qubit index")
    physical_qubit_0: int = Field(description="First physical qubit index")
    physical_qubit_1: int = Field(description="Second physical qubit index")
