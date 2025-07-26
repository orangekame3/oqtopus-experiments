#!/usr/bin/env python3
"""
Hadamard Test experiment models for parameter validation and result storage.
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class HadamardTestParameters(BaseModel):
    """Parameters for Hadamard Test experiment

    This implementation supports X, Y, Z unitaries with optimal initial state preparation:
    - X unitary: RY(θ) initial state → ⟨ψ(θ)|X|ψ(θ)⟩ = sin(θ)
    - Y unitary: RX(θ) initial state → ⟨ψ(θ)|Y|ψ(θ)⟩ = -sin(θ) (Qiskit RX convention)
    - Z unitary: RX(θ) initial state → ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ) (default)
    """

    experiment_name: str | None = Field(
        default=None, description="Name of the experiment"
    )
    physical_qubit: int = Field(default=0, description="Physical qubit to use")
    test_unitary: str = Field(default="Z", description="Test unitary gate (X, Y, or Z)")
    angle_points: int = Field(
        default=16, ge=4, description="Number of angle points to measure"
    )
    max_angle: float = Field(
        default=6.28318530718, gt=0, description="Maximum angle in radians (2π)"
    )


class HadamardTestCircuitParams(BaseModel):
    """Circuit parameters for individual Hadamard Test measurement"""

    angle: float = Field(description="Rotation angle parameter")
    test_unitary: str = Field(description="Type of unitary gate")
    logical_qubit: int = Field(default=0, description="Logical qubit index")
    physical_qubit: int = Field(default=0, description="Physical qubit index")


class HadamardTestFittingResult(BaseModel):
    """Results from Hadamard Test fitting analysis"""

    real_part: float = Field(description="Real part of expectation value")
    imaginary_part: float = Field(description="Imaginary part of expectation value")
    magnitude: float = Field(description="Magnitude of complex expectation value")
    phase: float = Field(description="Phase of complex expectation value")
    phase_degrees: float = Field(description="Phase in degrees")
    fidelity: float = Field(ge=0, le=1, description="Quality metric for the fit (0-1)")
    angles: list[float] = Field(description="Angle values used in fitting")
    probabilities: list[float] = Field(description="Measured probabilities")
    r_squared: float = Field(
        default=0.0, ge=0, le=1, description="R-squared goodness of fit"
    )
    error_info: str | None = Field(
        default=None, description="Error information if fitting failed"
    )


class HadamardTestAnalysisResult(BaseModel):
    """Complete analysis result for Hadamard Test"""

    fitting_result: HadamardTestFittingResult = Field(
        description="Fitting analysis results"
    )
    dataframe: pd.DataFrame = Field(description="Processed experiment data")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional experiment metadata"
    )

    class Config:
        arbitrary_types_allowed = True
