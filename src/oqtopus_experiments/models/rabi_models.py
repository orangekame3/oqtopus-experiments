#!/usr/bin/env python3
"""
Pydantic models for Rabi experiment
Provides structured data validation and serialization
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict


class RabiFittingResult(BaseModel):
    """Fitting results for Rabi oscillation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    pi_amplitude: float = Field(description="π-pulse amplitude")
    frequency: float = Field(description="Rabi frequency")
    fit_amplitude: float = Field(description="Fitted amplitude")
    offset: float = Field(description="Fitted offset")
    r_squared: float = Field(description="R-squared goodness of fit")
    amplitudes: list[float] = Field(description="Input amplitude values")
    probabilities: list[float] = Field(description="Measured probabilities")
    error_info: str | None = Field(default=None, description="Error information if fitting failed")


class RabiParameters(BaseModel):
    """Rabi experiment parameters"""
    experiment_name: str | None = Field(default=None, description="Experiment name")
    physical_qubit: int = Field(default=0, description="Target physical qubit")
    amplitude_points: int = Field(default=10, description="Number of amplitude points")
    max_amplitude: float = Field(default=2.0, description="Maximum amplitude")


class RabiAnalysisResult(BaseModel):
    """Complete Rabi analysis results"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    fitting_result: RabiFittingResult = Field(description="Fitting results")
    dataframe: pd.DataFrame = Field(description="Analysis results as DataFrame")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RabiCircuitParams(BaseModel):
    """Parameters for individual Rabi circuits"""
    experiment: str = Field(default="rabi", description="Experiment type")
    amplitude: float = Field(description="Drive amplitude")
    logical_qubit: int = Field(description="Logical qubit index")
    physical_qubit: int = Field(description="Physical qubit index")
    rotation_angle: float = Field(description="Rotation angle in radians")