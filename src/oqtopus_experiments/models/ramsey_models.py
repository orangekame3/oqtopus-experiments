#!/usr/bin/env python3
"""
Pydantic models for Ramsey experiment
Provides structured data validation and serialization
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class RamseyFittingResult(BaseModel):
    """Fitting results for Ramsey fringe oscillation"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t2_star_time: float = Field(description="T2* dephasing time in ns")
    frequency: float = Field(description="Oscillation frequency in Hz")
    amplitude: float = Field(description="Oscillation amplitude")
    offset: float = Field(description="Fitted offset")
    phase: float = Field(description="Oscillation phase")
    r_squared: float = Field(description="R-squared goodness of fit")
    delay_times: list[float] = Field(description="Input delay time values")
    probabilities: list[float] = Field(description="Measured probabilities")
    error_info: str | None = Field(
        default=None, description="Error information if fitting failed"
    )


class RamseyParameters(BaseModel):
    """Ramsey experiment parameters"""

    experiment_name: str | None = Field(default=None, description="Experiment name")
    physical_qubit: int = Field(default=0, description="Target physical qubit")
    delay_points: int = Field(default=20, description="Number of delay points")
    max_delay: float = Field(default=10000.0, description="Maximum delay time in ns")
    detuning_frequency: float = Field(
        default=0.0, description="Detuning frequency in Hz"
    )


class RamseyAnalysisResult(BaseModel):
    """Complete Ramsey analysis results"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fitting_result: RamseyFittingResult = Field(description="Fitting results")
    dataframe: pd.DataFrame = Field(description="Analysis results as DataFrame")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class RamseyCircuitParams(BaseModel):
    """Parameters for individual Ramsey circuits"""

    experiment: str = Field(default="ramsey", description="Experiment type")
    delay_time: float = Field(description="Delay time in ns")
    detuning_frequency: float = Field(description="Detuning frequency in Hz")
    logical_qubit: int = Field(description="Logical qubit index")
    physical_qubit: int = Field(description="Physical qubit index")
