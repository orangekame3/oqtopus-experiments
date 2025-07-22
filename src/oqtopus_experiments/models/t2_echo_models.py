#!/usr/bin/env python3
"""
Pydantic models for T2 Echo experiment
Provides structured data validation and serialization
"""

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class T2EchoFittingResult(BaseModel):
    """Fitting results for T2 Echo decay"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t2_time: float = Field(description="T2 coherence time in ns")
    amplitude: float = Field(description="Fitted amplitude")
    offset: float = Field(description="Fitted offset")
    r_squared: float = Field(description="R-squared goodness of fit")
    delay_times: list[float] = Field(description="Input delay time values")
    probabilities: list[float] = Field(description="Measured probabilities")
    error_info: str | None = Field(
        default=None, description="Error information if fitting failed"
    )


class T2EchoParameters(BaseModel):
    """T2 Echo experiment parameters"""

    experiment_name: str | None = Field(default=None, description="Experiment name")
    physical_qubit: int = Field(default=0, description="Target physical qubit")
    delay_points: int = Field(default=20, description="Number of delay points")
    max_delay: float = Field(default=30000.0, description="Maximum delay time in ns")


class T2EchoAnalysisResult(BaseModel):
    """Complete T2 Echo analysis results"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fitting_result: T2EchoFittingResult = Field(description="Fitting results")
    dataframe: pd.DataFrame = Field(description="Analysis results as DataFrame")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class T2EchoCircuitParams(BaseModel):
    """Parameters for individual T2 Echo circuits"""

    experiment: str = Field(default="t2_echo", description="Experiment type")
    delay_time: float = Field(description="Delay time in ns")
    logical_qubit: int = Field(description="Logical qubit index")
    physical_qubit: int = Field(description="Physical qubit index")
