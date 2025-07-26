#!/usr/bin/env python3
"""
Pydantic models for CHSH Phase Scan experiments
"""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class CHSHPhaseScanParameters(BaseModel):
    """Parameters for CHSH phase scan experiment"""

    experiment_name: str | None = Field(
        default=None, description="Name of the experiment"
    )
    physical_qubit_0: int = Field(
        default=0, description="Physical qubit index for Alice (first qubit)"
    )
    physical_qubit_1: int = Field(
        default=1, description="Physical qubit index for Bob (second qubit)"
    )
    shots_per_circuit: int = Field(
        default=1000, description="Number of shots per circuit"
    )
    phase_points: int = Field(default=21, description="Number of phase points to scan")
    phase_start: float = Field(default=0.0, description="Starting phase in radians")
    phase_end: float = Field(default=2 * np.pi, description="Ending phase in radians")


class CHSHPhaseScanPoint(BaseModel):
    """Results for a single phase point"""

    phase_radians: float = Field(description="Phase angle in radians")
    phase_degrees: float = Field(description="Phase angle in degrees")
    chsh1_value: float = Field(description="CHSH1 = <ZZ> - <ZX> + <XZ> + <XX>")
    chsh2_value: float = Field(description="CHSH2 = <ZZ> + <ZX> - <XZ> + <XX>")
    chsh_max: float = Field(description="Maximum of |CHSH1| and |CHSH2|")
    bell_violation: bool = Field(description="Whether Bell inequality is violated")
    correlations: dict[str, float] = Field(description="ZZ, ZX, XZ, XX correlations")
    correlation_errors: dict[str, float] = Field(
        description="Correlation standard errors"
    )
    total_shots: int = Field(description="Total shots for this phase point")


class CHSHPhaseScanAnalysisResult(BaseModel):
    """Complete analysis results for CHSH phase scan"""

    phase_points: list[CHSHPhaseScanPoint] = Field(
        description="Results for each phase point"
    )
    max_chsh_value: float = Field(description="Maximum CHSH value across all phases")
    max_chsh_phase: float = Field(
        description="Phase (radians) where maximum CHSH occurs"
    )
    violation_count: int = Field(
        description="Number of phase points with Bell violation"
    )
    theoretical_max: float = Field(
        default=2 * np.sqrt(2), description="Theoretical CHSH maximum"
    )

    def get_phases(self) -> np.ndarray:
        """Get array of phase values in radians"""
        return np.array([point.phase_radians for point in self.phase_points])

    def get_phases_degrees(self) -> np.ndarray:
        """Get array of phase values in degrees"""
        return np.array([point.phase_degrees for point in self.phase_points])

    def get_chsh1_values(self) -> np.ndarray:
        """Get array of CHSH1 values"""
        return np.array([point.chsh1_value for point in self.phase_points])

    def get_chsh2_values(self) -> np.ndarray:
        """Get array of CHSH2 values"""
        return np.array([point.chsh2_value for point in self.phase_points])

    def get_chsh_max_values(self) -> np.ndarray:
        """Get array of maximum CHSH values"""
        return np.array([point.chsh_max for point in self.phase_points])


class CHSHPhaseScanCircuitParams(BaseModel):
    """Circuit parameters for CHSH phase scan"""

    phase_radians: float = Field(description="Phase angle in radians")
    measurement_basis: str = Field(description="Measurement basis (ZZ, ZX, XZ, XX)")
    logical_qubit_0: int = Field(default=0, description="Logical qubit 0 index")
    logical_qubit_1: int = Field(default=1, description="Logical qubit 1 index")
    physical_qubit_0: int = Field(default=0, description="Physical qubit 0 index")
    physical_qubit_1: int = Field(default=1, description="Physical qubit 1 index")


class CHSHPhaseScanExperimentResult(BaseModel):
    """Complete experiment result for CHSH phase scan"""

    analysis_result: CHSHPhaseScanAnalysisResult = Field(description="Analysis results")
    dataframe: Any = Field(description="Results as pandas DataFrame")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Experiment metadata"
    )

    model_config = {"arbitrary_types_allowed": True}
