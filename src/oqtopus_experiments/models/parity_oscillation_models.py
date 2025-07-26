#!/usr/bin/env python3
"""
Pydantic models for Parity Oscillation experiments
"""

from typing import Any

from pydantic import BaseModel, Field


class ParityOscillationParameters(BaseModel):
    """Parameters for Parity Oscillation experiment"""

    experiment_name: str | None = Field(
        default=None, description="Name of the experiment"
    )
    num_qubits_list: list[int] = Field(
        default=[1, 2, 3, 4, 5], description="List of qubit counts to test"
    )
    delays_us: list[float] = Field(
        default=[0, 1, 2, 4, 8, 16], description="List of delay times in microseconds"
    )
    phase_points: int | None = Field(
        default=None, description="Number of phase points (None for auto: 4N+1)"
    )
    no_delay: bool = Field(
        default=False, description="Skip delay insertion for faster execution"
    )
    shots_per_circuit: int = Field(
        default=1000, description="Number of shots per circuit"
    )


class ParityOscillationPoint(BaseModel):
    """Results for a single measurement point"""

    num_qubits: int = Field(description="Number of qubits in GHZ state")
    delay_us: float = Field(description="Delay time in microseconds")
    phase_radians: float = Field(description="Rotation phase in radians")
    phase_degrees: float = Field(description="Rotation phase in degrees")
    parity: float = Field(description="Measured parity value")
    parity_error: float = Field(description="Parity measurement error")
    total_shots: int = Field(description="Total shots for this measurement")
    counts: dict[str, int] = Field(description="Raw measurement counts")


class ParityOscillationFitting(BaseModel):
    """Sinusoidal fitting results for parity oscillation"""

    amplitude: float = Field(description="Fitted oscillation amplitude")
    phase_offset: float = Field(description="Fitted phase offset")
    vertical_offset: float = Field(description="Fitted vertical offset")
    frequency: float = Field(description="Fitted frequency (should be ~1 for GHZ)")
    r_squared: float = Field(description="R-squared goodness of fit")
    coherence: float = Field(description="Extracted coherence C(N,τ)")


class ParityOscillationAnalysisResult(BaseModel):
    """Complete analysis results for Parity Oscillation experiment"""

    measurement_points: list[ParityOscillationPoint] = Field(
        description="All measurement points"
    )
    fitting_results: dict[str, ParityOscillationFitting] = Field(
        description="Fitting results for each (N, τ) combination"
    )
    coherence_matrix: dict[str, float] = Field(
        description="Coherence C(N,τ) for all combinations"
    )
    max_coherence: float = Field(
        description="Maximum coherence across all measurements"
    )
    decoherence_rates: dict[int, float] = Field(
        description="Decoherence rates γ for each qubit count"
    )

    def get_num_qubits_list(self) -> list[int]:
        """Get list of unique qubit counts"""
        return sorted({point.num_qubits for point in self.measurement_points})

    def get_delays_list(self) -> list[float]:
        """Get list of unique delay times"""
        return sorted({point.delay_us for point in self.measurement_points})

    def get_coherence_for_qubits(self, num_qubits: int) -> dict[float, float]:
        """Get coherence vs delay for specific qubit count"""
        coherence_vs_delay = {}
        for key, coherence in self.coherence_matrix.items():
            if key.startswith(f"N{num_qubits}_"):
                delay = float(key.split("_")[1].replace("τ", "").replace("us", ""))
                coherence_vs_delay[delay] = coherence
        return coherence_vs_delay


class ParityOscillationCircuitParams(BaseModel):
    """Circuit parameters for Parity Oscillation experiment"""

    num_qubits: int = Field(description="Number of qubits in GHZ state")
    delay_us: float = Field(description="Delay time in microseconds")
    phase_radians: float = Field(description="Rotation phase in radians")
    no_delay: bool = Field(default=False, description="Skip delay insertion")


class ParityOscillationExperimentResult(BaseModel):
    """Complete experiment result for Parity Oscillation"""

    analysis_result: ParityOscillationAnalysisResult = Field(
        description="Analysis results"
    )
    dataframe: Any = Field(description="Results as pandas DataFrame")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Experiment metadata"
    )

    model_config = {"arbitrary_types_allowed": True}
