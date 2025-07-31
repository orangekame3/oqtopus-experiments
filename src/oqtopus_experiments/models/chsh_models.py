#!/usr/bin/env python3
"""
Pydantic models for CHSH (Bell inequality) experiment
Provides structured data validation and serialization
"""

from typing import TYPE_CHECKING, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .experiment_result import ExperimentResult

if TYPE_CHECKING:
    from ..models.analysis_result import AnalysisResult


class CHSHData(BaseModel):
    """Data structure for CHSH measurements"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    measurement_counts: dict[str, dict[str, int]] = Field(
        description="Raw measurement counts for each setting"
    )
    correlations: dict[str, float] = Field(description="Individual correlation values")
    correlation_errors: dict[str, float] = Field(
        description="Standard errors of correlations"
    )
    total_shots: int = Field(description="Total number of shots")
    analysis_result: Union["CHSHAnalysisResult", None] = Field(
        default=None, description="CHSH analysis results"
    )


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


class CHSHExperimentResult(ExperimentResult):
    """Complete CHSH experiment result with comprehensive error handling"""

    def __init__(self, data: CHSHData, **kwargs):
        """Initialize with CHSH data"""
        super().__init__(**kwargs)
        self.data = data

    def analyze(
        self,
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze CHSH Bell inequality test results with comprehensive error handling

        Args:
            plot: Whether to generate plots using plot_settings.md guidelines
            save_data: Whether to save analysis results (handled by experiment classes)
            save_image: Whether to save generated plots

        Returns:
            DataFrame with analysis results including CHSH value and Bell violation test

        Note:
            Data saving is handled by experiment classes via data manager.
            This method performs data analysis with comprehensive error handling
            and validation to provide helpful feedback when issues occur.
        """
        from ..models.analysis_result import AnalysisResult

        try:
            # Extract measurement data
            data = self.data
            measurement_counts = data.measurement_counts
            total_shots = data.total_shots

            # Comprehensive input validation
            result = self._validate_chsh_inputs(measurement_counts, total_shots)
            if not result.success:
                return result.to_legacy_dataframe()

            # Check if we have sufficient measurement data
            # Support both naming conventions: A0B0/A0B1/A1B0/A1B1 and ZZ/ZX/XZ/XX
            required_settings_1 = {"A0B0", "A0B1", "A1B0", "A1B1"}
            required_settings_2 = {"ZZ", "ZX", "XZ", "XX"}
            available_settings = set(measurement_counts.keys())

            has_A_settings = required_settings_1.issubset(available_settings)
            has_Z_settings = required_settings_2.issubset(available_settings)

            if not (has_A_settings or has_Z_settings):
                missing_1 = required_settings_1 - available_settings
                missing_2 = required_settings_2 - available_settings
                result = AnalysisResult(
                    success=False,
                    errors=[
                        f"Missing measurement settings. Need either {missing_1} or {missing_2}"
                    ],
                    suggestions=[
                        "Ensure all four CHSH measurement settings are performed",
                        "Check that circuit generation includes either A0B0/A0B1/A1B0/A1B1 or ZZ/ZX/XZ/XX settings",
                    ],
                )
                return result.to_legacy_dataframe()

            # Perform CHSH analysis
            try:
                analysis_result = self._calculate_chsh_value(
                    measurement_counts, total_shots
                )

                # Validate analysis results quality
                quality_result = self._validate_analysis_quality(analysis_result)
                if not quality_result.success:
                    return quality_result.to_legacy_dataframe()

                # Update data with analysis results
                self.data.analysis_result = analysis_result

                # Generate plots if requested
                if plot:
                    try:
                        self._create_chsh_plot(analysis_result, save_image)
                    except Exception as e:
                        print(f"Warning: Plot generation failed: {e}")

                # Create DataFrame from analysis results
                df_data = []
                for setting, correlation in analysis_result.correlations.items():
                    counts = analysis_result.measurement_counts[setting]
                    total = sum(counts.values())

                    df_data.append(
                        {
                            "measurement_basis": setting,
                            "correlation": correlation,
                            "correlation_error": analysis_result.correlation_errors[
                                setting
                            ],
                            "counts_00": counts.get("00", 0),
                            "counts_01": counts.get("01", 0),
                            "counts_10": counts.get("10", 0),
                            "counts_11": counts.get("11", 0),
                            "total_shots": total,
                            "chsh_value": analysis_result.chsh_value,
                            "bell_violation": analysis_result.bell_violation,
                            "significance": analysis_result.significance,
                            "quantum_theoretical_max": analysis_result.quantum_theoretical_max,
                        }
                    )

                analysis_df = pd.DataFrame(df_data)

                result = AnalysisResult(
                    success=True,
                    data=analysis_df,
                    metadata={
                        "bell_violation": analysis_result.bell_violation,
                        "statistical_significance": f"{analysis_result.significance:.2f}σ",
                    },
                )
                return result.to_legacy_dataframe()

            except Exception as e:
                # Handle analysis failure gracefully
                result = AnalysisResult(
                    success=False,
                    errors=[f"CHSH analysis failed: {str(e)}"],
                    suggestions=[
                        "Check measurement count data format and completeness",
                        "Verify sufficient statistics for correlation calculations",
                        "Ensure all measurement settings have adequate shot counts",
                    ],
                )
                return result.to_legacy_dataframe()

        except Exception as e:
            # Handle unexpected errors
            result = AnalysisResult(
                success=False,
                errors=[f"Unexpected error in CHSH analysis: {str(e)}"],
                suggestions=[
                    "Check input data format and types",
                    "Verify all required dependencies are available",
                ],
            )
            return result.to_legacy_dataframe()

    def _validate_chsh_inputs(
        self, measurement_counts: dict[str, dict[str, int]], total_shots: int
    ) -> "AnalysisResult":
        """Validate CHSH input data"""
        from ..models.analysis_result import AnalysisResult

        # Basic validation
        if not measurement_counts:
            return AnalysisResult(
                success=False,
                errors=["Empty measurement counts"],
            )

        if total_shots <= 0:
            return AnalysisResult(
                success=False,
                errors=["Total shots must be positive"],
            )

        # Validate each measurement setting
        for setting, counts in measurement_counts.items():
            if not isinstance(counts, dict):
                return AnalysisResult(
                    success=False,
                    errors=[f"Invalid counts format for setting {setting}"],
                )

            # Ensure all required outcomes are present (fill missing with 0)
            required_outcomes = {"00", "01", "10", "11"}
            for outcome in required_outcomes:
                if outcome not in counts:
                    counts[outcome] = 0

            # Check for negative counts
            for outcome, count in counts.items():
                if count < 0:
                    return AnalysisResult(
                        success=False,
                        errors=[f"Negative count for {setting}:{outcome}"],
                    )

        return AnalysisResult(success=True, data={"validation": "passed"})

    def _calculate_chsh_value(
        self, measurement_counts: dict[str, dict[str, int]], total_shots: int
    ) -> CHSHAnalysisResult:
        """Calculate CHSH value from measurement counts"""
        import numpy as np

        correlations = {}
        correlation_errors = {}

        # Calculate correlations for each measurement setting
        for setting, counts in measurement_counts.items():
            # Calculate correlation E(A,B) = (N_00 + N_11 - N_01 - N_10) / N_total
            n_00 = counts.get("00", 0)
            n_01 = counts.get("01", 0)
            n_10 = counts.get("10", 0)
            n_11 = counts.get("11", 0)
            n_total = n_00 + n_01 + n_10 + n_11

            if n_total == 0:
                correlation = 0.0
                error = 1.0
            else:
                correlation = (n_00 + n_11 - n_01 - n_10) / n_total
                # Standard error calculation for correlation
                error = np.sqrt(1 / n_total) if n_total > 0 else 1.0

            correlations[setting] = correlation
            correlation_errors[setting] = error

        # Calculate CHSH value: S = |E(A0,B0) + E(A0,B1) + E(A1,B0) - E(A1,B1)|
        # Support both naming conventions
        if "ZZ" in correlations:
            # ZZ/ZX/XZ/XX naming (from experiment class)
            # Map to CHSH calculation: ZZ=A0B0, ZX=A0B1, XZ=A1B0, XX=A1B1
            e_a0b0 = correlations.get("ZZ", 0.0)
            e_a0b1 = correlations.get("ZX", 0.0)
            e_a1b0 = correlations.get("XZ", 0.0)
            e_a1b1 = correlations.get("XX", 0.0)

            err_a0b0 = correlation_errors.get("ZZ", 0.0)
            err_a0b1 = correlation_errors.get("ZX", 0.0)
            err_a1b0 = correlation_errors.get("XZ", 0.0)
            err_a1b1 = correlation_errors.get("XX", 0.0)
        else:
            # A0B0/A0B1/A1B0/A1B1 naming (traditional CHSH notation)
            e_a0b0 = correlations.get("A0B0", 0.0)
            e_a0b1 = correlations.get("A0B1", 0.0)
            e_a1b0 = correlations.get("A1B0", 0.0)
            e_a1b1 = correlations.get("A1B1", 0.0)

            err_a0b0 = correlation_errors.get("A0B0", 0.0)
            err_a0b1 = correlation_errors.get("A0B1", 0.0)
            err_a1b0 = correlation_errors.get("A1B0", 0.0)
            err_a1b1 = correlation_errors.get("A1B1", 0.0)

        chsh_value = abs(e_a0b0 + e_a0b1 + e_a1b0 - e_a1b1)

        # Error propagation for CHSH value
        chsh_std_error = np.sqrt(err_a0b0**2 + err_a0b1**2 + err_a1b0**2 + err_a1b1**2)

        # Bell inequality violation test
        bell_violation = chsh_value > 2.0
        significance = (
            (chsh_value - 2.0) / chsh_std_error if chsh_std_error > 0 else 0.0
        )

        return CHSHAnalysisResult(
            chsh_value=float(chsh_value),
            chsh_std_error=float(chsh_std_error),
            bell_violation=bell_violation,
            quantum_theoretical_max=2.0 * np.sqrt(2),
            significance=float(significance),
            correlations=correlations,
            correlation_errors=correlation_errors,
            measurement_counts=measurement_counts,
            total_shots=total_shots,
        )

    def _validate_analysis_quality(
        self, analysis_result: CHSHAnalysisResult
    ) -> "AnalysisResult":
        """Validate quality of CHSH analysis results"""
        from ..models.analysis_result import AnalysisResult

        warnings = []
        issues = []

        # Check statistical significance
        if analysis_result.significance < 2.0 and analysis_result.bell_violation:
            warnings.append(
                f"Low statistical significance ({analysis_result.significance:.2f}σ < 2σ) for Bell violation"
            )

        # Check for sufficient statistics
        if analysis_result.total_shots < 1000:
            warnings.append(
                f"Low shot count ({analysis_result.total_shots}) may affect statistical reliability"
            )

        # Check CHSH value reasonableness
        if analysis_result.chsh_value > analysis_result.quantum_theoretical_max + 0.1:
            issues.append(
                f"CHSH value ({analysis_result.chsh_value:.3f}) exceeds quantum limit ({analysis_result.quantum_theoretical_max:.3f})"
            )

        # Check error magnitudes
        if analysis_result.chsh_std_error > 1.0:
            warnings.append(
                f"Large CHSH error ({analysis_result.chsh_std_error:.3f}) indicates poor statistics"
            )

        if issues:
            return AnalysisResult(
                success=False,
                errors=["CHSH analysis quality issues: " + "; ".join(issues)],
                warnings=warnings,
                suggestions=[
                    "Increase number of shots for better statistics",
                    "Check for systematic errors in measurement setup",
                    "Verify proper Bell state preparation and measurement",
                ],
            )

        return AnalysisResult(
            success=True, warnings=warnings, data={"quality_check": "passed"}
        )

    def _create_chsh_plot(
        self, analysis_result: CHSHAnalysisResult, save_image: bool = True
    ):
        """Create CHSH analysis plot following plot_settings.md guidelines"""
        try:
            import plotly.graph_objects as go

            from ..utils.visualization import (
                get_experiment_colors,
                get_plotly_config,
                save_plotly_figure,
                setup_plotly_environment,
                show_plotly_figure,
            )

            # Setup plotly environment
            setup_plotly_environment()
            colors = get_experiment_colors()

            # Create figure with two subplots
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Correlation Values", "CHSH Test Result"],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]],
            )

            # Plot correlations
            settings = list(analysis_result.correlations.keys())
            corr_values = list(analysis_result.correlations.values())
            corr_errors = [analysis_result.correlation_errors[s] for s in settings]

            fig.add_trace(
                go.Bar(
                    x=settings,
                    y=corr_values,
                    error_y={"type": "data", "array": corr_errors, "visible": True},
                    name="Correlations",
                    marker_color=colors[1],
                ),
                row=1,
                col=1,
            )

            # Plot CHSH value comparison
            chsh_categories = ["Classical Limit", "Measured CHSH", "Quantum Limit"]
            chsh_values = [
                2.0,
                analysis_result.chsh_value,
                analysis_result.quantum_theoretical_max,
            ]
            bar_colors = [
                "gray",
                colors[0] if analysis_result.bell_violation else "red",
                "lightblue",
            ]

            fig.add_trace(
                go.Bar(
                    x=chsh_categories,
                    y=chsh_values,
                    name="CHSH Values",
                    marker_color=bar_colors,
                    error_y={
                        "type": "data",
                        "array": [0, analysis_result.chsh_std_error, 0],
                        "visible": True,
                    },
                ),
                row=1,
                col=2,
            )

            # Update layout
            fig.update_layout(
                title_text=f"CHSH Bell Inequality Test (S = {analysis_result.chsh_value:.3f})",
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
                width=1000,
                height=500,
            )

            # Update axes
            fig.update_xaxes(title_text="Measurement Setting", row=1, col=1)
            fig.update_yaxes(
                title_text="Correlation E(A,B)", row=1, col=1, range=[-1.1, 1.1]
            )
            fig.update_xaxes(title_text="", row=1, col=2)
            fig.update_yaxes(title_text="CHSH Value", row=1, col=2)

            # Add violation annotation
            violation_text = (
                "Bell Violation!" if analysis_result.bell_violation else "No Violation"
            )
            significance_text = f"({analysis_result.significance:.2f}σ)"

            fig.add_annotation(
                x=0.98,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"{violation_text}<br>{significance_text}",
                showarrow=False,
                align="right",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#CCCCCC",
                borderwidth=1,
            )

            # Save and show plot
            if save_image:
                save_plotly_figure(fig, name="chsh_analysis", images_dir="./plots")

            config = get_plotly_config("chsh_analysis", 1000, 500)
            show_plotly_figure(fig, config)

        except Exception as e:
            print(f"Warning: CHSH plot generation failed: {e}")


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


# Fix forward reference
CHSHData.model_rebuild()
