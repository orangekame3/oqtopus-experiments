#!/usr/bin/env python3
"""
Pydantic models for Rabi experiment
Provides structured data validation and serialization
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .experiment_result import ExperimentResult


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
    error_info: str | None = Field(
        default=None, description="Error information if fitting failed"
    )


class RabiParameters(BaseModel):
    """Rabi experiment parameters"""

    experiment_name: str | None = Field(default=None, description="Experiment name")
    physical_qubit: int = Field(default=0, description="Target physical qubit")
    amplitude_points: int = Field(default=10, description="Number of amplitude points")
    max_amplitude: float = Field(default=2.0, description="Maximum amplitude")


class RabiData(BaseModel):
    """Data structure for Rabi measurements"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    amplitudes: list[float] = Field(description="Drive amplitudes")
    probabilities: list[float] = Field(description="Measured probabilities")
    probability_errors: list[float] = Field(
        description="Standard errors in probabilities"
    )
    shots_per_point: int = Field(description="Number of shots per amplitude point")
    fitting_result: RabiFittingResult | None = Field(
        default=None, description="Rabi oscillation fitting results"
    )


class RabiAnalysisResult(ExperimentResult):
    """Complete Rabi experiment result with comprehensive error handling"""

    def __init__(self, data: RabiData, **kwargs):
        """Initialize with Rabi data"""
        super().__init__(**kwargs)
        self.data = data

    def analyze(
        self,
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze Rabi oscillation results with comprehensive error handling

        Args:
            plot: Whether to generate plots using plot_settings.md guidelines
            save_data: Whether to save analysis results (handled by experiment classes)
            save_image: Whether to save generated plots

        Returns:
            DataFrame with analysis results including fitted parameters

        Note:
            Data saving is handled by experiment classes via data manager.
            This method performs data analysis with comprehensive error handling
            and validation to provide helpful feedback when issues occur.
        """
        from ..exceptions import FittingError, InsufficientDataError
        from ..models.analysis_result import AnalysisResult

        try:
            # Extract measurement data
            data = self.data
            amplitudes = data.amplitudes
            probabilities = data.probabilities
            prob_errors = data.probability_errors

            # Comprehensive input validation
            result = self._validate_rabi_inputs(amplitudes, probabilities, prob_errors)
            if not result.success:
                return result.to_legacy_dataframe()

            # Check if we have enough data for fitting
            if len(amplitudes) < 3:
                # Create result with insufficient data
                fitting_result = RabiFittingResult(
                    pi_amplitude=0.0,
                    frequency=0.0,
                    fit_amplitude=0.5,
                    offset=0.5,
                    r_squared=0.0,
                    amplitudes=amplitudes,
                    probabilities=probabilities,
                    error_info="Insufficient data points for fitting (need at least 3)",
                )
                result.add_warning("Insufficient data for Rabi curve fitting")
                result.add_suggestion(
                    "Collect more amplitude points for reliable Rabi fitting"
                )
            else:
                # Attempt curve fitting with detailed error handling
                try:
                    fitting_result = self._fit_rabi_oscillation(
                        amplitudes, probabilities, prob_errors
                    )
                    if fitting_result.error_info:
                        result.add_warning(
                            f"Rabi fitting issues: {fitting_result.error_info}"
                        )
                        result.add_suggestion(
                            "Check data quality and Rabi drive calibration"
                        )
                except Exception as e:
                    # Create failed fitting result
                    fitting_result = RabiFittingResult(
                        pi_amplitude=0.0,
                        frequency=0.0,
                        fit_amplitude=0.5,
                        offset=0.5,
                        r_squared=0.0,
                        amplitudes=amplitudes,
                        probabilities=probabilities,
                        error_info=f"Rabi fitting failed: {str(e)}",
                    )
                    result.add_error(f"Rabi curve fitting failed: {str(e)}")
                    result.add_suggestion(
                        "Check if data shows expected Rabi oscillation pattern"
                    )
                    result.add_suggestion("Verify drive amplitude calibration")

            # Create DataFrame with comprehensive data
            df_data = []
            for i, amplitude in enumerate(amplitudes):
                row_data = {
                    "amplitude": amplitude,
                    "probability": probabilities[i],
                    "probability_error": prob_errors[i],
                    "shots": data.shots_per_point,
                }

                # Add quality indicators
                if prob_errors[i] > 0.1:  # High uncertainty
                    result.add_warning(
                        f"High uncertainty at amplitude {amplitude}: σ={prob_errors[i]:.3f}"
                    )

                df_data.append(row_data)

            df = pd.DataFrame(df_data)

            # Add fitted curve if successful
            if not fitting_result.error_info:
                df["fitted_probability"] = self._rabi_oscillation_function(
                    df["amplitude"].values,
                    fitting_result.fit_amplitude,
                    fitting_result.frequency,
                    fitting_result.offset,
                )

                # Validate fitting quality
                self._validate_rabi_fitting_quality(fitting_result, result)

            # Store fitting result
            self.data.fitting_result = fitting_result

            # Generate plots with error handling
            if plot:
                try:
                    self._plot_rabi_results(df, fitting_result, save_image, plot)
                except Exception as e:
                    result.add_warning(f"Plotting failed: {str(e)}")
                    result.add_suggestion("Check plotting dependencies and data format")

            # Update result with successful data
            result.data = df
            result.metadata = {
                "experiment_type": "rabi",
                "num_amplitudes": len(amplitudes),
                "fitting_successful": not bool(fitting_result.error_info),
                "pi_amplitude": fitting_result.pi_amplitude,
                "rabi_frequency": fitting_result.frequency,
                "r_squared": fitting_result.r_squared,
            }

            # Return legacy DataFrame format for backward compatibility
            return result.to_legacy_dataframe()

        except InsufficientDataError as e:
            result = AnalysisResult.error_result(
                errors=[str(e)],
                suggestions=e.suggestions,
                metadata={"experiment_type": "rabi"},
            )
            return result.to_legacy_dataframe()

        except FittingError as e:
            result = AnalysisResult.error_result(
                errors=[str(e)],
                suggestions=e.suggestions,
                metadata={"experiment_type": "rabi"},
            )
            return result.to_legacy_dataframe()

        except Exception as e:
            # Fallback for unexpected errors
            result = AnalysisResult.error_result(
                errors=[f"Unexpected Rabi analysis error: {str(e)}"],
                suggestions=[
                    "Check input data format and types",
                    "Verify all required dependencies are installed",
                    "Report this error if it persists",
                ],
                metadata={"experiment_type": "rabi"},
            )
            return result.to_legacy_dataframe()

    def _validate_rabi_inputs(self, amplitudes, probabilities, prob_errors):
        """Validate Rabi experiment input data"""
        from ..models.analysis_result import AnalysisResult
        from ..utils.validation_helpers import (
            validate_fitting_data,
            validate_positive_values,
            validate_probability_values,
        )

        result = AnalysisResult.success_result(data=pd.DataFrame())

        try:
            # Validate amplitudes (should be positive)
            validate_positive_values(amplitudes, "amplitudes")

            # Validate probabilities
            validate_probability_values(probabilities)
            validate_probability_values(prob_errors, allow_zero=True)

            # Validate data consistency
            if len(amplitudes) != len(probabilities) or len(amplitudes) != len(
                prob_errors
            ):
                result.add_error(
                    "Data length mismatch between amplitudes, probabilities, and errors"
                )
                result.add_suggestion("Check data collection and processing pipeline")
                return result

            # Validate fitting data if enough points
            if len(amplitudes) >= 3:
                validate_fitting_data(amplitudes, probabilities, "Rabi")

        except Exception as e:
            result.add_error(str(e))
            if hasattr(e, "suggestions"):
                result.suggestions.extend(e.suggestions)

        return result

    def _validate_rabi_fitting_quality(self, fitting_result, analysis_result):
        """Validate quality of Rabi oscillation fitting"""
        # Check R-squared
        if fitting_result.r_squared < 0.7:
            analysis_result.add_warning(
                f"Poor Rabi fit quality: R² = {fitting_result.r_squared:.3f}"
            )
            analysis_result.add_suggestion(
                "Consider collecting more data points or checking drive calibration"
            )

        # Check π-pulse amplitude reasonableness
        if fitting_result.pi_amplitude > 10.0:
            analysis_result.add_warning(
                f"Unusually high π-pulse amplitude: {fitting_result.pi_amplitude:.3f}"
            )
            analysis_result.add_suggestion("Check drive power and calibration")

        # Check Rabi frequency bounds
        if fitting_result.frequency <= 0:
            analysis_result.add_warning(
                f"Invalid Rabi frequency: {fitting_result.frequency:.3f} <= 0"
            )
            analysis_result.add_suggestion(
                "Check fitting bounds and initial conditions"
            )

    def _rabi_oscillation_function(
        self, amplitudes: list[float], A: float, f: float, offset: float
    ) -> list[float]:
        """Rabi oscillation function: P(amp) = A * sin²(f * amp) + offset"""
        import numpy as np

        result = A * np.sin(f * np.array(amplitudes)) ** 2 + offset
        return result.tolist()  # type: ignore

    def _fit_rabi_oscillation(
        self,
        amplitudes: list[float],
        probabilities: list[float],
        prob_errors: list[float],
    ) -> RabiFittingResult:
        """Fit Rabi oscillation to measured data"""
        try:
            import numpy as np
            from scipy.optimize import curve_fit

            # Initial guess: A=0.5, f=π/max_amp, offset=0.5
            max_amp = max(amplitudes)
            initial_guess = [0.5, np.pi / max_amp, 0.5]
            bounds = ([0, 0, 0], [1, 10 * np.pi / max_amp, 1])  # Physical bounds

            def rabi_func(amp, A, f, offset):
                return A * np.sin(f * amp) ** 2 + offset

            popt, pcov = curve_fit(
                rabi_func,
                np.array(amplitudes),
                np.array(probabilities),
                p0=initial_guess,
                bounds=bounds,
                sigma=prob_errors,
                absolute_sigma=True,
            )

            A, f, offset = popt

            # Calculate π-pulse amplitude
            pi_amplitude = np.pi / (2 * f) if f > 0 else 0

            # Calculate R-squared
            y_pred = rabi_func(np.array(amplitudes), A, f, offset)
            ss_res = np.sum((np.array(probabilities) - y_pred) ** 2)
            ss_tot = np.sum((np.array(probabilities) - np.mean(probabilities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return RabiFittingResult(
                pi_amplitude=pi_amplitude,
                frequency=f,
                fit_amplitude=A,
                offset=offset,
                r_squared=r_squared,
                amplitudes=amplitudes,
                probabilities=probabilities,
            )

        except Exception as e:
            return RabiFittingResult(
                pi_amplitude=0.0,
                frequency=0.0,
                fit_amplitude=0.5,
                offset=0.5,
                r_squared=0.0,
                amplitudes=amplitudes,
                probabilities=probabilities,
                error_info=f"Rabi fitting failed: {str(e)}",
            )

    def _plot_rabi_results(
        self,
        df: pd.DataFrame,
        fitting_result: RabiFittingResult,
        save_image: bool = False,
        plot: bool = True,
    ):
        """Plot Rabi oscillation curve following plot_settings.md guidelines"""
        try:
            import plotly.graph_objects as go

            from ..utils.visualization import (
                apply_experiment_layout,
                get_experiment_colors,
                get_plotly_config,
                setup_plotly_environment,
                show_plotly_figure,
            )

            setup_plotly_environment()
            colors = get_experiment_colors()

            # Create figure with plot_settings.md specifications
            fig = go.Figure()

            # Plot data points with error bars (primary data in green - colors[1])
            fig.add_trace(
                go.Scatter(
                    x=df["amplitude"],
                    y=df["probability"],
                    error_y={"array": df["probability_error"]},
                    mode="markers",
                    name="Data",
                    marker={"color": colors[1], "size": 8},
                    line={"color": colors[1]},
                )
            )

            # Plot fitted curve if available (fit in blue - colors[0])
            if "fitted_probability" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["amplitude"],
                        y=df["fitted_probability"],
                        mode="lines",
                        name=f"Fit: π-amp={fitting_result.pi_amplitude:.3f}",
                        line={"color": colors[0], "width": 2},
                    )
                )

            # Apply layout using standard utility
            apply_experiment_layout(
                fig,
                title="Rabi Oscillation",
                xaxis_title="Drive Amplitude",
                yaxis_title="Excited State Probability",
                width=1000,
                height=500,
            )

            # Add statistics box if fitting succeeded
            if not fitting_result.error_info:
                stats_text = f"""π-pulse amplitude: {fitting_result.pi_amplitude:.4f}
Rabi frequency: {fitting_result.frequency:.4f}
R²: {fitting_result.r_squared:.4f}"""

                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=stats_text,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#CCCCCC",
                    borderwidth=1,
                    font={"size": 10},
                    align="left",
                    valign="top",
                )

            # Show plot - saving handled by experiment classes
            if save_image:
                # Plot saving handled by experiment classes via data manager
                pass

            config = get_plotly_config("rabi", width=1000, height=500)
            show_plotly_figure(fig, config)

        except ImportError:
            print("Plotly not available. Skipping plot.")
        except Exception as e:
            print(f"Rabi plotting failed: {e}")


class RabiCircuitParams(BaseModel):
    """Parameters for individual Rabi circuits"""

    experiment: str = Field(default="rabi", description="Experiment type")
    amplitude: float = Field(description="Drive amplitude")
    logical_qubit: int = Field(description="Logical qubit index")
    physical_qubit: int = Field(description="Physical qubit index")
    rotation_angle: float = Field(description="Rotation angle in radians")
