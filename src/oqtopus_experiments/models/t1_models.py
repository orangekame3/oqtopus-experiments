#!/usr/bin/env python3
"""
Pydantic models for T1 experiment
Provides structured data validation and serialization
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .experiment_result import ExperimentResult


class T1FittingResult(BaseModel):
    """Fitting results for T1 decay"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    t1_time: float = Field(description="T1 relaxation time in ns")
    amplitude: float = Field(description="Fitted amplitude")
    offset: float = Field(description="Fitted offset")
    r_squared: float = Field(description="R-squared goodness of fit")
    delay_times: list[float] = Field(description="Input delay time values")
    probabilities: list[float] = Field(description="Measured probabilities")
    error_info: str | None = Field(
        default=None, description="Error information if fitting failed"
    )


class T1Parameters(BaseModel):
    """T1 experiment parameters"""

    experiment_name: str | None = Field(default=None, description="Experiment name")
    physical_qubit: int = Field(default=0, description="Target physical qubit")
    delay_points: int = Field(default=20, description="Number of delay points")
    max_delay: float = Field(default=50000.0, description="Maximum delay time in ns")


class T1Data(BaseModel):
    """Data structure for T1 measurements"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    delay_times: list[float] = Field(description="Delay times in ns")
    probabilities: list[float] = Field(description="Measured probabilities")
    probability_errors: list[float] = Field(
        description="Standard errors in probabilities"
    )
    shots_per_point: int = Field(description="Number of shots per delay point")
    fitting_result: T1FittingResult | None = Field(
        default=None, description="T1 decay fitting results"
    )


class T1AnalysisResult(ExperimentResult):
    """Complete T1 experiment result with comprehensive error handling"""

    def __init__(self, data: T1Data, **kwargs):
        """Initialize with T1 data"""
        super().__init__(**kwargs)
        self.data = data

    def analyze(
        self,
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze T1 decay results with comprehensive error handling

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
            delay_times = data.delay_times
            probabilities = data.probabilities
            prob_errors = data.probability_errors

            # Comprehensive input validation
            result = self._validate_t1_inputs(delay_times, probabilities, prob_errors)
            if not result.success:
                return result.to_legacy_dataframe()

            # Check if we have enough data for fitting
            if len(delay_times) < 3:
                # Create result with insufficient data
                fitting_result = T1FittingResult(
                    t1_time=0.0,
                    amplitude=0.5,
                    offset=0.5,
                    r_squared=0.0,
                    delay_times=delay_times,
                    probabilities=probabilities,
                    error_info="Insufficient data points for fitting (need at least 3)",
                )
                result.add_warning("Insufficient data for T1 decay fitting")
                result.add_suggestion(
                    "Collect more delay time points for reliable T1 fitting"
                )
            else:
                # Attempt curve fitting with detailed error handling
                try:
                    fitting_result = self._fit_t1_decay(
                        delay_times, probabilities, prob_errors
                    )
                    if fitting_result.error_info:
                        result.add_warning(
                            f"T1 fitting issues: {fitting_result.error_info}"
                        )
                        result.add_suggestion(
                            "Check data quality and T1 measurement setup"
                        )
                except Exception as e:
                    # Create failed fitting result
                    fitting_result = T1FittingResult(
                        t1_time=0.0,
                        amplitude=0.5,
                        offset=0.5,
                        r_squared=0.0,
                        delay_times=delay_times,
                        probabilities=probabilities,
                        error_info=f"T1 fitting failed: {str(e)}",
                    )
                    result.add_error(f"T1 decay curve fitting failed: {str(e)}")
                    result.add_suggestion(
                        "Check if data shows expected exponential decay pattern"
                    )
                    result.add_suggestion("Verify measurement timing and calibration")

            # Create DataFrame with comprehensive data
            df_data = []
            for i, delay_time in enumerate(delay_times):
                row_data = {
                    "delay_time_ns": delay_time,
                    "probability": probabilities[i],
                    "probability_error": prob_errors[i],
                    "shots": data.shots_per_point,
                }

                # Add quality indicators
                if prob_errors[i] > 0.1:  # High uncertainty
                    result.add_warning(
                        f"High uncertainty at delay {delay_time}ns: σ={prob_errors[i]:.3f}"
                    )

                df_data.append(row_data)

            df = pd.DataFrame(df_data)

            # Add fitted curve if successful
            if not fitting_result.error_info:
                df["fitted_probability"] = self._t1_decay_function(
                    df["delay_time_ns"].values,
                    fitting_result.amplitude,
                    fitting_result.t1_time,
                    fitting_result.offset,
                )

                # Validate fitting quality
                self._validate_t1_fitting_quality(fitting_result, result)

            # Store fitting result
            self.data.fitting_result = fitting_result

            # Generate plots with error handling
            if plot:
                try:
                    self._plot_t1_results(df, fitting_result, save_image, plot)
                except Exception as e:
                    result.add_warning(f"Plotting failed: {str(e)}")
                    result.add_suggestion("Check plotting dependencies and data format")

            # Update result with successful data
            result.data = df
            result.metadata = {
                "experiment_type": "t1",
                "num_delay_points": len(delay_times),
                "fitting_successful": not bool(fitting_result.error_info),
                "t1_time_ns": fitting_result.t1_time,
                "r_squared": fitting_result.r_squared,
            }

            # Return legacy DataFrame format for backward compatibility
            return result.to_legacy_dataframe()

        except InsufficientDataError as e:
            result = AnalysisResult.error_result(
                errors=[str(e)],
                suggestions=e.suggestions,
                metadata={"experiment_type": "t1"},
            )
            return result.to_legacy_dataframe()

        except FittingError as e:
            result = AnalysisResult.error_result(
                errors=[str(e)],
                suggestions=e.suggestions,
                metadata={"experiment_type": "t1"},
            )
            return result.to_legacy_dataframe()

        except Exception as e:
            # Fallback for unexpected errors
            result = AnalysisResult.error_result(
                errors=[f"Unexpected T1 analysis error: {str(e)}"],
                suggestions=[
                    "Check input data format and types",
                    "Verify all required dependencies are installed",
                    "Report this error if it persists",
                ],
                metadata={"experiment_type": "t1"},
            )
            return result.to_legacy_dataframe()

    def _validate_t1_inputs(self, delay_times, probabilities, prob_errors):
        """Validate T1 experiment input data"""
        from ..models.analysis_result import AnalysisResult
        from ..utils.validation_helpers import (
            validate_probability_values,
        )

        result = AnalysisResult.success_result(data=pd.DataFrame())

        try:
            # Validate delay times (should be non-negative for T1)
            if any(t < 0 for t in delay_times):
                result.add_error("Delay times must be non-negative")
                result.add_suggestion("Check delay time values and units")
                return result

            # Validate probabilities
            validate_probability_values(probabilities)
            validate_probability_values(prob_errors, allow_zero=True)

            # Validate data consistency
            if len(delay_times) != len(probabilities) or len(delay_times) != len(
                prob_errors
            ):
                result.add_error(
                    "Data length mismatch between delay times, probabilities, and errors"
                )
                result.add_suggestion("Check data collection and processing pipeline")
                return result

            # Validate fitting data if enough points (custom for T1 since delay times can include 0)
            if len(delay_times) >= 3:
                # Check data consistency without using validate_fitting_data (which requires positive x values)
                if len(set(delay_times)) < len(delay_times):
                    result.add_error("Duplicate delay time values found")
                    result.add_suggestion("Ensure each delay time is unique")
                    return result

                delay_range = max(delay_times) - min(delay_times)
                if delay_range == 0:
                    result.add_error("No variation in delay times")
                    result.add_suggestion("Use different delay time values")
                    return result

        except Exception as e:
            result.add_error(str(e))
            if hasattr(e, "suggestions"):
                result.suggestions.extend(e.suggestions)

        return result

    def _validate_t1_fitting_quality(self, fitting_result, analysis_result):
        """Validate quality of T1 decay fitting"""
        # Check R-squared
        if fitting_result.r_squared < 0.8:
            analysis_result.add_warning(
                f"Poor T1 fit quality: R² = {fitting_result.r_squared:.3f}"
            )
            analysis_result.add_suggestion(
                "Consider collecting more data points or checking measurement setup"
            )

        # Check T1 time reasonableness (should be positive and reasonable)
        if fitting_result.t1_time <= 0:
            analysis_result.add_warning(
                f"Invalid T1 time: {fitting_result.t1_time:.1f} ns <= 0"
            )
            analysis_result.add_suggestion(
                "Check fitting bounds and initial conditions"
            )

        if fitting_result.t1_time > 1e6:  # > 1 ms seems unusually long
            analysis_result.add_warning(
                f"Unusually long T1 time: {fitting_result.t1_time:.1f} ns"
            )
            analysis_result.add_suggestion("Verify measurement setup and environment")

    def _t1_decay_function(
        self, delay_times: list[float], A: float, T1: float, offset: float
    ) -> list[float]:
        """T1 decay function: P(t) = A * exp(-t/T1) + offset"""
        import numpy as np

        result = A * np.exp(-np.array(delay_times) / T1) + offset
        return result.tolist()  # type: ignore

    def _fit_t1_decay(
        self,
        delay_times: list[float],
        probabilities: list[float],
        prob_errors: list[float],
    ) -> T1FittingResult:
        """Fit T1 exponential decay to measured data"""
        try:
            import numpy as np
            from scipy.optimize import curve_fit

            # Initial guess: A=0.5, T1=max_delay/3, offset=0.5
            max_delay = max(delay_times)
            initial_guess = [0.5, max_delay / 3, 0.5]
            bounds = ([0, 0, 0], [1, max_delay * 10, 1])  # Physical bounds

            def t1_func(t, A, T1, offset):
                return A * np.exp(-t / T1) + offset

            popt, pcov = curve_fit(
                t1_func,
                np.array(delay_times),
                np.array(probabilities),
                p0=initial_guess,
                bounds=bounds,
                sigma=prob_errors,
                absolute_sigma=True,
            )

            A, T1, offset = popt

            # Calculate R-squared
            y_pred = t1_func(np.array(delay_times), A, T1, offset)
            ss_res = np.sum((np.array(probabilities) - y_pred) ** 2)
            ss_tot = np.sum((np.array(probabilities) - np.mean(probabilities)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return T1FittingResult(
                t1_time=T1,
                amplitude=A,
                offset=offset,
                r_squared=r_squared,
                delay_times=delay_times,
                probabilities=probabilities,
            )

        except Exception as e:
            return T1FittingResult(
                t1_time=0.0,
                amplitude=0.5,
                offset=0.5,
                r_squared=0.0,
                delay_times=delay_times,
                probabilities=probabilities,
                error_info=f"T1 fitting failed: {str(e)}",
            )

    def _plot_t1_results(
        self,
        df: pd.DataFrame,
        fitting_result: T1FittingResult,
        save_image: bool = False,
        plot: bool = True,
    ):
        """Plot T1 decay curve following plot_settings.md guidelines"""
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
                    x=df["delay_time_ns"],
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
                        x=df["delay_time_ns"],
                        y=df["fitted_probability"],
                        mode="lines",
                        name=f"Fit: T1={fitting_result.t1_time:.1f}ns",
                        line={"color": colors[0], "width": 2},
                    )
                )

            # Apply layout using standard utility
            apply_experiment_layout(
                fig,
                title="T1 Relaxation",
                xaxis_title="Delay Time (ns)",
                yaxis_title="Excited State Probability",
                width=1000,
                height=500,
            )

            # Add statistics box if fitting succeeded
            if not fitting_result.error_info:
                stats_text = f"""T1 time: {fitting_result.t1_time:.1f} ns
Amplitude: {fitting_result.amplitude:.4f}
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

            config = get_plotly_config("t1", width=1000, height=500)
            show_plotly_figure(fig, config)

        except ImportError:
            print("Plotly not available. Skipping plot.")
        except Exception as e:
            print(f"T1 plotting failed: {e}")


class T1CircuitParams(BaseModel):
    """Parameters for individual T1 circuits"""

    experiment: str = Field(default="t1", description="Experiment type")
    delay_time: float = Field(description="Delay time in ns")
    logical_qubit: int = Field(description="Logical qubit index")
    physical_qubit: int = Field(description="Physical qubit index")
