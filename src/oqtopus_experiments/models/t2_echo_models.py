#!/usr/bin/env python3
"""
Pydantic models for T2 Echo experiment
Provides structured data validation and serialization
"""

from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .experiment_result import ExperimentResult

if TYPE_CHECKING:
    from ..models.analysis_result import AnalysisResult


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


class T2EchoData(BaseModel):
    """Data structure for T2 Echo measurements"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    delay_times: list[float] = Field(description="Delay time values in ns")
    probabilities: list[float] = Field(description="Measured probabilities")
    probability_errors: list[float] = Field(
        description="Standard errors in probabilities"
    )
    shots_per_point: int = Field(description="Number of shots per delay point")
    fitting_result: T2EchoFittingResult | None = Field(
        default=None, description="T2 Echo decay fitting results"
    )


class T2EchoAnalysisResult(ExperimentResult):
    """Complete T2 Echo experiment result with comprehensive error handling"""

    def __init__(self, data: T2EchoData, **kwargs):
        """Initialize with T2 Echo data"""
        super().__init__(**kwargs)
        self.data = data

    def analyze(
        self,
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze T2 Echo decay results with comprehensive error handling

        Args:
            plot: Whether to generate plots using plot_settings.md guidelines
            save_data: Whether to save analysis results (handled by experiment classes)
            save_image: Whether to save generated plots

        Returns:
            DataFrame with analysis results including fitted T2 parameters

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
            result = self._validate_t2_echo_inputs(delay_times, probabilities, prob_errors)
            if not result.success:
                return result.to_legacy_dataframe()

            # Check if we have enough data for fitting
            if len(delay_times) < 3:
                # Create result with insufficient data
                fitting_result = T2EchoFittingResult(
                    t2_time=0.0,
                    amplitude=0.5,
                    offset=0.5,
                    r_squared=0.0,
                    delay_times=delay_times,
                    probabilities=probabilities,
                    error_info="Insufficient data points for fitting (need at least 3)",
                )
                result.add_warning("Insufficient data for T2 Echo fitting")
                result.add_suggestion(
                    "Collect more delay time points for reliable T2 fitting"
                )

            # Attempt curve fitting
            try:
                fitting_result = self._fit_t2_echo_decay(
                    delay_times, probabilities, prob_errors
                )

                # Validate fitting results quality
                quality_result = self._validate_fitting_quality(fitting_result)
                if not quality_result.success:
                    return quality_result.to_legacy_dataframe()

                # Update data with fitting results
                self.data.fitting_result = fitting_result

                # Generate plots if requested
                if plot:
                    try:
                        self._create_t2_echo_plot(fitting_result, save_image)
                    except Exception as e:
                        print(f"Warning: Plot generation failed: {e}")

                # Create successful result
                analysis_data = {
                    "delay_times": delay_times,
                    "probabilities": probabilities,
                    "probability_errors": prob_errors,
                    "fitted_probabilities": self._calculate_fitted_probabilities(
                        fitting_result
                    ),
                    "t2_time": fitting_result.t2_time,
                    "amplitude": fitting_result.amplitude,
                    "offset": fitting_result.offset,
                    "r_squared": fitting_result.r_squared,
                }

                result = AnalysisResult(
                    success=True,
                    data=analysis_data,
                    metadata={"fitting_quality": "good", "error_info": None},
                )
                return result.to_legacy_dataframe()

            except Exception as e:
                # Handle fitting failure gracefully
                fitting_result = T2EchoFittingResult(
                    t2_time=0.0,
                    amplitude=0.5,
                    offset=0.5,
                    r_squared=0.0,
                    delay_times=delay_times,
                    probabilities=probabilities,
                    error_info=f"Fitting failed: {str(e)}",
                )
                result = AnalysisResult(
                    success=False,
                    error_message=f"T2 Echo fitting failed: {str(e)}",
                    error_type="FittingError",
                    data={"fitting_result": fitting_result},
                    suggestions=[
                        "Check if delay times span sufficient range for T2 measurement",
                        "Verify measurement data quality and noise levels",
                        "Consider using more delay points for better fitting",
                    ],
                )
                return result.to_legacy_dataframe()

        except Exception as e:
            # Handle unexpected errors
            result = AnalysisResult(
                success=False,
                error_message=f"Unexpected error in T2 Echo analysis: {str(e)}",
                error_type="UnexpectedError",
                suggestions=[
                    "Check input data format and types",
                    "Verify all required dependencies are available",
                ],
            )
            return result.to_legacy_dataframe()

    def _validate_t2_echo_inputs(
        self,
        delay_times: list[float],
        probabilities: list[float],
        prob_errors: list[float],
    ) -> "AnalysisResult":
        """Validate T2 Echo input data"""
        from ..models.analysis_result import AnalysisResult
        from ..utils.validation_helpers import (
            check_for_nan_inf,
            validate_data_ranges,
            validate_measurement_data,
        )

        # Basic data validation
        validation_result = validate_measurement_data(
            x_data=delay_times,
            y_data=probabilities,
            y_errors=prob_errors,
            x_name="delay_times",
            y_name="probabilities",
        )
        if not validation_result.success:
            return validation_result

        # Check for NaN/inf values
        nan_result = check_for_nan_inf(
            {
                "delay_times": delay_times,
                "probabilities": probabilities,
                "errors": prob_errors,
            }
        )
        if not nan_result.success:
            return nan_result

        # Validate ranges
        range_result = validate_data_ranges(
            delay_times=delay_times,
            probabilities=probabilities,
            prob_errors=prob_errors,
        )
        if not range_result.success:
            return range_result

        # T2 Echo specific validations
        if min(delay_times) < 0:
            return AnalysisResult(
                success=False,
                error_message="Delay times must be non-negative",
                error_type="ValidationError",
            )

        if max(delay_times) <= min(delay_times):
            return AnalysisResult(
                success=False,
                error_message="Delay times must span a range (max > min)",
                error_type="ValidationError",
            )

        return AnalysisResult(success=True, data={"validation": "passed"})

    def _fit_t2_echo_decay(
        self,
        delay_times: list[float],
        probabilities: list[float],
        prob_errors: list[float],
    ) -> T2EchoFittingResult:
        """Fit exponential decay to T2 Echo data"""
        import numpy as np
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score

        def t2_echo_func(t, amplitude, t2_time, offset):
            """T2 Echo decay function: P = amplitude * exp(-t/T2) + offset"""
            return amplitude * np.exp(-np.array(t) / t2_time) + offset

        x_data = np.array(delay_times)
        y_data = np.array(probabilities)
        y_errors = np.array(prob_errors) if prob_errors else None

        # Initial parameter guesses
        amplitude_guess = max(y_data) - min(y_data)
        offset_guess = min(y_data)
        t2_guess = max(x_data) / 3  # Rough estimate

        # Parameter bounds
        bounds = (
            [0.0, max(x_data) * 0.01, 0.0],  # Lower bounds
            [1.0, max(x_data) * 10, 1.0],  # Upper bounds
        )

        # Perform fitting
        popt, pcov = curve_fit(
            t2_echo_func,
            x_data,
            y_data,
            p0=[amplitude_guess, t2_guess, offset_guess],
            sigma=y_errors,
            absolute_sigma=True if y_errors is not None else False,
            bounds=bounds,
            maxfev=2000,
        )

        amplitude_fit, t2_time_fit, offset_fit = popt

        # Calculate R-squared
        y_pred = t2_echo_func(x_data, *popt)
        r_squared = r2_score(y_data, y_pred)

        return T2EchoFittingResult(
            t2_time=float(t2_time_fit),
            amplitude=float(amplitude_fit),
            offset=float(offset_fit),
            r_squared=float(r_squared),
            delay_times=delay_times,
            probabilities=probabilities,
            error_info=None,
        )

    def _validate_fitting_quality(
        self, fitting_result: T2EchoFittingResult
    ) -> "AnalysisResult":
        """Validate quality of T2 Echo fitting results"""
        from ..models.analysis_result import AnalysisResult

        issues = []
        warnings = []

        # Check R-squared
        if fitting_result.r_squared < 0.5:
            issues.append(
                f"Poor fit quality (R² = {fitting_result.r_squared:.3f} < 0.5)"
            )
        elif fitting_result.r_squared < 0.8:
            warnings.append(
                f"Moderate fit quality (R² = {fitting_result.r_squared:.3f})"
            )

        # Check T2 time reasonableness
        max_delay = max(fitting_result.delay_times)
        if fitting_result.t2_time > max_delay * 5:
            warnings.append(
                f"T2 time ({fitting_result.t2_time:.1f} ns) much larger than measurement range"
            )
        elif fitting_result.t2_time < max_delay * 0.1:
            warnings.append(
                f"T2 time ({fitting_result.t2_time:.1f} ns) much smaller than measurement range"
            )

        # Check amplitude
        if fitting_result.amplitude < 0.05:
            warnings.append(f"Small fitted amplitude ({fitting_result.amplitude:.3f})")

        if issues:
            return AnalysisResult(
                success=False,
                error_message="T2 Echo fitting quality issues: " + "; ".join(issues),
                error_type="FittingQualityError",
                warnings=warnings,
                suggestions=[
                    "Increase measurement range to better capture T2 decay",
                    "Use more delay points for better statistics",
                    "Check for systematic errors in measurements",
                ],
            )

        return AnalysisResult(
            success=True, warnings=warnings, data={"quality_check": "passed"}
        )

    def _calculate_fitted_probabilities(
        self, fitting_result: T2EchoFittingResult
    ) -> list[float]:
        """Calculate fitted probabilities from T2 Echo parameters"""
        import numpy as np

        def t2_echo_func(t, amplitude, t2_time, offset):
            return amplitude * np.exp(-np.array(t) / t2_time) + offset

        fitted_probs = t2_echo_func(
            fitting_result.delay_times,
            fitting_result.amplitude,
            fitting_result.t2_time,
            fitting_result.offset,
        )
        return fitted_probs.tolist()

    def _create_t2_echo_plot(
        self, fitting_result: T2EchoFittingResult, save_image: bool = True
    ):
        """Create T2 Echo decay plot following plot_settings.md guidelines"""
        try:
            import numpy as np
            import plotly.graph_objects as go

            from ..utils.visualization import (
                apply_experiment_layout,
                get_experiment_colors,
                get_plotly_config,
                save_plotly_figure,
                setup_plotly_environment,
                show_plotly_figure,
            )

            # Setup plotly environment
            setup_plotly_environment()
            colors = get_experiment_colors()

            # Create figure
            fig = go.Figure()

            # Data points with error bars
            fig.add_trace(
                go.Scatter(
                    x=fitting_result.delay_times,
                    y=fitting_result.probabilities,
                    mode="markers",
                    name="Measured Data",
                    marker={"color": colors[1], "size": 8},
                    error_y={
                        "type": "data",
                        "array": [0.02]
                        * len(fitting_result.probabilities),  # Default error
                        "visible": True,
                        "color": colors[1],
                    },
                )
            )

            # Fitted curve
            delay_fine = np.linspace(
                min(fitting_result.delay_times), max(fitting_result.delay_times), 200
            )
            fitted_fine = (
                fitting_result.amplitude * np.exp(-delay_fine / fitting_result.t2_time)
                + fitting_result.offset
            )

            fig.add_trace(
                go.Scatter(
                    x=delay_fine,
                    y=fitted_fine,
                    mode="lines",
                    name="Fitted Curve",
                    line={"color": colors[0], "width": 2},
                )
            )

            # Apply standard layout
            apply_experiment_layout(
                fig=fig,
                title=f"T2 Echo Decay (T2 = {fitting_result.t2_time:.1f} ns)",
                xaxis_title="Delay Time (ns)",
                yaxis_title="P(|0⟩)",
                width=1000,
                height=500,
            )

            # Add fitting results annotation
            annotation_text = (
                f"T2 = {fitting_result.t2_time:.1f} ns<br>"
                f"Amplitude = {fitting_result.amplitude:.3f}<br>"
                f"Offset = {fitting_result.offset:.3f}<br>"
                f"R² = {fitting_result.r_squared:.3f}"
            )

            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=annotation_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#CCCCCC",
                borderwidth=1,
            )

            # Save and show plot
            if save_image:
                save_plotly_figure(fig, name="t2_echo_decay", images_dir="./plots")

            config = get_plotly_config("t2_echo_decay", 1000, 500)
            show_plotly_figure(fig, config)

        except Exception as e:
            print(f"Warning: T2 Echo plot generation failed: {e}")


class T2EchoCircuitParams(BaseModel):
    """Parameters for individual T2 Echo circuits"""

    experiment: str = Field(default="t2_echo", description="Experiment type")
    delay_time: float = Field(description="Delay time in ns")
    logical_qubit: int = Field(description="Logical qubit index")
    physical_qubit: int = Field(description="Physical qubit index")
