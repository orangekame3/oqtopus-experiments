#!/usr/bin/env python3
"""
Pydantic models for Ramsey experiment
Provides structured data validation and serialization
"""

from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .experiment_result import ExperimentResult

if TYPE_CHECKING:
    from ..models.analysis_result import AnalysisResult


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


class RamseyData(BaseModel):
    """Data structure for Ramsey measurements"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    delay_times: list[float] = Field(description="Delay time values in ns")
    probabilities: list[float] = Field(description="Measured probabilities")
    probability_errors: list[float] = Field(
        description="Standard errors in probabilities"
    )
    shots_per_point: int = Field(description="Number of shots per delay point")
    detuning_frequency: float = Field(description="Detuning frequency in Hz")
    fitting_result: RamseyFittingResult | None = Field(
        default=None, description="Ramsey fringe fitting results"
    )


class RamseyAnalysisResult(ExperimentResult):
    """Complete Ramsey experiment result with comprehensive error handling"""

    def __init__(self, data: RamseyData, **kwargs):
        """Initialize with Ramsey data"""
        super().__init__(**kwargs)
        self.data = data

    def analyze(
        self,
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze Ramsey fringe results with comprehensive error handling

        Args:
            plot: Whether to generate plots using plot_settings.md guidelines
            save_data: Whether to save analysis results (handled by experiment classes)
            save_image: Whether to save generated plots

        Returns:
            DataFrame with analysis results including fitted T2* and frequency parameters

        Note:
            Data saving is handled by experiment classes via data manager.
            This method performs data analysis with comprehensive error handling
            and validation to provide helpful feedback when issues occur.
        """
        from ..models.analysis_result import AnalysisResult

        try:
            # Extract measurement data
            data = self.data
            delay_times = data.delay_times
            probabilities = data.probabilities
            prob_errors = data.probability_errors
            detuning_freq = data.detuning_frequency

            # Comprehensive input validation
            result = self._validate_ramsey_inputs(
                delay_times, probabilities, prob_errors
            )
            if not result.success:
                return result.to_legacy_dataframe()

            # Check if we have enough data for fitting
            if len(delay_times) < 4:
                # Create result with insufficient data
                fitting_result = RamseyFittingResult(
                    t2_star_time=0.0,
                    frequency=detuning_freq,
                    amplitude=0.5,
                    offset=0.5,
                    phase=0.0,
                    r_squared=0.0,
                    delay_times=delay_times,
                    probabilities=probabilities,
                    error_info="Insufficient data points for fitting (need at least 4)",
                )
                result = AnalysisResult(
                    success=False,
                    error_message="Insufficient data for Ramsey fitting",
                    error_type="InsufficientDataError",
                    data={"fitting_result": fitting_result},
                )
                return result.to_legacy_dataframe()

            # Attempt curve fitting
            try:
                fitting_result = self._fit_ramsey_fringes(
                    delay_times, probabilities, prob_errors, detuning_freq
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
                        self._create_ramsey_plot(fitting_result, save_image)
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
                    "t2_star_time": fitting_result.t2_star_time,
                    "frequency": fitting_result.frequency,
                    "amplitude": fitting_result.amplitude,
                    "offset": fitting_result.offset,
                    "phase": fitting_result.phase,
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
                fitting_result = RamseyFittingResult(
                    t2_star_time=0.0,
                    frequency=detuning_freq,
                    amplitude=0.5,
                    offset=0.5,
                    phase=0.0,
                    r_squared=0.0,
                    delay_times=delay_times,
                    probabilities=probabilities,
                    error_info=f"Fitting failed: {str(e)}",
                )
                result = AnalysisResult(
                    success=False,
                    error_message=f"Ramsey fitting failed: {str(e)}",
                    error_type="FittingError",
                    data={"fitting_result": fitting_result},
                    suggestions=[
                        "Check if delay times span sufficient range for fringe observation",
                        "Verify detuning frequency is appropriate",
                        "Consider using more delay points for better fitting",
                    ],
                )
                return result.to_legacy_dataframe()

        except Exception as e:
            # Handle unexpected errors
            result = AnalysisResult(
                success=False,
                error_message=f"Unexpected error in Ramsey analysis: {str(e)}",
                error_type="UnexpectedError",
                suggestions=[
                    "Check input data format and types",
                    "Verify all required dependencies are available",
                ],
            )
            return result.to_legacy_dataframe()

    def _validate_ramsey_inputs(
        self,
        delay_times: list[float],
        probabilities: list[float],
        prob_errors: list[float],
    ) -> "AnalysisResult":
        """Validate Ramsey input data"""
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

        # Ramsey specific validations
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

    def _fit_ramsey_fringes(
        self,
        delay_times: list[float],
        probabilities: list[float],
        prob_errors: list[float],
        detuning_frequency: float,
    ) -> RamseyFittingResult:
        """Fit damped oscillation to Ramsey data"""
        import numpy as np
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score

        def ramsey_func(t, amplitude, t2_star, offset, phase, frequency):
            """Ramsey fringe function: P = amplitude * exp(-t/T2*) * cos(2πft + φ) + offset"""
            return (
                amplitude
                * np.exp(-np.array(t) / t2_star)
                * np.cos(2 * np.pi * frequency * np.array(t) + phase)
                + offset
            )

        x_data = np.array(delay_times)
        y_data = np.array(probabilities)
        y_errors = np.array(prob_errors) if prob_errors else None

        # Initial parameter guesses
        amplitude_guess = (max(y_data) - min(y_data)) / 2
        offset_guess = np.mean(y_data)
        t2_star_guess = max(x_data) / 2  # Rough estimate
        phase_guess = 0.0
        freq_guess = (
            detuning_frequency if detuning_frequency != 0 else 1e6
        )  # 1 MHz default

        # Parameter bounds
        bounds = (
            [0.0, max(x_data) * 0.01, 0.0, -2 * np.pi, 0.0],  # Lower bounds
            [1.0, max(x_data) * 10, 1.0, 2 * np.pi, 1e9],  # Upper bounds
        )

        # Perform fitting
        popt, pcov = curve_fit(
            ramsey_func,
            x_data,
            y_data,
            p0=[amplitude_guess, t2_star_guess, offset_guess, phase_guess, freq_guess],
            sigma=y_errors,
            absolute_sigma=True if y_errors is not None else False,
            bounds=bounds,
            maxfev=5000,
        )

        amplitude_fit, t2_star_fit, offset_fit, phase_fit, frequency_fit = popt

        # Calculate R-squared
        y_pred = ramsey_func(x_data, *popt)
        r_squared = r2_score(y_data, y_pred)

        return RamseyFittingResult(
            t2_star_time=float(t2_star_fit),
            frequency=float(frequency_fit),
            amplitude=float(amplitude_fit),
            offset=float(offset_fit),
            phase=float(phase_fit),
            r_squared=float(r_squared),
            delay_times=delay_times,
            probabilities=probabilities,
            error_info=None,
        )

    def _validate_fitting_quality(
        self, fitting_result: RamseyFittingResult
    ) -> "AnalysisResult":
        """Validate quality of Ramsey fitting results"""
        import numpy as np

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

        # Check T2* time reasonableness
        max_delay = max(fitting_result.delay_times)
        if fitting_result.t2_star_time > max_delay * 5:
            warnings.append(
                f"T2* time ({fitting_result.t2_star_time:.1f} ns) much larger than measurement range"
            )
        elif fitting_result.t2_star_time < max_delay * 0.1:
            warnings.append(
                f"T2* time ({fitting_result.t2_star_time:.1f} ns) much smaller than measurement range"
            )

        # Check amplitude
        if fitting_result.amplitude < 0.05:
            warnings.append(f"Small fitted amplitude ({fitting_result.amplitude:.3f})")

        # Check frequency reasonableness
        if fitting_result.frequency < 0:
            warnings.append("Negative frequency fitted")
        elif fitting_result.frequency > 1e9:
            warnings.append(
                f"Very high frequency ({fitting_result.frequency / 1e6:.1f} MHz)"
            )

        # Check phase range
        if abs(fitting_result.phase) > 2 * np.pi:
            warnings.append(
                f"Phase outside typical range: {fitting_result.phase:.2f} rad"
            )

        if issues:
            return AnalysisResult(
                success=False,
                error_message="Ramsey fitting quality issues: " + "; ".join(issues),
                error_type="FittingQualityError",
                warnings=warnings,
                suggestions=[
                    "Increase measurement range to better capture fringe oscillations",
                    "Use more delay points for better statistics",
                    "Check detuning frequency setting",
                ],
            )

        return AnalysisResult(
            success=True, warnings=warnings, data={"quality_check": "passed"}
        )

    def _calculate_fitted_probabilities(
        self, fitting_result: RamseyFittingResult
    ) -> list[float]:
        """Calculate fitted probabilities from Ramsey parameters"""
        import numpy as np

        def ramsey_func(t, amplitude, t2_star, offset, phase, frequency):
            return (
                amplitude
                * np.exp(-np.array(t) / t2_star)
                * np.cos(2 * np.pi * frequency * np.array(t) + phase)
                + offset
            )

        fitted_probs = ramsey_func(
            fitting_result.delay_times,
            fitting_result.amplitude,
            fitting_result.t2_star_time,
            fitting_result.offset,
            fitting_result.phase,
            fitting_result.frequency,
        )
        return fitted_probs.tolist()

    def _create_ramsey_plot(
        self, fitting_result: RamseyFittingResult, save_image: bool = True
    ):
        """Create Ramsey fringe plot following plot_settings.md guidelines"""
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
                min(fitting_result.delay_times), max(fitting_result.delay_times), 500
            )
            fitted_fine = (
                fitting_result.amplitude
                * np.exp(-delay_fine / fitting_result.t2_star_time)
                * np.cos(
                    2 * np.pi * fitting_result.frequency * delay_fine
                    + fitting_result.phase
                )
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
                title=f"Ramsey Fringes (T2* = {fitting_result.t2_star_time:.1f} ns, f = {fitting_result.frequency / 1e6:.2f} MHz)",
                xaxis_title="Delay Time (ns)",
                yaxis_title="P(|0⟩)",
                width=1000,
                height=500,
            )

            # Add fitting results annotation
            annotation_text = (
                f"T2* = {fitting_result.t2_star_time:.1f} ns<br>"
                f"Frequency = {fitting_result.frequency / 1e6:.2f} MHz<br>"
                f"Amplitude = {fitting_result.amplitude:.3f}<br>"
                f"Phase = {fitting_result.phase:.2f} rad<br>"
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
                save_plotly_figure(fig, name="ramsey_fringes", images_dir="./plots")

            config = get_plotly_config("ramsey_fringes", 1000, 500)
            show_plotly_figure(fig, config)

        except Exception as e:
            print(f"Warning: Ramsey plot generation failed: {e}")


class RamseyCircuitParams(BaseModel):
    """Parameters for individual Ramsey circuits"""

    experiment: str = Field(default="ramsey", description="Experiment type")
    delay_time: float = Field(description="Delay time in ns")
    detuning_frequency: float = Field(description="Detuning frequency in Hz")
    logical_qubit: int = Field(description="Logical qubit index")
    physical_qubit: int = Field(description="Physical qubit index")
