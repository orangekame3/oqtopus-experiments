#!/usr/bin/env python3
"""
Hadamard Test experiment models for parameter validation and result storage.
"""

from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

from .experiment_result import ExperimentResult

if TYPE_CHECKING:
    from ..models.analysis_result import AnalysisResult


class HadamardTestParameters(BaseModel):
    """Parameters for Hadamard Test experiment

    This implementation supports X, Y, Z unitaries with optimal initial state preparation:
    - X unitary: RY(θ) initial state → ⟨ψ(θ)|X|ψ(θ)⟩ = sin(θ)
    - Y unitary: RX(θ) initial state → ⟨ψ(θ)|Y|ψ(θ)⟩ = -sin(θ) (Qiskit RX convention)
    - Z unitary: RX(θ) initial state → ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ) (default)
    """

    experiment_name: str | None = Field(
        default=None, description="Name of the experiment"
    )
    physical_qubit: int = Field(default=0, description="Physical qubit to use")
    test_unitary: str = Field(default="Z", description="Test unitary gate (X, Y, or Z)")
    angle_points: int = Field(
        default=16, ge=4, description="Number of angle points to measure"
    )
    max_angle: float = Field(
        default=6.28318530718, gt=0, description="Maximum angle in radians (2π)"
    )


class HadamardTestCircuitParams(BaseModel):
    """Circuit parameters for individual Hadamard Test measurement"""

    angle: float = Field(description="Rotation angle parameter")
    test_unitary: str = Field(description="Type of unitary gate")
    logical_qubit: int = Field(default=0, description="Logical qubit index")
    physical_qubit: int = Field(default=0, description="Physical qubit index")


class HadamardTestFittingResult(BaseModel):
    """Results from Hadamard Test fitting analysis"""

    real_part: float = Field(description="Real part of expectation value")
    imaginary_part: float = Field(description="Imaginary part of expectation value")
    magnitude: float = Field(description="Magnitude of complex expectation value")
    phase: float = Field(description="Phase of complex expectation value")
    phase_degrees: float = Field(description="Phase in degrees")
    fidelity: float = Field(ge=0, le=1, description="Quality metric for the fit (0-1)")
    angles: list[float] = Field(description="Angle values used in fitting")
    probabilities: list[float] = Field(description="Measured probabilities")
    r_squared: float = Field(
        default=0.0, ge=0, le=1, description="R-squared goodness of fit"
    )
    error_info: str | None = Field(
        default=None, description="Error information if fitting failed"
    )


class HadamardTestData(BaseModel):
    """Data structure for Hadamard Test measurements"""

    model_config = {"arbitrary_types_allowed": True}

    angles: list[float] = Field(description="Rotation angle values in radians")
    probabilities: list[float] = Field(description="Measured probabilities")
    probability_errors: list[float] = Field(
        description="Standard errors in probabilities"
    )
    test_unitary: str = Field(description="Type of unitary gate (X, Y, or Z)")
    shots_per_point: int = Field(description="Number of shots per angle point")
    fitting_result: HadamardTestFittingResult | None = Field(
        default=None, description="Hadamard Test fitting results"
    )


class HadamardTestAnalysisResult(ExperimentResult):
    """Complete Hadamard Test experiment result with comprehensive error handling"""

    def __init__(self, data: HadamardTestData, **kwargs):
        """Initialize with Hadamard Test data"""
        super().__init__(**kwargs)
        self.data = data

    def analyze(
        self,
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze Hadamard Test results with comprehensive error handling

        Args:
            plot: Whether to generate plots using plot_settings.md guidelines
            save_data: Whether to save analysis results (handled by experiment classes)
            save_image: Whether to save generated plots

        Returns:
            DataFrame with analysis results including expectation values and phase analysis

        Note:
            Data saving is handled by experiment classes via data manager.
            This method performs data analysis with comprehensive error handling
            and validation to provide helpful feedback when issues occur.
        """
        from ..models.analysis_result import AnalysisResult

        try:
            # Extract measurement data
            data = self.data
            angles = data.angles
            probabilities = data.probabilities
            prob_errors = data.probability_errors
            test_unitary = data.test_unitary

            # Comprehensive input validation
            result = self._validate_hadamard_test_inputs(
                angles, probabilities, prob_errors, test_unitary
            )
            if not result.success:
                return result.to_legacy_dataframe()

            # Check if we have enough data for fitting
            if len(angles) < 4:
                # Create result with insufficient data
                fitting_result = HadamardTestFittingResult(
                    real_part=0.0,
                    imaginary_part=0.0,
                    magnitude=0.0,
                    phase=0.0,
                    phase_degrees=0.0,
                    fidelity=0.0,
                    angles=angles,
                    probabilities=probabilities,
                    r_squared=0.0,
                    error_info="Insufficient data points for fitting (need at least 4)",
                )
                result = AnalysisResult(
                    success=False,
                    error_message="Insufficient data for Hadamard Test fitting",
                    error_type="InsufficientDataError",
                    data={"fitting_result": fitting_result},
                )
                return result.to_legacy_dataframe()

            # Attempt expectation value analysis
            try:
                fitting_result = self._analyze_expectation_values(
                    angles, probabilities, prob_errors, test_unitary
                )

                # Validate analysis results quality
                quality_result = self._validate_analysis_quality(fitting_result)
                if not quality_result.success:
                    return quality_result.to_legacy_dataframe()

                # Update data with fitting results
                self.data.fitting_result = fitting_result

                # Generate plots if requested
                if plot:
                    try:
                        self._create_hadamard_test_plot(fitting_result, save_image)
                    except Exception as e:
                        print(f"Warning: Plot generation failed: {e}")

                # Create successful result
                analysis_data = {
                    "angles": angles,
                    "probabilities": probabilities,
                    "probability_errors": prob_errors,
                    "fitted_probabilities": self._calculate_fitted_probabilities(
                        fitting_result
                    ),
                    "real_part": fitting_result.real_part,
                    "imaginary_part": fitting_result.imaginary_part,
                    "magnitude": fitting_result.magnitude,
                    "phase": fitting_result.phase,
                    "phase_degrees": fitting_result.phase_degrees,
                    "fidelity": fitting_result.fidelity,
                    "r_squared": fitting_result.r_squared,
                    "test_unitary": test_unitary,
                }

                result = AnalysisResult(
                    success=True,
                    data=analysis_data,
                    metadata={"analysis_quality": "good", "error_info": None},
                )
                return result.to_legacy_dataframe()

            except Exception as e:
                # Handle analysis failure gracefully
                fitting_result = HadamardTestFittingResult(
                    real_part=0.0,
                    imaginary_part=0.0,
                    magnitude=0.0,
                    phase=0.0,
                    phase_degrees=0.0,
                    fidelity=0.0,
                    angles=angles,
                    probabilities=probabilities,
                    r_squared=0.0,
                    error_info=f"Analysis failed: {str(e)}",
                )
                result = AnalysisResult(
                    success=False,
                    error_message=f"Hadamard Test analysis failed: {str(e)}",
                    error_type="AnalysisError",
                    data={"fitting_result": fitting_result},
                    suggestions=[
                        "Check if angle range spans sufficient values for expectation value calculation",
                        "Verify test unitary type is correctly specified (X, Y, or Z)",
                        "Consider using more angle points for better statistics",
                    ],
                )
                return result.to_legacy_dataframe()

        except Exception as e:
            # Handle unexpected errors
            result = AnalysisResult(
                success=False,
                error_message=f"Unexpected error in Hadamard Test analysis: {str(e)}",
                error_type="UnexpectedError",
                suggestions=[
                    "Check input data format and types",
                    "Verify all required dependencies are available",
                ],
            )
            return result.to_legacy_dataframe()

    def _validate_hadamard_test_inputs(
        self,
        angles: list[float],
        probabilities: list[float],
        prob_errors: list[float],
        test_unitary: str,
    ) -> "AnalysisResult":
        """Validate Hadamard Test input data"""
        import numpy as np

        from ..models.analysis_result import AnalysisResult
        from ..utils.validation_helpers import (
            check_for_nan_inf,
            validate_data_ranges,
            validate_measurement_data,
        )

        # Basic data validation
        validation_result = validate_measurement_data(
            x_data=angles,
            y_data=probabilities,
            y_errors=prob_errors,
            x_name="angles",
            y_name="probabilities",
        )
        if not validation_result.success:
            return validation_result

        # Check for NaN/inf values
        nan_result = check_for_nan_inf(
            {"angles": angles, "probabilities": probabilities, "errors": prob_errors}
        )
        if not nan_result.success:
            return nan_result

        # Validate ranges
        range_result = validate_data_ranges(
            angles=angles, probabilities=probabilities, prob_errors=prob_errors
        )
        if not range_result.success:
            return range_result

        # Hadamard Test specific validations
        if test_unitary not in ["X", "Y", "Z"]:
            return AnalysisResult(
                success=False,
                error_message=f"Invalid test unitary '{test_unitary}'. Must be 'X', 'Y', or 'Z'",
                error_type="ValidationError",
            )

        # Check angle range
        angle_range = max(angles) - min(angles)
        if angle_range < np.pi:
            return AnalysisResult(
                success=False,
                error_message=f"Insufficient angle range ({angle_range:.2f} rad < π rad) for expectation value calculation",
                error_type="ValidationError",
                suggestions=[
                    "Use angle range of at least π radians for accurate expectation value measurement",
                    "Consider using 2π range for complete characterization",
                ],
            )

        return AnalysisResult(success=True, data={"validation": "passed"})

    def _analyze_expectation_values(
        self,
        angles: list[float],
        probabilities: list[float],
        prob_errors: list[float],
        test_unitary: str,
    ) -> HadamardTestFittingResult:
        """Analyze expectation values from Hadamard Test data"""
        import numpy as np
        from scipy.optimize import curve_fit
        from sklearn.metrics import r2_score

        # Define theoretical expectation functions for each unitary
        def x_expectation(theta, amplitude):
            """X unitary: ⟨ψ(θ)|X|ψ(θ)⟩ = sin(θ)"""
            return 0.5 * (1 + amplitude * np.sin(theta))

        def y_expectation(theta, amplitude):
            """Y unitary: ⟨ψ(θ)|Y|ψ(θ)⟩ = -sin(θ) (Qiskit convention)"""
            return 0.5 * (1 - amplitude * np.sin(theta))

        def z_expectation(theta, amplitude):
            """Z unitary: ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ)"""
            return 0.5 * (1 + amplitude * np.cos(theta))

        # Select appropriate function
        if test_unitary == "X":
            fit_func = x_expectation
            theoretical_func = np.sin
        elif test_unitary == "Y":
            fit_func = y_expectation

            def theoretical_func(x):
                return -np.sin(x)  # Negative for Qiskit convention

        else:  # Z
            fit_func = z_expectation

            def theoretical_func(x):
                return np.cos(x)

        x_data = np.array(angles)
        y_data = np.array(probabilities)
        y_errors = np.array(prob_errors) if prob_errors else None

        # Fit to extract amplitude (should be 1.0 for perfect state preparation)
        try:
            popt, pcov = curve_fit(
                fit_func,
                x_data,
                y_data,
                p0=[1.0],  # Initial guess for amplitude
                sigma=y_errors,
                absolute_sigma=True if y_errors is not None else False,
                bounds=([0.0], [2.0]),  # Amplitude bounds
                maxfev=2000,
            )
            amplitude_fit = popt[0]
        except Exception:
            # Fallback: estimate amplitude from data range
            amplitude_fit = 2 * (max(y_data) - min(y_data))

        # Calculate R-squared
        y_pred = fit_func(x_data, amplitude_fit)
        r_squared = r2_score(y_data, y_pred)

        # Calculate expectation value components
        # For complex expectation values, we need both real and imaginary parts
        if test_unitary in ["X", "Z"]:
            # Real expectation values
            real_part = amplitude_fit if test_unitary == "X" else amplitude_fit
            imaginary_part = 0.0
        else:  # Y
            # Pure imaginary expectation value for Y
            real_part = 0.0
            imaginary_part = -amplitude_fit  # Negative due to i factor

        # Calculate magnitude and phase
        magnitude = np.sqrt(real_part**2 + imaginary_part**2)
        phase = np.arctan2(imaginary_part, real_part)
        phase_degrees = np.degrees(phase)

        # Calculate fidelity (how close amplitude is to 1.0)
        fidelity = max(0.0, min(1.0, amplitude_fit))  # Clamp to [0, 1]

        return HadamardTestFittingResult(
            real_part=float(real_part),
            imaginary_part=float(imaginary_part),
            magnitude=float(magnitude),
            phase=float(phase),
            phase_degrees=float(phase_degrees),
            fidelity=float(fidelity),
            angles=angles,
            probabilities=probabilities,
            r_squared=float(r_squared),
            error_info=None,
        )

    def _validate_analysis_quality(
        self, fitting_result: HadamardTestFittingResult
    ) -> "AnalysisResult":
        """Validate quality of Hadamard Test analysis results"""
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

        # Check fidelity
        if fitting_result.fidelity < 0.5:
            warnings.append(
                f"Low fidelity ({fitting_result.fidelity:.3f}) indicates poor state preparation or measurement"
            )
        elif fitting_result.fidelity < 0.8:
            warnings.append(f"Moderate fidelity ({fitting_result.fidelity:.3f})")

        # Check magnitude reasonableness
        if fitting_result.magnitude > 1.1:
            warnings.append(
                f"Expectation value magnitude ({fitting_result.magnitude:.3f}) exceeds theoretical limit"
            )

        if issues:
            return AnalysisResult(
                success=False,
                error_message="Hadamard Test analysis quality issues: "
                + "; ".join(issues),
                error_type="AnalysisQualityError",
                warnings=warnings,
                suggestions=[
                    "Increase number of angle points for better statistics",
                    "Check state preparation and measurement procedures",
                    "Verify unitary gate implementation",
                ],
            )

        return AnalysisResult(
            success=True, warnings=warnings, data={"quality_check": "passed"}
        )

    def _calculate_fitted_probabilities(
        self, fitting_result: HadamardTestFittingResult
    ) -> list[float]:
        """Calculate fitted probabilities from Hadamard Test parameters"""
        import numpy as np

        # Use fidelity as amplitude for fitted curve
        amplitude = fitting_result.fidelity
        angles = np.array(fitting_result.angles)

        # Calculate based on unitary type (inferred from expectation values)
        if abs(fitting_result.imaginary_part) < 1e-6:  # Real expectation value
            if fitting_result.real_part >= 0:  # X or Z unitary
                # Assume Z unitary if real part is positive
                fitted_probs = 0.5 * (1 + amplitude * np.cos(angles))
            else:
                # Assume X unitary if real part is negative (rare case)
                fitted_probs = 0.5 * (1 + amplitude * np.sin(angles))
        else:  # Y unitary (imaginary expectation value)
            fitted_probs = 0.5 * (1 - amplitude * np.sin(angles))

        return fitted_probs.tolist()

    def _create_hadamard_test_plot(
        self, fitting_result: HadamardTestFittingResult, save_image: bool = True
    ):
        """Create Hadamard Test plot following plot_settings.md guidelines"""
        try:
            import numpy as np
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

            # Create figure with subplots
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Measurement vs Angle", "Complex Expectation Value"],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]],
            )

            # Left plot: measurement data and fit
            fig.add_trace(
                go.Scatter(
                    x=np.degrees(fitting_result.angles),
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
                ),
                row=1,
                col=1,
            )

            # Fitted curve
            angles_fine = np.linspace(
                min(fitting_result.angles), max(fitting_result.angles), 200
            )
            fitted_fine = self._calculate_fitted_probabilities(
                HadamardTestFittingResult(
                    angles=angles_fine.tolist(),
                    probabilities=[],
                    real_part=fitting_result.real_part,
                    imaginary_part=fitting_result.imaginary_part,
                    magnitude=fitting_result.magnitude,
                    phase=fitting_result.phase,
                    phase_degrees=fitting_result.phase_degrees,
                    fidelity=fitting_result.fidelity,
                    r_squared=fitting_result.r_squared,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=np.degrees(angles_fine),
                    y=fitted_fine,
                    mode="lines",
                    name="Fitted Curve",
                    line={"color": colors[0], "width": 2},
                ),
                row=1,
                col=1,
            )

            # Right plot: Complex expectation value on unit circle
            # Unit circle
            circle_angles = np.linspace(0, 2 * np.pi, 100)
            fig.add_trace(
                go.Scatter(
                    x=np.cos(circle_angles),
                    y=np.sin(circle_angles),
                    mode="lines",
                    name="Unit Circle",
                    line={"color": "lightgray", "width": 1, "dash": "dash"},
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # Expectation value point
            fig.add_trace(
                go.Scatter(
                    x=[fitting_result.real_part],
                    y=[fitting_result.imaginary_part],
                    mode="markers",
                    name="Expectation Value",
                    marker={"color": colors[0], "size": 12, "symbol": "star"},
                ),
                row=1,
                col=2,
            )

            # Arrow from origin to expectation value
            fig.add_annotation(
                x=fitting_result.real_part,
                y=fitting_result.imaginary_part,
                ax=0,
                ay=0,
                xref="x2",
                yref="y2",
                axref="x2",
                ayref="y2",
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=colors[0],
                row=1,
                col=2,
            )

            # Update layout
            fig.update_layout(
                title_text=f"Hadamard Test Analysis (Fidelity = {fitting_result.fidelity:.3f})",
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=True,
                width=1000,
                height=500,
            )

            # Update axes
            fig.update_xaxes(title_text="Angle (degrees)", row=1, col=1)
            fig.update_yaxes(title_text="P(|0⟩)", row=1, col=1, range=[0, 1])
            fig.update_xaxes(title_text="Real Part", row=1, col=2, range=[-1.2, 1.2])
            fig.update_yaxes(
                title_text="Imaginary Part", row=1, col=2, range=[-1.2, 1.2]
            )

            # Add results annotation
            annotation_text = (
                f"⟨U⟩ = {fitting_result.real_part:.3f} + {fitting_result.imaginary_part:.3f}i<br>"
                f"Magnitude = {fitting_result.magnitude:.3f}<br>"
                f"Phase = {fitting_result.phase_degrees:.1f}°<br>"
                f"Fidelity = {fitting_result.fidelity:.3f}<br>"
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
                save_plotly_figure(
                    fig, name="hadamard_test_analysis", images_dir="./plots"
                )

            config = get_plotly_config("hadamard_test_analysis", 1000, 500)
            show_plotly_figure(fig, config)

        except Exception as e:
            print(f"Warning: Hadamard Test plot generation failed: {e}")
