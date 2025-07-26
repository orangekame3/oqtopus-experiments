#!/usr/bin/env python3
"""
Hadamard Test Experiment Class
"""

from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from ..core.base_experiment import BaseExperiment
from ..models.hadamard_test_models import (
    HadamardTestAnalysisResult,
    HadamardTestCircuitParams,
    HadamardTestFittingResult,
    HadamardTestParameters,
)


class HadamardTest(BaseExperiment):
    """Hadamard Test experiment using user's approach

    Measures ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ) where:
    - |ψ(θ)⟩ = RX(θ)|0⟩ (parameterized initial state preparation)
    - Z is the fixed test unitary
    - θ varies from 0 to max_angle with angle_points steps

    This approach provides intuitive understanding of quantum expectation values
    and shows "phase kickback effect as theta dependence".
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        physical_qubit: int | None = None,
        test_unitary: str = "Z",
        angle_points: int = 16,
        max_angle: float = 2 * np.pi,
    ):
        """Initialize Hadamard Test experiment"""
        # Track if physical_qubit was explicitly specified
        self._physical_qubit_specified = physical_qubit is not None
        actual_physical_qubit = physical_qubit if physical_qubit is not None else 0

        # Validate test_unitary
        test_unitary = test_unitary.upper()
        if test_unitary not in ["X", "Y", "Z"]:
            raise ValueError(f"test_unitary must be X, Y, or Z, got {test_unitary}")

        self.params = HadamardTestParameters(
            experiment_name=experiment_name,
            physical_qubit=actual_physical_qubit,
            test_unitary=test_unitary,
            angle_points=angle_points,
            max_angle=max_angle,
        )
        super().__init__(self.params.experiment_name or "hadamard_test_experiment")

        self.physical_qubit = self.params.physical_qubit
        self.test_unitary = self.params.test_unitary
        self.angle_points = self.params.angle_points
        self.max_angle = self.params.max_angle

        # Determine optimal initial state preparation based on test unitary
        self.initial_state_prep = self._get_optimal_initial_prep()

    def _get_optimal_initial_prep(self) -> str:
        """Get optimal initial state preparation for the test unitary

        Returns the rotation gate that provides maximum variation in expectation value:
        - X unitary: RY(θ) → ⟨ψ(θ)|X|ψ(θ)⟩ = sin(θ)
        - Y unitary: RX(θ) → ⟨ψ(θ)|Y|ψ(θ)⟩ = -sin(θ) (Qiskit RX convention)
        - Z unitary: RX(θ) → ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ)
        """
        if self.test_unitary == "X":
            return "RY"  # RY(θ)|0⟩ creates superposition in X-Z plane → optimal for X measurement
        elif self.test_unitary == "Y":
            return "RX"  # RX(θ)|0⟩ creates superposition in Y-Z plane → optimal for Y measurement
        elif self.test_unitary == "Z":
            return "RX"  # RX(θ)|0⟩ creates superposition in X-Z plane → optimal for Z measurement
        else:
            raise ValueError(f"Unknown test_unitary: {self.test_unitary}")

    def _get_theoretical_formula(self) -> str:
        """Get theoretical expectation value formula for current setup"""
        if self.test_unitary == "X" and self.initial_state_prep == "RY":
            return "sin(θ)"
        elif self.test_unitary == "Y" and self.initial_state_prep == "RX":
            return "-sin(θ)"  # Qiskit RX(θ) convention gives negative sign
        elif self.test_unitary == "Z" and self.initial_state_prep == "RX":
            return "cos(θ)"
        else:
            return "unknown"

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """Analyze Hadamard Test results"""

        if not results:
            return pd.DataFrame()

        # Flatten all results into single list
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Calculate expectation values
        fitting_result = self._calculate_expectation_values(all_results)
        if not fitting_result:
            return pd.DataFrame()

        # Get device name from results
        device_name = "unknown"
        if all_results:
            device_name = all_results[0].get("backend", "unknown")

        # Create DataFrame
        df = self._create_dataframe(fitting_result, device_name)

        # Create analysis result
        analysis_result = HadamardTestAnalysisResult(
            fitting_result=fitting_result,
            dataframe=df,
            metadata={
                "experiment_type": "hadamard_test",
                "physical_qubit": self.physical_qubit,
                "test_unitary": self.test_unitary,
            },
        )

        # Optional actions
        if plot:
            self._create_plot(analysis_result, save_image)
        if save_data:
            self._save_results(analysis_result)

        return df

    def circuits(self, **kwargs: Any) -> list["QuantumCircuit"]:
        """Generate Hadamard Test circuits with optimal initial state preparation

        Creates circuits with:
        - Optimal initial state preparation (RX/RY) based on test unitary
        - Test unitary measurement via controlled gate (CX/CY/CZ)
        - Measures ⟨ψ(θ)|U|ψ(θ)⟩ with theoretical formulas:
          * X unitary + RY(θ): sin(θ)
          * Y unitary + RX(θ): sin(θ)
          * Z unitary + RX(θ): cos(θ)
        """
        circuits = []
        angles = np.linspace(0, self.max_angle, self.angle_points)

        for angle in angles:
            # Create Hadamard test circuit
            qc = self._create_hadamard_test_circuit(angle)
            circuits.append(qc)

        # Store parameters for analysis
        self.experiment_params = {
            "angles": angles,
            "logical_qubit": 0,
            "physical_qubit": self.physical_qubit,
            "test_unitary": self.test_unitary,
        }

        # Auto-transpile if physical qubit explicitly specified
        if (
            hasattr(self, "_physical_qubit_specified")
            and self._physical_qubit_specified
        ):
            # For 2-qubit circuits, map logical qubits [0,1] to physical qubits
            circuits = self._transpile_circuits_with_tranqu(
                circuits, [0, 1], [self.physical_qubit, self.physical_qubit + 1]
            )

        return circuits

    def _create_hadamard_test_circuit(self, angle: float) -> QuantumCircuit:
        """Create Hadamard Test circuit with optimal initial state preparation

        Automatically selects optimal initial state preparation:
        - X unitary: RY(θ) preparation → ⟨ψ(θ)|X|ψ(θ)⟩ = sin(θ)
        - Y unitary: RX(θ) preparation → ⟨ψ(θ)|Y|ψ(θ)⟩ = sin(θ)
        - Z unitary: RX(θ) preparation → ⟨ψ(θ)|Z|ψ(θ)⟩ = cos(θ)

        Circuit topology:
        q_0: ──[H]───■─[H]──[M]    (ancilla)
        q_1: [R(θ)]──■─────────    (target with optimal preparation)

        Args:
            angle: Angle θ for initial state preparation
        """
        qc = QuantumCircuit(2, 1)

        # Apply optimal initial state preparation
        if self.initial_state_prep == "RX":
            qc.rx(angle, 1)  # For Y and Z unitaries
        elif self.initial_state_prep == "RY":
            qc.ry(angle, 1)  # For X unitary
        else:
            raise ValueError(f"Unknown initial_state_prep: {self.initial_state_prep}")

        # Prepare ancilla in superposition for quantum interference
        qc.h(0)

        # Apply controlled test unitary
        if self.test_unitary == "X":
            qc.cx(0, 1)  # Controlled-X
        elif self.test_unitary == "Y":
            qc.cy(0, 1)  # Controlled-Y
        elif self.test_unitary == "Z":
            qc.cz(0, 1)  # Controlled-Z
        else:
            raise ValueError(f"Unknown test_unitary: {self.test_unitary}")

        # Final Hadamard on ancilla to complete interference measurement
        qc.h(0)

        # Measure ancilla qubit
        qc.measure(0, 0)

        return qc

    def _calculate_expectation_values(
        self, all_results: list[dict[str, Any]]
    ) -> HadamardTestFittingResult | None:
        """Calculate expectation values from Hadamard Test results"""
        try:
            # Extract data points
            angles, probs = self._extract_data_points(all_results)
            if len(angles) < 4:
                return None

            # Fit expectation values
            real_expectation, r_squared = self._fit_expectation_curve(angles, probs)

            # For real-only Hermitian operators
            imaginary_expectation = 0.0
            magnitude = abs(real_expectation)
            phase = 0.0 if real_expectation >= 0 else np.pi
            phase_degrees = np.degrees(phase)

            # Fidelity based on magnitude (ideal = 1)
            fidelity = min(magnitude, 1.0)

            return HadamardTestFittingResult(
                real_part=real_expectation,
                imaginary_part=imaginary_expectation,
                magnitude=magnitude,
                phase=phase,
                phase_degrees=phase_degrees,
                fidelity=fidelity,
                angles=angles.tolist(),
                probabilities=probs.tolist(),
                r_squared=r_squared,
            )

        except Exception as e:
            return HadamardTestFittingResult(
                real_part=0.0,
                imaginary_part=0.0,
                magnitude=0.0,
                phase=0.0,
                phase_degrees=0.0,
                fidelity=0.0,
                angles=[],
                probabilities=[],
                r_squared=0.0,
                error_info=str(e),
            )

    def _extract_data_points(
        self, all_results: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract angle and probability data points"""
        return self._extract_angle_data_points(all_results)

    def _extract_angle_data_points(
        self, all_results: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract angle and probability data points"""
        angles: list[float] = []
        probs: list[float] = []

        for result in all_results:
            # Get angle from embedded params
            angle = None

            if "params" in result:
                angle = result["params"].get("angle")
            elif (
                hasattr(self, "experiment_params")
                and "angles" in self.experiment_params
            ):
                # Fallback to experiment params
                circuit_idx = result.get("params", {}).get("circuit_index", len(angles))
                angles_data = self.experiment_params["angles"]
                if (
                    hasattr(angles_data, "__len__")
                    and hasattr(angles_data, "__getitem__")
                    and circuit_idx < len(angles_data)
                ):
                    angle = angles_data[circuit_idx]

            if angle is None:
                continue

            # Extract probability P(|0⟩)
            counts = result.get("counts", {})
            total = sum(counts.values())
            if total > 0:
                prob_0 = counts.get("0", counts.get(0, 0)) / total
                angles.append(angle)
                probs.append(prob_0)

        if not angles:
            return np.array([]), np.array([])

        # Convert to arrays and sort by angle
        angles_array = np.array(angles)
        probs_array = np.array(probs)

        sort_indices = np.argsort(angles_array)
        sorted_angles = angles_array[sort_indices]
        sorted_probs = probs_array[sort_indices]

        return sorted_angles, sorted_probs

    def _fit_expectation_curve(
        self, angles: np.ndarray, probabilities: np.ndarray
    ) -> tuple[float, float]:
        """Fit expectation value based on test unitary and initial state preparation"""
        # Hadamard Test: P(|0⟩) = (1 + Re⟨ψ|U|ψ⟩)/2
        # So: Re⟨ψ|U|ψ⟩ = 2*P(|0⟩) - 1

        # Convert probabilities to expectation values
        expectation_values = 2 * probabilities - 1

        # Fit to appropriate theoretical curve based on setup
        theoretical_formula = self._get_theoretical_formula()
        if theoretical_formula == "cos(θ)":
            return self._fit_cosine_curve(angles, expectation_values)
        elif theoretical_formula == "sin(θ)":
            return self._fit_sine_curve(angles, expectation_values)
        elif theoretical_formula == "-sin(θ)":
            return self._fit_negative_sine_curve(angles, expectation_values)
        else:
            # Fallback to cosine fitting
            return self._fit_cosine_curve(angles, expectation_values)

    def _fit_cosine_curve(
        self, angles: np.ndarray, expectation_values: np.ndarray
    ) -> tuple[float, float]:
        """Fit to cos(θ) curve for user's RX(θ) + Z approach"""
        from scipy.optimize import curve_fit

        def cosine_func(theta, amplitude, phase, offset):
            return amplitude * np.cos(theta + phase) + offset

        try:
            # Initial guess for cos(θ) fitting
            amp_guess = (np.max(expectation_values) - np.min(expectation_values)) / 2
            phase_guess = 0.0  # For RX(θ) + Z, should start at cos(0) = 1
            offset_guess = np.mean(expectation_values)

            popt, _ = curve_fit(
                cosine_func,
                angles,
                expectation_values,
                p0=[amp_guess, phase_guess, offset_guess],
                bounds=([-2, -np.pi, -2], [2, np.pi, 2]),
            )

            amplitude, phase, offset = popt

            # Calculate R-squared
            y_pred = cosine_func(angles, *popt)
            ss_res = np.sum((expectation_values - y_pred) ** 2)
            ss_tot = np.sum((expectation_values - np.mean(expectation_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return amplitude, max(0, min(1, r_squared))

        except Exception:
            # Fallback: simple average
            return np.mean(expectation_values), 0.0

    def _fit_sine_curve(
        self, angles: np.ndarray, expectation_values: np.ndarray
    ) -> tuple[float, float]:
        """Fit to sin(θ) curve for X and Y unitary measurements"""
        from scipy.optimize import curve_fit

        def sine_func(theta, amplitude, phase, offset):
            return amplitude * np.sin(theta + phase) + offset

        try:
            # Initial guess for sin(θ) fitting
            amp_guess = (np.max(expectation_values) - np.min(expectation_values)) / 2
            phase_guess = 0.0  # For sin(θ), should start at sin(0) = 0
            offset_guess = np.mean(expectation_values)

            popt, _ = curve_fit(
                sine_func,
                angles,
                expectation_values,
                p0=[amp_guess, phase_guess, offset_guess],
                bounds=([-2, -np.pi, -2], [2, np.pi, 2]),
            )

            amplitude, phase, offset = popt

            # Calculate R-squared
            y_pred = sine_func(angles, *popt)
            ss_res = np.sum((expectation_values - y_pred) ** 2)
            ss_tot = np.sum((expectation_values - np.mean(expectation_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return amplitude, max(0, min(1, r_squared))

        except Exception:
            # Fallback: simple average
            return np.mean(expectation_values), 0.0

    def _fit_negative_sine_curve(
        self, angles: np.ndarray, expectation_values: np.ndarray
    ) -> tuple[float, float]:
        """Fit to -sin(θ) curve for Y unitary with RX(θ) preparation (Qiskit convention)"""
        from scipy.optimize import curve_fit

        def negative_sine_func(theta, amplitude, phase, offset):
            return -amplitude * np.sin(theta + phase) + offset

        try:
            # Initial guess for -sin(θ) fitting
            amp_guess = (np.max(expectation_values) - np.min(expectation_values)) / 2
            phase_guess = 0.0  # For -sin(θ), should start at -sin(0) = 0
            offset_guess = np.mean(expectation_values)

            popt, _ = curve_fit(
                negative_sine_func,
                angles,
                expectation_values,
                p0=[amp_guess, phase_guess, offset_guess],
                bounds=([-2, -np.pi, -2], [2, np.pi, 2]),
            )

            amplitude, phase, offset = popt

            # Calculate R-squared
            y_pred = negative_sine_func(angles, *popt)
            ss_res = np.sum((expectation_values - y_pred) ** 2)
            ss_tot = np.sum((expectation_values - np.mean(expectation_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return amplitude, max(0, min(1, r_squared))

        except Exception:
            # Fallback: simple average
            return np.mean(expectation_values), 0.0

    def _create_dataframe(
        self, fitting_result: HadamardTestFittingResult, device_name: str = "unknown"
    ) -> pd.DataFrame:
        """Create DataFrame from fitting results"""
        df_data = []

        for i, angle in enumerate(fitting_result.angles):
            prob = (
                fitting_result.probabilities[i]
                if i < len(fitting_result.probabilities)
                else 0.5
            )

            df_data.append(
                {
                    "device": device_name,
                    "angle": angle,
                    "angle_degrees": np.degrees(angle),
                    "probability": prob,
                    "expectation": 2 * prob - 1,
                    "real_part": fitting_result.real_part,
                    "imaginary_part": fitting_result.imaginary_part,
                    "magnitude": fitting_result.magnitude,
                    "phase": fitting_result.phase,
                    "fidelity": fitting_result.fidelity,
                    "test_unitary": self.test_unitary,
                }
            )
        return pd.DataFrame(df_data) if df_data else pd.DataFrame()

    def _create_plot(
        self, analysis_result: HadamardTestAnalysisResult, save_image: bool = False
    ):
        """Create comprehensive visualization for Hadamard Test results"""
        try:
            import plotly.graph_objects as go

            from ..utils.visualization import (
                apply_experiment_layout,
                get_experiment_colors,
                get_plotly_config,
                save_plotly_figure,
                setup_plotly_environment,
                show_plotly_figure,
            )

            setup_plotly_environment()
            colors = get_experiment_colors()

            df = analysis_result.dataframe
            result = analysis_result.fitting_result

            # Get device name from dataframe or use fallback
            device_name = "unknown"
            if not df.empty and "device" in df.columns:
                device_name = df["device"].iloc[0]
            elif hasattr(self, "_last_backend_device"):
                device_name = self._last_backend_device

            # Create single plot for expectation vs angle
            fig = go.Figure()

            # Expectation vs angle plot
            fig.add_trace(
                go.Scatter(
                    x=df["angle_degrees"],
                    y=df["expectation"],
                    mode="markers",
                    name="Data",
                    marker={
                        "size": 7,
                        "color": colors[1],
                        "line": {"width": 1, "color": "white"},
                    },
                )
            )

            if not result.error_info:
                x_fine = np.linspace(0, np.degrees(self.max_angle), 200)
                x_fine_rad = np.radians(x_fine)
                # Get theoretical expectation values based on test unitary
                theoretical_formula = self._get_theoretical_formula()
                if theoretical_formula == "cos(θ)":
                    y_fine = np.cos(x_fine_rad)
                elif theoretical_formula == "sin(θ)":
                    y_fine = np.sin(x_fine_rad)
                elif theoretical_formula == "-sin(θ)":
                    y_fine = -np.sin(x_fine_rad)
                else:
                    y_fine = np.cos(x_fine_rad)  # Fallback

                fig.add_trace(
                    go.Scatter(
                        x=x_fine,
                        y=y_fine,
                        mode="lines",
                        name="Theory",
                        line={"width": 3, "color": colors[0]},
                    )
                )

            # Apply layout with white background and proper styling
            theoretical_formula = self._get_theoretical_formula()
            apply_experiment_layout(
                fig,
                title=f"Hadamard Test ({self.initial_state_prep}(θ) + {self.test_unitary}) : Q{self.physical_qubit} ({device_name}) → {theoretical_formula}",
                xaxis_title="Angle (degrees)",
                yaxis_title="⟨U⟩",
                height=400,
                width=700,
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
            fig.update_yaxes(
                range=[-1.05, 1.05],
                showgrid=True,
                gridwidth=1,
                gridcolor="LightGray",
            )

            # Add comprehensive annotations
            if not result.error_info:
                annotation_text = (
                    f"<b>Device:</b> {device_name}<br>"
                    f"<b>Unitary:</b> {self.test_unitary}<br>"
                    f"<b>Real Part:</b> {result.real_part:.3f}<br>"
                    f"<b>Imaginary Part:</b> {result.imaginary_part:.3f}<br>"
                    f"<b>Magnitude:</b> {result.magnitude:.3f}<br>"
                    f"<b>Phase:</b> {result.phase_degrees:.1f}°<br>"
                    f"<b>Fidelity:</b> {result.fidelity:.3f}<br>"
                    f"<b>R²:</b> {result.r_squared:.3f}"
                )

                fig.add_annotation(
                    x=0.02,
                    y=0.98,
                    text=annotation_text,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font={"size": 11, "color": "#333333"},
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#CCCCCC",
                    borderwidth=1,
                    align="left",
                )

            # Save and show
            if save_image:
                images_dir = (
                    getattr(self.data_manager, "session_dir", "./images") + "/plots"
                )
                save_plotly_figure(
                    fig,
                    name=f"hadamard_test_{self.test_unitary}_{self.physical_qubit}",
                    images_dir=images_dir,
                    width=700,
                    height=400,
                )

            config = get_plotly_config(
                f"hadamard_test_{self.test_unitary}_Q{self.physical_qubit}",
                width=700,
                height=400,
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("plotly not available, skipping plot")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def _save_results(self, analysis_result: HadamardTestAnalysisResult):
        """Save analysis results"""
        try:
            result = analysis_result.fitting_result
            saved_path = self.save_experiment_data(
                analysis_result.dataframe.to_dict(orient="records"),
                metadata={
                    "fitting_summary": {
                        "real_part": result.real_part,
                        "imaginary_part": result.imaginary_part,
                        "magnitude": result.magnitude,
                        "phase": result.phase,
                        "fidelity": result.fidelity,
                        "test_unitary": self.test_unitary,
                        "r_squared": result.r_squared,
                    },
                    **analysis_result.metadata,
                },
                experiment_type="hadamard_test",
            )
            print(f"Analysis data saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: Could not save analysis data: {e}")

    def _get_circuit_params(self) -> list[dict[str, Any]] | None:
        """Get circuit parameters for OQTOPUS"""
        if not hasattr(self, "experiment_params"):
            return None

        angles = self.experiment_params["angles"]
        test_unitary = self.experiment_params.get("test_unitary", "RZ")
        logical_qubit = self.experiment_params.get("logical_qubit", 0)
        physical_qubit = self.experiment_params.get("physical_qubit", logical_qubit)

        circuit_params = []
        if hasattr(angles, "__iter__"):
            for angle in angles:
                # Circuit measurement
                param_model = HadamardTestCircuitParams(
                    angle=float(angle),
                    test_unitary=test_unitary,
                    logical_qubit=(
                        int(logical_qubit)
                        if isinstance(logical_qubit, (int | float))
                        else 0
                    ),
                    physical_qubit=(
                        int(physical_qubit)
                        if isinstance(physical_qubit, (int | float))
                        else 0
                    ),
                )
                circuit_params.append(param_model.model_dump())

        return circuit_params
