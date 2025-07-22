#!/usr/bin/env python3
"""
Rabi Experiment Class
"""

from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ..core.base_experiment import BaseExperiment
from ..models.rabi_models import (
    RabiAnalysisResult,
    RabiCircuitParams,
    RabiFittingResult,
    RabiParameters,
)


class Rabi(BaseExperiment):
    """Rabi experiment"""

    def __init__(
        self,
        experiment_name: str | None = None,
        physical_qubit: int | None = None,
        amplitude_points: int = 10,
        max_amplitude: float = 2.0,
    ):
        """Initialize Rabi experiment with explicit parameters"""
        # Track if physical_qubit was explicitly specified
        self._physical_qubit_specified = physical_qubit is not None
        actual_physical_qubit = physical_qubit if physical_qubit is not None else 0

        self.params = RabiParameters(
            experiment_name=experiment_name,
            physical_qubit=actual_physical_qubit,
            amplitude_points=amplitude_points,
            max_amplitude=max_amplitude,
        )
        super().__init__(self.params.experiment_name or "rabi_experiment")

        self.physical_qubit = self.params.physical_qubit
        self.amplitude_points = self.params.amplitude_points
        self.max_amplitude = self.params.max_amplitude

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """Analyze Rabi results with simplified single-result processing"""

        if not results:
            return pd.DataFrame()

        # Flatten all results into single list (no device separation)
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Fit the data
        fitting_result = self._fit_rabi_data(all_results)
        if not fitting_result:
            return pd.DataFrame()

        # Get device name from results
        device_name = "unknown"
        if all_results:
            # Get device name from first result's backend field
            device_name = all_results[0].get("backend", "unknown")

        # Create DataFrame
        df = self._create_dataframe(fitting_result, device_name)

        # Create analysis result
        analysis_result = RabiAnalysisResult(
            fitting_result=fitting_result,
            dataframe=df,
            metadata={"experiment_type": "rabi", "physical_qubit": self.physical_qubit},
        )

        # Optional actions
        if plot:
            self._create_plot(analysis_result, save_image)
        if save_data:
            self._save_results(analysis_result)

        return df

    def circuits(self, **kwargs: Any) -> list[Any]:
        """Generate Rabi circuits with automatic transpilation"""
        amplitudes = np.linspace(0, self.max_amplitude, self.amplitude_points)
        circuits = []

        for amplitude in amplitudes:
            qc = QuantumCircuit(1, 1)
            if amplitude > 0:
                qc.rx(amplitude * np.pi, 0)
            qc.measure(0, 0)
            circuits.append(qc)

        # Store parameters for analysis and OQTOPUS
        self.experiment_params = {
            "amplitudes": amplitudes,
            "logical_qubit": 0,
            "physical_qubit": self.physical_qubit,
        }

        # Auto-transpile if physical qubit explicitly specified using base class method
        if (
            hasattr(self, "_physical_qubit_specified")
            and self._physical_qubit_specified
        ):
            circuits = self._transpile_circuits_with_tranqu(
                circuits, 0, self.physical_qubit
            )

        return circuits  # type: ignore

    def _fit_rabi_data(
        self, all_results: list[dict[str, Any]]
    ) -> RabiFittingResult | None:
        """Fit Rabi oscillation for all data combined"""
        try:
            # Extract data points
            amplitudes, probabilities = self._extract_data_points(all_results)
            if len(amplitudes) < 4:
                return None

            # Perform fitting
            popt = self._perform_rabi_fit(amplitudes, probabilities)
            if popt is None:
                return None

            fitted_amplitude, fitted_freq, fitted_offset = popt
            pi_amplitude = 0.5 / fitted_freq
            r_squared = self._calculate_r_squared(amplitudes, probabilities, popt)

            return RabiFittingResult(
                pi_amplitude=pi_amplitude,
                frequency=fitted_freq,
                fit_amplitude=fitted_amplitude,
                offset=fitted_offset,
                r_squared=r_squared,
                amplitudes=amplitudes.tolist(),
                probabilities=probabilities.tolist(),
            )

        except Exception as e:
            return RabiFittingResult(
                pi_amplitude=1.0,
                frequency=0.5,
                fit_amplitude=0.5,
                offset=0.5,
                r_squared=0.0,
                amplitudes=[],
                probabilities=[],
                error_info=str(e),
            )

    def _extract_data_points(
        self, all_results: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract amplitude and probability data points"""
        amplitudes: list[float] = []
        probabilities: list[float] = []

        for result in all_results:
            # Get amplitude from embedded params
            amplitude = None
            if "params" in result and "amplitude" in result["params"]:
                amplitude = result["params"]["amplitude"]
            elif (
                hasattr(self, "experiment_params")
                and "amplitudes" in self.experiment_params
            ):
                # Fallback to experiment params
                circuit_idx = result.get("params", {}).get(
                    "circuit_index", len(amplitudes)
                )
                amplitudes_data = self.experiment_params["amplitudes"]
                if (
                    hasattr(amplitudes_data, "__len__")
                    and hasattr(amplitudes_data, "__getitem__")
                    and circuit_idx < len(amplitudes_data)
                ):
                    amplitude = amplitudes_data[circuit_idx]

            if amplitude is None:
                continue

            # Extract probability of |1⟩
            counts = result.get("counts", {})
            total = sum(counts.values())
            if total > 0:
                prob_1 = counts.get("1", counts.get(1, 0)) / total
                amplitudes.append(amplitude)
                probabilities.append(prob_1)

        if not amplitudes:
            return np.array([]), np.array([])

        amplitudes_array = np.array(amplitudes)
        probabilities_array = np.array(probabilities)

        # Sort by amplitude
        sort_indices = np.argsort(amplitudes_array)
        return amplitudes_array[sort_indices], probabilities_array[sort_indices]

    def _perform_rabi_fit(
        self, amplitudes: np.ndarray, probabilities: np.ndarray
    ) -> np.ndarray | None:
        """Perform Rabi oscillation fitting"""

        def rabi_func(amp, amplitude, frequency, offset):
            return amplitude * np.sin(np.pi * amp * frequency) ** 2 + offset

        # Estimate initial parameters
        amplitude_guess = np.max(probabilities) - np.min(probabilities)
        offset_guess = np.min(probabilities)

        # Estimate frequency from peaks
        threshold = offset_guess + 0.6 * amplitude_guess
        peaks, _ = find_peaks(probabilities, height=threshold, distance=2)

        if len(peaks) >= 2:
            peak_amps = amplitudes[peaks]
            frequency_guess = 1.0 / np.mean(np.diff(peak_amps))
        else:
            frequency_guess = 0.75

        try:
            popt, _ = curve_fit(
                rabi_func,
                amplitudes,
                probabilities,
                p0=[amplitude_guess, frequency_guess, offset_guess],
                bounds=([0, 0.1, 0], [1, 5, 1]),
                maxfev=2000,
            )
            return popt  # type: ignore
        except Exception:
            return None

    def _calculate_r_squared(
        self, amplitudes: np.ndarray, probabilities: np.ndarray, popt: np.ndarray
    ) -> float:
        """Calculate R-squared for fit quality"""

        def rabi_func(amp, amplitude, frequency, offset):
            return amplitude * np.sin(np.pi * amp * frequency) ** 2 + offset

        y_pred = rabi_func(amplitudes, *popt)
        ss_res = np.sum((probabilities - y_pred) ** 2)
        ss_tot = np.sum((probabilities - np.mean(probabilities)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _create_dataframe(
        self, fitting_result: RabiFittingResult, device_name: str = "unknown"
    ) -> pd.DataFrame:
        """Create DataFrame from fitting results"""
        df_data = []
        for amp, prob in zip(
            fitting_result.amplitudes, fitting_result.probabilities, strict=True
        ):
            df_data.append(
                {
                    "device": device_name,
                    "amplitude": amp,
                    "probability": prob,
                    "pi_amplitude": fitting_result.pi_amplitude,
                    "frequency": fitting_result.frequency,
                    "r_squared": fitting_result.r_squared,
                }
            )
        return pd.DataFrame(df_data) if df_data else pd.DataFrame()

    def _create_plot(
        self, analysis_result: RabiAnalysisResult, save_image: bool = False
    ):
        """Create visualization using utilities"""
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
            fig = go.Figure()

            df = analysis_result.dataframe
            result = analysis_result.fitting_result

            # Get device name from dataframe or use fallback
            device_name = "unknown"
            if not df.empty and "device" in df.columns:
                device_name = df["device"].iloc[0]
            elif hasattr(self, "_last_backend_device"):
                device_name = self._last_backend_device

            # Data points
            fig.add_trace(
                go.Scatter(
                    x=df["amplitude"],
                    y=df["probability"],
                    mode="markers",
                    name="Data",
                    marker={
                        "size": 7,
                        "color": colors[1],
                        "line": {"width": 1, "color": "white"},
                    },
                )
            )

            # Fit curve
            if not result.error_info:
                x_fine = np.linspace(df["amplitude"].min(), df["amplitude"].max(), 200)
                y_fine = (
                    result.fit_amplitude
                    * np.sin(np.pi * x_fine * result.frequency) ** 2
                    + result.offset
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_fine,
                        y=y_fine,
                        mode="lines",
                        name="Fit",
                        line={"width": 3, "color": colors[0]},
                    )
                )

            # Apply layout
            apply_experiment_layout(
                fig,
                title=f"Rabi oscillation : Q{self.physical_qubit} ({device_name})",
                xaxis_title="Drive amplitude (arb. unit)",
                yaxis_title="P(|1⟩)",
                height=400,
                width=700,
            )
            fig.update_yaxes(range=[0, 1.05])  # Add 5% padding at top

            # Add vertical line at π-pulse amplitude
            if not result.error_info and result.pi_amplitude <= df["amplitude"].max():
                fig.add_vline(
                    x=result.pi_amplitude,
                    line_dash="dash",
                    line_color=colors[2],
                    line_width=1.5,
                    opacity=0.6,
                )

            # Add annotations for key parameters
            if not result.error_info:
                # Add π-pulse amplitude annotation
                fig.add_annotation(
                    x=result.pi_amplitude,
                    y=0.95,
                    text=f"π-pulse: {result.pi_amplitude:.3f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=colors[2],
                    font={"size": 11, "color": colors[2]},
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=colors[2],
                    borderwidth=1,
                )

                # Add fit quality annotation
                fig.add_annotation(
                    x=0.98,
                    y=0.02,
                    text=f"Device: {device_name}<br>R² = {result.r_squared:.3f}<br>f = {result.frequency:.3f}",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font={"size": 10, "color": "#666666"},
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#CCCCCC",
                    borderwidth=1,
                    align="right",
                )

            # Save and show
            if save_image:
                images_dir = (
                    getattr(self.data_manager, "session_dir", "./images") + "/plots"
                )
                save_plotly_figure(
                    fig,
                    name=f"rabi_{self.physical_qubit}",
                    images_dir=images_dir,
                    width=700,
                    height=400,
                )

            config = get_plotly_config(
                f"rabi_Q{self.physical_qubit}", width=700, height=400
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("plotly not available, skipping plot")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def _save_results(self, analysis_result: RabiAnalysisResult):
        """Save analysis results"""
        try:
            result = analysis_result.fitting_result
            saved_path = self.save_experiment_data(
                analysis_result.dataframe.to_dict(orient="records"),
                metadata={
                    "fitting_summary": {
                        "pi_amplitude": result.pi_amplitude,
                        "frequency": result.frequency,
                        "r_squared": result.r_squared,
                    },
                    **analysis_result.metadata,
                },
                experiment_type="rabi",
            )
            print(f"Analysis data saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: Could not save analysis data: {e}")

    def _get_circuit_params(self) -> list[dict[str, Any]] | None:
        """Get circuit parameters for OQTOPUS"""
        if not hasattr(self, "experiment_params"):
            return None

        amplitudes = self.experiment_params["amplitudes"]
        logical_qubit = self.experiment_params.get("logical_qubit", 0)
        physical_qubit = self.experiment_params.get("physical_qubit", logical_qubit)

        circuit_params = []
        if hasattr(amplitudes, "__iter__"):
            for amp in amplitudes:
                param_model = RabiCircuitParams(
                    amplitude=float(amp),
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
                    rotation_angle=float(amp * np.pi),
                )
                circuit_params.append(param_model.model_dump())

        return circuit_params
