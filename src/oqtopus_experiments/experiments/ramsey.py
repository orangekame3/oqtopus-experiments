#!/usr/bin/env python3
"""
Ramsey Experiment Class
"""

from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit

from ..core.base_experiment import BaseExperiment
from ..models.ramsey_models import (
    RamseyAnalysisResult,
    RamseyCircuitParams,
    RamseyFittingResult,
    RamseyParameters,
)


class Ramsey(BaseExperiment):
    """Ramsey fringe experiment for T2* measurement"""

    def __init__(
        self,
        experiment_name: str | None = None,
        physical_qubit: int | None = None,
        delay_points: int = 20,
        max_delay: float = 10000.0,
        detuning_frequency: float = 0.0,
    ):
        """Initialize Ramsey experiment with explicit parameters"""
        # Track if physical_qubit was explicitly specified
        self._physical_qubit_specified = physical_qubit is not None
        actual_physical_qubit = physical_qubit if physical_qubit is not None else 0

        self.params = RamseyParameters(
            experiment_name=experiment_name,
            physical_qubit=actual_physical_qubit,
            delay_points=delay_points,
            max_delay=max_delay,
            detuning_frequency=detuning_frequency,
        )
        super().__init__(self.params.experiment_name or "ramsey_experiment")

        self.physical_qubit = self.params.physical_qubit
        self.delay_points = self.params.delay_points
        self.max_delay = self.params.max_delay
        self.detuning_frequency = self.params.detuning_frequency

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """Analyze Ramsey results with simplified single-result processing"""

        if not results:
            return pd.DataFrame()

        # Flatten all results into single list (no device separation)
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Fit the data
        fitting_result = self._fit_ramsey_data(all_results)
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
        analysis_result = RamseyAnalysisResult(
            fitting_result=fitting_result,
            dataframe=df,
            metadata={
                "experiment_type": "ramsey",
                "physical_qubit": self.physical_qubit,
            },
        )

        # Optional actions
        if plot:
            self._create_plot(analysis_result, save_image)
        if save_data:
            self._save_results(analysis_result)

        return df

    def circuits(self, **kwargs: Any) -> list["QuantumCircuit"]:
        """Generate Ramsey circuits with automatic transpilation"""
        delay_times = np.linspace(0, self.max_delay, self.delay_points)
        circuits = []

        for delay in delay_times:
            qc = QuantumCircuit(1, 1)
            qc.ry(np.pi / 2, 0)  # First π/2 pulse (creates superposition)

            if delay > 0:
                qc.delay(delay, 0, unit="ns")  # Free evolution

            # Apply detuning rotation if specified
            if self.detuning_frequency != 0:
                # Phase accumulation during delay: φ = 2π * f * t
                phase = (
                    2 * np.pi * self.detuning_frequency * delay * 1e-9
                )  # delay in seconds
                qc.rz(phase, 0)

            qc.ry(np.pi / 2, 0)  # Second π/2 pulse (analysis pulse)
            qc.measure(0, 0)  # Measure final state
            circuits.append(qc)

        # Store parameters for analysis and OQTOPUS
        self.experiment_params = {
            "delay_times": delay_times,
            "detuning_frequency": self.detuning_frequency,
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

        return circuits

    def _fit_ramsey_data(
        self, all_results: list[dict[str, Any]]
    ) -> RamseyFittingResult | None:
        """Fit Ramsey fringe oscillation for all data combined"""
        try:
            # Extract data points
            delay_times, probabilities = self._extract_data_points(all_results)
            if len(delay_times) < 4:
                return None

            # Perform fitting
            popt = self._perform_ramsey_fit(delay_times, probabilities)
            if popt is None:
                return None

            (
                fitted_amplitude,
                fitted_freq,
                fitted_t2_star,
                fitted_offset,
                fitted_phase,
            ) = popt
            r_squared = self._calculate_r_squared(delay_times, probabilities, popt)

            return RamseyFittingResult(
                t2_star_time=fitted_t2_star,
                frequency=fitted_freq,
                amplitude=fitted_amplitude,
                offset=fitted_offset,
                phase=fitted_phase,
                r_squared=r_squared,
                delay_times=delay_times.tolist(),
                probabilities=probabilities.tolist(),
            )

        except Exception as e:
            return RamseyFittingResult(
                t2_star_time=1000.0,
                frequency=1e6,
                amplitude=0.5,
                offset=0.5,
                phase=0.0,
                r_squared=0.0,
                delay_times=[],
                probabilities=[],
                error_info=str(e),
            )

    def _extract_data_points(
        self, all_results: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract delay time and probability data points"""
        delay_times: list[float] = []
        probabilities: list[float] = []

        for result in all_results:
            # Get delay from embedded params
            delay = None
            if "params" in result and "delay_time" in result["params"]:
                delay = result["params"]["delay_time"]
            elif (
                hasattr(self, "experiment_params")
                and "delay_times" in self.experiment_params
            ):
                # Fallback to experiment params
                circuit_idx = result.get("params", {}).get(
                    "circuit_index", len(delay_times)
                )
                delay_times_data = self.experiment_params["delay_times"]
                if (
                    hasattr(delay_times_data, "__len__")
                    and hasattr(delay_times_data, "__getitem__")
                    and circuit_idx < len(delay_times_data)
                ):
                    delay = delay_times_data[circuit_idx]

            if delay is None:
                continue

            # Extract probability of |1⟩
            counts = result.get("counts", {})
            total = sum(counts.values())
            if total > 0:
                prob_1 = counts.get("1", counts.get(1, 0)) / total
                delay_times.append(delay)
                probabilities.append(prob_1)

        if not delay_times:
            return np.array([]), np.array([])

        delay_times_array = np.array(delay_times)
        probabilities_array = np.array(probabilities)

        # Sort by delay time
        sort_indices = np.argsort(delay_times_array)
        return delay_times_array[sort_indices], probabilities_array[sort_indices]

    def _perform_ramsey_fit(
        self, delay_times: np.ndarray, probabilities: np.ndarray
    ) -> np.ndarray | None:
        """Perform Ramsey fringe oscillation fitting"""

        def ramsey_func(t, amplitude, frequency, t2_star, offset, phase):
            # Ramsey signal: A * exp(-t/T2*) * cos(2πft + φ) + C
            return (
                amplitude
                * np.exp(-t / t2_star)
                * np.cos(2 * np.pi * frequency * t * 1e-9 + phase)
                + offset
            )

        # Estimate initial parameters
        amplitude_guess = (np.max(probabilities) - np.min(probabilities)) / 2
        offset_guess = np.mean(probabilities)
        t2_star_guess = self.max_delay / 3  # Initial estimate

        # Try to estimate frequency from data
        if len(delay_times) > 10:
            # Simple frequency estimation from zero crossings
            detrended = probabilities - offset_guess
            zero_crossings = np.where(np.diff(np.signbit(detrended)))[0]
            if len(zero_crossings) >= 2:
                period_estimate = 2 * np.mean(np.diff(delay_times[zero_crossings]))
                freq_guess = 1e9 / period_estimate  # Convert to Hz
            else:
                freq_guess = 1e6  # 1 MHz default
        else:
            freq_guess = 1e6

        phase_guess = 0.0

        try:
            # Bounds: [amplitude, frequency, t2_star, offset, phase]
            bounds_lower = [0, 1e3, 10, 0, -np.pi]
            bounds_upper = [1, 1e8, self.max_delay * 10, 1, np.pi]

            popt, _ = curve_fit(
                ramsey_func,
                delay_times,
                probabilities,
                p0=[
                    amplitude_guess,
                    freq_guess,
                    t2_star_guess,
                    offset_guess,
                    phase_guess,
                ],
                bounds=(bounds_lower, bounds_upper),
                maxfev=3000,
            )
            return popt  # type: ignore
        except Exception:
            return None

    def _calculate_r_squared(
        self, delay_times: np.ndarray, probabilities: np.ndarray, popt: np.ndarray
    ) -> float:
        """Calculate R-squared for fit quality"""

        def ramsey_func(t, amplitude, frequency, t2_star, offset, phase):
            return (
                amplitude
                * np.exp(-t / t2_star)
                * np.cos(2 * np.pi * frequency * t * 1e-9 + phase)
                + offset
            )

        y_pred = ramsey_func(delay_times, *popt)
        ss_res = np.sum((probabilities - y_pred) ** 2)
        ss_tot = np.sum((probabilities - np.mean(probabilities)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _create_dataframe(
        self, fitting_result: RamseyFittingResult, device_name: str = "unknown"
    ) -> pd.DataFrame:
        """Create DataFrame from fitting results"""
        df_data = []
        for delay, prob in zip(
            fitting_result.delay_times, fitting_result.probabilities, strict=True
        ):
            df_data.append(
                {
                    "device": device_name,
                    "delay_time": delay,
                    "probability": prob,
                    "t2_star_time": fitting_result.t2_star_time,
                    "frequency": fitting_result.frequency,
                    "amplitude": fitting_result.amplitude,
                    "r_squared": fitting_result.r_squared,
                }
            )
        return pd.DataFrame(df_data) if df_data else pd.DataFrame()

    def _create_plot(
        self, analysis_result: RamseyAnalysisResult, save_image: bool = False
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
                    x=df["delay_time"],
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
                x_fine = np.linspace(
                    df["delay_time"].min(), df["delay_time"].max(), 500
                )
                y_fine = (
                    result.amplitude
                    * np.exp(-x_fine / result.t2_star_time)
                    * np.cos(
                        2 * np.pi * result.frequency * x_fine * 1e-9 + result.phase
                    )
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
                title=f"Ramsey fringe : Q{self.physical_qubit} ({device_name})",
                xaxis_title="Delay time (ns)",
                yaxis_title="P(|1⟩)",
                height=400,
                width=700,
            )
            fig.update_yaxes(range=[0, 1.05])  # Add 5% padding at top

            # Add annotations for key parameters
            if not result.error_info:
                # Add T2* time annotation
                fig.add_annotation(
                    x=0.98,
                    y=0.02,
                    text=f"Device: {device_name}<br>T₂* = {result.t2_star_time:.1f} ns ({result.t2_star_time / 1000:.2f} μs)<br>f = {result.frequency / 1e6:.2f} MHz<br>R² = {result.r_squared:.3f}",
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
                    name=f"ramsey_{self.physical_qubit}",
                    images_dir=images_dir,
                    width=700,
                    height=400,
                )

            config = get_plotly_config(
                f"ramsey_Q{self.physical_qubit}", width=700, height=400
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("plotly not available, skipping plot")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def _save_results(self, analysis_result: RamseyAnalysisResult):
        """Save analysis results"""
        try:
            result = analysis_result.fitting_result
            saved_path = self.save_experiment_data(
                analysis_result.dataframe.to_dict(orient="records"),
                metadata={
                    "fitting_summary": {
                        "t2_star_time": result.t2_star_time,
                        "frequency": result.frequency,
                        "amplitude": result.amplitude,
                        "r_squared": result.r_squared,
                    },
                    **analysis_result.metadata,
                },
                experiment_type="ramsey",
            )
            print(f"Analysis data saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: Could not save analysis data: {e}")

    def _get_circuit_params(self) -> list[dict[str, Any]] | None:
        """Get circuit parameters for OQTOPUS"""
        if not hasattr(self, "experiment_params"):
            return None

        delay_times = self.experiment_params["delay_times"]
        detuning_frequency = self.experiment_params.get("detuning_frequency", 0.0)
        logical_qubit = self.experiment_params.get("logical_qubit", 0)
        physical_qubit = self.experiment_params.get("physical_qubit", logical_qubit)

        circuit_params = []
        if hasattr(delay_times, "__iter__"):
            for delay in delay_times:
                param_model = RamseyCircuitParams(
                    delay_time=float(delay),
                    detuning_frequency=float(detuning_frequency),
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
