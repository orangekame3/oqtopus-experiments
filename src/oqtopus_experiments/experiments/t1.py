#!/usr/bin/env python3
"""
T1 Experiment Class
"""

from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit

from ..core.base_experiment import BaseExperiment
from ..models.t1_models import (
    T1AnalysisResult,
    T1CircuitParams,
    T1FittingResult,
    T1Parameters,
)


class T1(BaseExperiment):
    """T1 decay experiment"""

    def __init__(
        self,
        experiment_name: str | None = None,
        physical_qubit: int | None = None,
        delay_points: int = 20,
        max_delay: float = 50000.0,
    ):
        """Initialize T1 experiment with explicit parameters"""
        # Track if physical_qubit was explicitly specified
        self._physical_qubit_specified = physical_qubit is not None
        actual_physical_qubit = physical_qubit if physical_qubit is not None else 0

        self.params = T1Parameters(
            experiment_name=experiment_name,
            physical_qubit=actual_physical_qubit,
            delay_points=delay_points,
            max_delay=max_delay,
        )
        super().__init__(self.params.experiment_name or "t1_experiment")

        self.physical_qubit = self.params.physical_qubit
        self.delay_points = self.params.delay_points
        self.max_delay = self.params.max_delay

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True
    ) -> pd.DataFrame:
        """Analyze T1 results with simplified single-result processing"""

        if not results:
            return pd.DataFrame()

        # Flatten all results into single list (no device separation)
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Fit the data
        fitting_result = self._fit_t1_data(all_results)
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
        analysis_result = T1AnalysisResult(
            fitting_result=fitting_result,
            dataframe=df,
            metadata={"experiment_type": "t1", "physical_qubit": self.physical_qubit},
        )

        # Optional actions
        if plot:
            self._create_plot(analysis_result, save_image)
        if save_data:
            self._save_results(analysis_result)

        return df

    def circuits(self, **kwargs: Any) -> list[Any]:
        """Generate T1 circuits with automatic transpilation"""
        delay_times = np.logspace(
            np.log10(1.0),
            np.log10(self.max_delay),
            self.delay_points,
        )
        circuits = []

        for delay in delay_times:
            qc = QuantumCircuit(1, 1)
            qc.x(0)  # Prepare |1⟩ state
            qc.delay(delay, 0, unit="ns")  # Wait for decay
            qc.measure(0, 0)  # Measure final state
            circuits.append(qc)

        # Store parameters for analysis and OQTOPUS
        self.experiment_params = {
            "delay_times": delay_times,
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

    def _fit_t1_data(self, all_results: list[dict[str, Any]]) -> T1FittingResult | None:
        """Fit T1 decay for all data combined"""
        try:
            # Extract data points
            delay_times, probabilities = self._extract_data_points(all_results)
            if len(delay_times) < 4:
                return None

            # Perform fitting
            popt = self._perform_t1_fit(delay_times, probabilities)
            if popt is None:
                return None

            fitted_amplitude, fitted_t1, fitted_offset = popt
            r_squared = self._calculate_r_squared(delay_times, probabilities, popt)

            return T1FittingResult(
                t1_time=fitted_t1,
                amplitude=fitted_amplitude,
                offset=fitted_offset,
                r_squared=r_squared,
                delay_times=delay_times.tolist(),
                probabilities=probabilities.tolist(),
            )

        except Exception as e:
            return T1FittingResult(
                t1_time=1000.0,
                amplitude=0.5,
                offset=0.5,
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

    def _perform_t1_fit(
        self, delay_times: np.ndarray, probabilities: np.ndarray
    ) -> np.ndarray | None:
        """Perform T1 exponential decay fitting"""

        def t1_func(t, amplitude, t1, offset):
            return amplitude * np.exp(-t / t1) + offset

        # Estimate initial parameters
        amplitude_guess = np.max(probabilities) - np.min(probabilities)
        offset_guess = np.min(probabilities)
        t1_guess = 1000.0  # Initial estimate [ns]

        try:
            popt, _ = curve_fit(
                t1_func,
                delay_times,
                probabilities,
                p0=[amplitude_guess, t1_guess, offset_guess],
                bounds=([0, 1, -0.1], [2, 1e6, 1.1]),
                maxfev=2000,
            )
            return popt  # type: ignore
        except Exception:
            return None

    def _calculate_r_squared(
        self, delay_times: np.ndarray, probabilities: np.ndarray, popt: np.ndarray
    ) -> float:
        """Calculate R-squared for fit quality"""

        def t1_func(t, amplitude, t1, offset):
            return amplitude * np.exp(-t / t1) + offset

        y_pred = t1_func(delay_times, *popt)
        ss_res = np.sum((probabilities - y_pred) ** 2)
        ss_tot = np.sum((probabilities - np.mean(probabilities)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _create_dataframe(
        self, fitting_result: T1FittingResult, device_name: str = "unknown"
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
                    "t1_time": fitting_result.t1_time,
                    "amplitude": fitting_result.amplitude,
                    "r_squared": fitting_result.r_squared,
                }
            )
        return pd.DataFrame(df_data) if df_data else pd.DataFrame()

    def _create_plot(self, analysis_result: T1AnalysisResult, save_image: bool = False):
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
                    df["delay_time"].min(), df["delay_time"].max(), 200
                )
                y_fine = (
                    result.amplitude * np.exp(-x_fine / result.t1_time) + result.offset
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
                title=f"T1 decay : Q{self.physical_qubit} ({device_name})",
                xaxis_title="Delay time (ns)",
                yaxis_title="P(|1⟩)",
                height=400,
                width=700,
            )
            fig.update_yaxes(range=[0, 1.05])  # Add 5% padding at top
            fig.update_xaxes(type="log")  # Use logarithmic scale for delay time

            # Add annotations for key parameters
            if not result.error_info:
                # Add T1 time annotation
                fig.add_annotation(
                    x=0.98,
                    y=0.02,
                    text=f"Device: {device_name}<br>T₁ = {result.t1_time:.1f} ns ({result.t1_time/1000:.2f} μs)<br>R² = {result.r_squared:.3f}",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=10, color="#666666"),
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
                    name=f"t1_{self.physical_qubit}",
                    images_dir=images_dir,
                    width=700,
                    height=400,
                )

            config = get_plotly_config(
                f"t1_Q{self.physical_qubit}", width=700, height=400
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("plotly not available, skipping plot")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def _save_results(self, analysis_result: T1AnalysisResult):
        """Save analysis results"""
        try:
            result = analysis_result.fitting_result
            saved_path = self.save_experiment_data(
                analysis_result.dataframe.to_dict(orient="records"),
                metadata={
                    "fitting_summary": {
                        "t1_time": result.t1_time,
                        "amplitude": result.amplitude,
                        "r_squared": result.r_squared,
                    },
                    **analysis_result.metadata,
                },
                experiment_type="t1",
            )
            print(f"Analysis data saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: Could not save analysis data: {e}")

    def _get_circuit_params(self) -> list[dict[str, Any]] | None:
        """Get circuit parameters for OQTOPUS"""
        if not hasattr(self, "experiment_params"):
            return None

        delay_times = self.experiment_params["delay_times"]
        logical_qubit = self.experiment_params.get("logical_qubit", 0)
        physical_qubit = self.experiment_params.get("physical_qubit", logical_qubit)

        circuit_params = []
        if hasattr(delay_times, "__iter__"):
            for delay in delay_times:
                param_model = T1CircuitParams(
                    delay_time=float(delay),
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
