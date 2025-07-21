#!/usr/bin/env python3
"""
Simplified Rabi Experiment Class with Pydantic models
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
    """Simplified Rabi oscillation experiment with Pydantic validation"""

    def __init__(self, **kwargs):
        """Initialize with Pydantic parameter validation"""
        # Validate parameters using Pydantic
        self.params = RabiParameters(**kwargs)

        # Initialize base experiment
        super().__init__(self.params.experiment_name or "rabi_experiment")

        # Store validated parameters as instance attributes
        self.physical_qubit = self.params.physical_qubit
        self.amplitude_points = self.params.amplitude_points
        self.max_amplitude = self.params.max_amplitude

    def analyze(
        self, results: dict[str, list[dict[str, Any]]], **kwargs: Any
    ) -> pd.DataFrame:
        """Analyze Rabi results with structured output"""
        plot = kwargs.get("plot", False)
        save_data = kwargs.get("save_data", False)
        save_image = kwargs.get("save_image", False)

        if not results:
            return pd.DataFrame()

        device_results = {}

        # Process results for each device
        for device, device_data in self._prepare_results(results).items():
            fitting_result = self._fit_device_data(device, device_data)
            if fitting_result:
                device_results[device] = fitting_result

        # Create structured analysis result
        df = self._create_dataframe(device_results)
        analysis_result = RabiAnalysisResult(
            device_results=device_results,
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

        # Auto-transpile if physical qubit specified
        if self.physical_qubit is not None:
            circuits = self._transpile_circuits_if_needed(
                circuits, 0, self.physical_qubit
            )

        return circuits  # type: ignore

    def _prepare_results(
        self, results: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Prepare and normalize results structure"""
        if any(key.startswith("circuit_") for key in results.keys()):
            # Circuit-based structure: merge into single device
            device_results = []
            for circuit_key in sorted(results.keys()):
                if results[circuit_key]:
                    device_results.extend(results[circuit_key])

            device_name = "unknown"
            if device_results and "backend" in device_results[0]:
                device_name = device_results[0]["backend"]
            elif hasattr(self, "_last_backend_device"):
                device_name = self._last_backend_device

            return {device_name: device_results}
        else:
            # Device-based structure: use as-is
            return results

    def _fit_device_data(
        self, device: str, device_data: list[dict[str, Any]]
    ) -> RabiFittingResult | None:
        """Fit Rabi oscillation for a single device"""
        try:
            # Extract data
            amplitudes, probabilities = self._extract_data_points(device_data)
            if not amplitudes:
                return None

            # Fit Rabi function
            popt = self._perform_rabi_fit(amplitudes, probabilities)
            if popt is None:
                return None

            fitted_amplitude, fitted_freq, fitted_offset = popt
            pi_amplitude = 0.5 / fitted_freq

            # Calculate fit quality
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
        self, device_data: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract amplitude and probability data points"""
        amplitudes: list[float] = []
        probabilities = []

        for result in device_data:
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
        if len(amplitudes) < 4:  # Need minimum points for fitting
            return None

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
            avg_spacing = np.mean(np.diff(peak_amps))
            frequency_guess = 1.0 / avg_spacing
        else:
            frequency_guess = 0.75  # Default guess

        # Bounds and initial guess
        initial_guess = [amplitude_guess, frequency_guess, offset_guess]
        bounds = ([0, 0.1, 0], [1, 5, 1])

        try:
            popt, _ = curve_fit(
                rabi_func,
                amplitudes,
                probabilities,
                p0=initial_guess,
                bounds=bounds,
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
        self, device_results: dict[str, RabiFittingResult]
    ) -> pd.DataFrame:
        """Create DataFrame from fitting results"""
        df_data = []

        for device, result in device_results.items():
            for amp, prob in zip(result.amplitudes, result.probabilities, strict=False):
                df_data.append(
                    {
                        "device": device,
                        "amplitude": amp,
                        "probability": prob,
                        "pi_amplitude": result.pi_amplitude,
                        "frequency": result.frequency,
                        "r_squared": result.r_squared,
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

            # Plot data for each device
            for device in df["device"].unique():
                device_df = df[df["device"] == device]
                device_result = analysis_result.device_results.get(device)

                # Data points
                fig.add_trace(
                    go.Scatter(
                        x=device_df["amplitude"],
                        y=device_df["probability"],
                        mode="markers",
                        name="Data",
                        marker=dict(size=8, color=colors[1]),
                    )
                )

                # Fit curve
                if device_result and not device_result.error_info:
                    x_fine = np.linspace(
                        device_df["amplitude"].min(), device_df["amplitude"].max(), 200
                    )
                    y_fine = (
                        device_result.fit_amplitude
                        * np.sin(np.pi * x_fine * device_result.frequency) ** 2
                        + device_result.offset
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=x_fine,
                            y=y_fine,
                            mode="lines",
                            name="Fit",
                            line=dict(width=3, color=colors[0]),
                        )
                    )

            # Apply layout
            apply_experiment_layout(
                fig,
                title=f"Rabi oscillation : Q{self.physical_qubit}",
                xaxis_title="Drive amplitude (arb. unit)",
                yaxis_title="P(|1⟩)",
                height=400,
                width=600,
            )
            fig.update_yaxes(range=[0, 1])

            # Save and show
            if save_image:
                images_dir = (
                    getattr(self.data_manager, "session_dir", "./images") + "/plots"
                )
                save_plotly_figure(
                    fig,
                    name=f"rabi_{self.physical_qubit}",
                    images_dir=images_dir,
                    width=600,
                    height=400,
                )

            config = get_plotly_config(
                f"rabi_Q{self.physical_qubit}", width=600, height=400
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("⚠️  plotly not available, skipping plot")
        except Exception as e:
            print(f"⚠️  Plot creation failed: {e}")

    def _save_results(self, analysis_result: RabiAnalysisResult):
        """Save analysis results"""
        try:
            saved_path = self.save_experiment_data(
                analysis_result.dataframe.to_dict(orient="records"),
                metadata={
                    "fitting_summary": {
                        device: {
                            "pi_amplitude": result.pi_amplitude,
                            "frequency": result.frequency,
                            "r_squared": result.r_squared,
                        }
                        for device, result in analysis_result.device_results.items()
                    },
                    **analysis_result.metadata,
                },
                experiment_type="rabi",
            )
            print(f"💾 Analysis data saved to: {saved_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save analysis data: {e}")

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
                        if isinstance(logical_qubit, (int, float))
                        else 0
                    ),
                    physical_qubit=(
                        int(physical_qubit)
                        if isinstance(physical_qubit, (int, float))
                        else 0
                    ),
                    rotation_angle=float(amp * np.pi),
                )
            circuit_params.append(param_model.model_dump())

        return circuit_params

    def _transpile_circuits_if_needed(self, circuits, logical_qubit, physical_qubit):
        """Simplified transpilation using Tranqu"""
        try:
            from tranqu import Tranqu

            from ..devices import DeviceInfo

            device_info = DeviceInfo("anemone")
            if not device_info.available:
                print("⚠️  Device info not available, using original circuits")
                return circuits

            tranqu = Tranqu()
            transpiled_circuits = []

            for i, circuit in enumerate(circuits):
                try:
                    initial_layout = {circuit.qubits[logical_qubit]: physical_qubit}
                    result = tranqu.transpile(
                        program=circuit,
                        transpiler_lib="qiskit",
                        program_lib="qiskit",
                        transpiler_options={
                            "basis_gates": ["sx", "x", "rz", "cx"],
                            "optimization_level": 1,
                            "initial_layout": initial_layout,
                        },
                        device=device_info.device_info,
                        device_lib="oqtopus",
                    )
                    transpiled_circuits.append(result.transpiled_program)
                except Exception as e:
                    print(f"⚠️  Circuit {i+1} transpilation failed: {e}")
                    transpiled_circuits.append(circuit)

            print(
                f"✅ Transpiled {len(transpiled_circuits)} circuits to physical qubit {physical_qubit}"
            )
            return transpiled_circuits

        except ImportError:
            print("⚠️  Tranqu not available, using original circuits")
            return circuits
        except Exception as e:
            print(f"⚠️  Transpilation failed: {e}, using original circuits")
            return circuits
