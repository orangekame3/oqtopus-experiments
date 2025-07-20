#!/usr/bin/env python3
"""
Rabi Experiment Class - Simplified Rabi oscillation experiment
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ..core.base_experiment import BaseExperiment
from ..models.circuit_collection import CircuitCollection


class Rabi(BaseExperiment):
    """
    Rabi oscillation experiment

    Creates circuits with varying rotation amplitudes and analyzes oscillation patterns.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        physical_qubit: int = 0,
        amplitude_points: int = 10,
        max_amplitude: float = 2.0,
    ):
        super().__init__(experiment_name or "rabi_experiment")
        self.physical_qubit = physical_qubit
        self.amplitude_points = amplitude_points
        self.max_amplitude = max_amplitude

    def analyze(
        self, results: dict[str, list[dict[str, Any]]], **kwargs: Any
    ) -> pd.DataFrame:
        """
        Analyze Rabi experiment results using embedded parameters

        Args:
            results: Raw measurement results with embedded parameters
            **kwargs: Optional parameters (plot, save_data, save_image)

        Returns:
            DataFrame with π-pulse amplitude estimates and fitting results
        """
        # Extract optional parameters
        plot = kwargs.get("plot", False)
        save_data = kwargs.get("save_data", False)
        save_image = kwargs.get("save_image", False)
        if not results:
            return pd.DataFrame()  # Return empty DataFrame instead of dict

        analysis_results: dict[str, dict[str, Any]] = {}

        # Handle both device-based and circuit-based result structures
        if any(key.startswith("circuit_") for key in results.keys()):
            # Circuit-based structure from run_parallel: {circuit_0: [result], circuit_1: [result]}
            device_results = []
            for circuit_key in sorted(results.keys()):
                if results[circuit_key]:
                    device_results.extend(results[circuit_key])

            # Process all results for a single device (extract device name from results)
            device_name = "unknown"
            if device_results and "backend" in device_results[0]:
                device_name = device_results[0]["backend"]
            elif hasattr(self, "_last_backend_device"):
                device_name = self._last_backend_device

            result_groups = {device_name: device_results}
        else:
            # Device-based structure: {device_name: [results]}
            result_groups = results

        for device, device_results in result_groups.items():
            if not device_results:
                continue

            # Extract amplitudes and probabilities from results with embedded params
            amplitudes: list[float] = []
            probabilities: list[float] = []

            for result in device_results:
                # Get amplitude from embedded params (preferred) or fallback to experiment params
                amplitude = None
                if "params" in result and "amplitude" in result["params"]:
                    amplitude = result["params"]["amplitude"]
                elif (
                    hasattr(self, "experiment_params")
                    and "amplitudes" in self.experiment_params
                ):
                    # Fallback: use index-based lookup from experiment params
                    circuit_idx = result.get("params", {}).get(
                        "circuit_index", len(amplitudes)
                    )
                    if circuit_idx < len(self.experiment_params["amplitudes"]):
                        amplitude = self.experiment_params["amplitudes"][circuit_idx]

                if amplitude is None:
                    print(
                        f"⚠️  Warning: Could not determine amplitude for result, skipping"
                    )
                    continue

                amplitudes.append(amplitude)

                # Extract probability of measuring |1⟩
                counts = result.get("counts", {})
                total = sum(counts.values())
                if total > 0:
                    prob_1 = counts.get("1", counts.get(1, 0)) / total
                    probabilities.append(prob_1)
                else:
                    probabilities.append(0.0)

            if not amplitudes:
                print(f"⚠️  No valid amplitude parameters found for {device}")
                continue

            amplitudes_array = np.array(amplitudes)
            probabilities_array = np.array(probabilities)

            # Sort by amplitude for proper fitting
            sort_indices = np.argsort(amplitudes_array)
            amplitudes_array = amplitudes_array[sort_indices]
            probabilities_array = probabilities_array[sort_indices]

            print(
                f"📋 Data range: amp {amplitudes_array[0]:.3f}-{amplitudes_array[-1]:.3f}, prob {probabilities_array.min():.3f}-{probabilities_array.max():.3f}"
            )

            # Simple Rabi fitting: P = A * sin²(π * amp * freq) + offset
            try:

                def rabi_func(amp, amplitude, frequency, offset):
                    return amplitude * np.sin(np.pi * amp * frequency) ** 2 + offset

                # Estimate better initial guess from data
                max_prob = np.max(probabilities_array)
                min_prob = np.min(probabilities_array)
                amplitude_guess = max_prob - min_prob
                offset_guess = min_prob

                # Find peaks to estimate frequency better
                # Look for local maxima that are significantly above baseline

                threshold = min_prob + 0.6 * amplitude_guess
                peaks, _ = find_peaks(probabilities_array, height=threshold, distance=2)

                if len(peaks) >= 2:
                    # Use spacing between peaks to estimate frequency
                    # Distance between consecutive π-pulses should be 1/freq
                    peak_amps = amplitudes_array[peaks]
                    avg_spacing = np.mean(np.diff(peak_amps))
                    frequency_guess = 1.0 / avg_spacing  # π-pulse spacing = 1/freq
                    print(f"🔍 Found {len(peaks)} peaks at amplitudes: {peak_amps}")
                    print(
                        f"🔍 Average spacing: {avg_spacing:.3f} → freq guess: {frequency_guess:.3f}"
                    )
                elif len(peaks) == 1 and amplitudes[peaks[0]] > 0:
                    # Single peak: assume it's the first π-pulse
                    first_pi_amp = amplitudes_array[peaks[0]]
                    frequency_guess = 0.5 / first_pi_amp
                    print(
                        f"🔍 Single peak at amplitude: {first_pi_amp:.3f} → freq guess: {frequency_guess:.3f}"
                    )
                else:
                    # Fallback: estimate from data range and visible oscillations
                    # If we see data up to amplitude 4 with ~3 cycles, freq ≈ 3/4 = 0.75
                    max_amp = amplitudes_array[-1]
                    visible_cycles = max_amp * 0.75  # rough estimate
                    frequency_guess = visible_cycles / max_amp
                    print(f"🔍 Fallback frequency guess: {frequency_guess:.3f}")

                # Ensure reasonable bounds
                amplitude_guess = max(0.1, min(1.0, amplitude_guess))
                frequency_guess = max(
                    0.3, min(2.0, frequency_guess)
                )  # Allow higher freq
                offset_guess = max(0.0, min(0.5, offset_guess))

                initial_guess = [amplitude_guess, frequency_guess, offset_guess]
                print(
                    f"📊 Initial guess: amp={amplitude_guess:.3f}, freq={frequency_guess:.3f}, offset={offset_guess:.3f}"
                )
                # Set reasonable bounds for fitting parameters
                # amplitude: [0, 1], frequency: [0.1, 5], offset: [0, 1]
                bounds = ([0, 0.1, 0], [1, 5, 1])

                popt, pcov = curve_fit(
                    rabi_func,
                    amplitudes_array,
                    probabilities_array,
                    p0=initial_guess,
                    bounds=bounds,
                    maxfev=2000,
                )

                fitted_amplitude, fitted_freq, fitted_offset = popt
                pi_amplitude = 0.5 / fitted_freq  # First π pulse

                # Calculate R² for fit quality
                y_pred = rabi_func(amplitudes_array, *popt)
                ss_res = np.sum((probabilities_array - y_pred) ** 2)
                ss_tot = np.sum(
                    (probabilities_array - np.mean(probabilities_array)) ** 2
                )
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                analysis_results[device] = {
                    "pi_amplitude": float(pi_amplitude),
                    "frequency": float(fitted_freq),
                    "fit_amplitude": float(fitted_amplitude),
                    "offset": float(fitted_offset),
                    "r_squared": float(r_squared),
                    "amplitudes": amplitudes_array.tolist(),
                    "probabilities": probabilities_array.tolist(),
                }

                print(
                    f"📊 {device}: π-pulse amp = {pi_amplitude:.3f}, freq = {fitted_freq:.3f}, R² = {r_squared:.3f} (fitted: A={fitted_amplitude:.3f}, offset={fitted_offset:.3f})"
                )

            except Exception as e:
                print(f"⚠️  Fitting failed for {device}: {e}")
                print(
                    f"    Data range: amp {amplitudes_array.min():.3f}-{amplitudes_array.max():.3f}, prob {probabilities_array.min():.3f}-{probabilities_array.max():.3f}"
                )

                # Try simple linear fit as fallback
                try:
                    from scipy import stats

                    slope, intercept, r_value, _, _ = stats.linregress(
                        amplitudes_array, probabilities_array
                    )
                    print(
                        f"    Fallback linear fit: slope={slope:.3f}, R²={r_value**2:.3f}"
                    )

                    analysis_results[device] = {
                        "pi_amplitude": 1.0,  # Default estimate
                        "frequency": 0.5,
                        "fit_amplitude": float(abs(slope * amplitudes_array.max())),
                        "offset": float(intercept),
                        "r_squared": float(r_value**2),
                        "error_info": f"Rabi fit failed, used linear: {str(e)}",
                        "amplitudes": amplitudes_array.tolist(),
                        "probabilities": probabilities_array.tolist(),
                    }
                except Exception as e2:
                    print(f"    Even linear fit failed: {e2}")
                    analysis_results[device] = {
                        "pi_amplitude": 1.0,
                        "frequency": 0.5,
                        "fit_amplitude": 0.5,
                        "offset": 0.5,
                        "r_squared": 0.0,
                        "error_info": f"All fitting failed: {str(e)}",
                        "amplitudes": amplitudes_array.tolist(),
                        "probabilities": probabilities_array.tolist(),
                    }

        # Create DataFrame from all analysis results
        df_data = []
        for device, device_result in analysis_results.items():
            if "amplitudes" in device_result and "probabilities" in device_result:
                device_amplitudes = device_result["amplitudes"]
                device_probabilities = device_result["probabilities"]

                for amp, prob in zip(device_amplitudes, device_probabilities):
                    df_data.append(
                        {
                            "device": device,
                            "amplitude": amp,
                            "probability": prob,
                            "pi_amplitude": device_result.get("pi_amplitude", None),
                            "frequency": device_result.get("frequency", None),
                            "r_squared": device_result.get("r_squared", None),
                        }
                    )

        if not df_data:
            print("⚠️  No data available for DataFrame, returning empty DataFrame")
            return pd.DataFrame()

        df = pd.DataFrame(df_data)
        print(
            f"📊 Rabi DataFrame: {len(df)} data points from {len(analysis_results)} devices"
        )

        # Create interactive plot if requested
        if plot:
            self._create_simple_plot(df, analysis_results, save_image=save_image)

        # Save data if requested
        if save_data:
            try:
                # Simply save the DataFrame directly - much cleaner!
                saved_path = self.save_experiment_data(
                    df.to_dict(orient="records"),
                    metadata={
                        "fitting_summary": {
                            device: {
                                "pi_amplitude": float(result.get("pi_amplitude", 0)),
                                "frequency": float(result.get("frequency", 0)),
                                "r_squared": float(result.get("r_squared", 0)),
                            }
                            for device, result in analysis_results.items()
                        },
                    },
                    experiment_type="rabi",
                )
                print(f"💾 Analysis data saved to: {saved_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not save analysis data: {e}")

        return df

    def circuits(self, **kwargs: Any) -> list[Any]:
        """Create Rabi oscillation circuits"""
        # Use constructor defaults if not provided
        amplitude_points = kwargs.get("amplitude_points", self.amplitude_points)
        max_amplitude = kwargs.get("max_amplitude", self.max_amplitude)
        qubit = kwargs.get("qubit", self.physical_qubit)

        amplitudes = np.linspace(0, max_amplitude, amplitude_points)

        circuits = []
        # Always create circuits with logical qubits first (starting from 0)
        logical_qubit = 0

        for amplitude in amplitudes:
            qc = QuantumCircuit(1, 1)
            if amplitude > 0:
                qc.rx(amplitude * np.pi, logical_qubit)
            qc.measure(logical_qubit, 0)
            circuits.append(qc)

        # Store parameters for analysis
        self.experiment_params = {
            "amplitudes": amplitudes,
            "logical_qubit": logical_qubit,
            "physical_qubit": qubit,  # Target physical qubit
        }

        print(
            f"Created {len(circuits)} Rabi circuits (amplitude range: {amplitudes[0]:.3f} - {amplitudes[-1]:.3f})"
        )

        # If physical qubit specified, perform transpilation
        if qubit is not None:
            print(f"Physical qubit {qubit} specified, performing transpilation...")
            circuits = self._transpile_circuits_if_needed(
                circuits, logical_qubit, qubit
            )
        else:
            print(
                "Rabi circuit structure: |0⟩ → RX(amp·π) → measure (expected: oscillation with amp)"
            )

        circuit_collection = CircuitCollection(circuits)
        self._circuits = circuit_collection
        return circuits  # Return list instead of CircuitCollection for compatibility

    def _get_circuit_params(self) -> Optional[list[dict[str, Any]]]:
        """Get parameters for each circuit for OQTOPUS description field"""
        if not hasattr(self, "experiment_params"):
            return None

        amplitudes = self.experiment_params["amplitudes"]
        logical_qubit = self.experiment_params.get("logical_qubit", 0)
        physical_qubit = self.experiment_params.get("physical_qubit", logical_qubit)

        return [
            {
                "experiment": "rabi",
                "amplitude": float(amp),
                "logical_qubit": logical_qubit,
                "physical_qubit": physical_qubit,
                "rotation_angle": float(amp * np.pi),
            }
            for amp in amplitudes
        ]

    def _transpile_circuits_if_needed(self, circuits, logical_qubit, physical_qubit):
        """
        Transpile circuits using available backend for physical qubit mapping

        Args:
            circuits: List of circuits to transpile
            logical_qubit: Source logical qubit index
            physical_qubit: Target physical qubit index

        Returns:
            Transpiled circuits or original if transpilation fails
        """
        try:
            from tranqu import Tranqu

            from ..devices import DeviceInfo

            # Get device info for the physical qubit
            device_info = DeviceInfo("anemone")  # Default to anemone for now
            if not device_info.available:
                print("⚠️  Device info not available, using original circuits")
                return circuits

            tranqu = Tranqu()
            transpiled_circuits = []

            for i, circuit in enumerate(circuits):
                try:
                    # Create initial layout mapping logical to physical qubit
                    initial_layout = {circuit.qubits[logical_qubit]: physical_qubit}

                    # Transpile with physical qubit mapping
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
                    print(f"⚠️  Circuit {i+1} transpilation failed: {e}, using original")
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

    def _create_simple_plot(self, df, analysis_results, save_image: bool = False):
        """Create simple Rabi oscillation plot using visualization utilities"""
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

            # Plot data for each device
            for device in df["device"].unique():
                device_df = df[df["device"] == device]
                device_results = analysis_results.get(device, {})

                # Add data points
                fig.add_trace(
                    go.Scatter(
                        x=device_df["amplitude"],
                        y=device_df["probability"],
                        mode="markers",
                        name="Data",
                        marker=dict(size=8, color=colors[1]),
                    )
                )

                # Add fitted curve if available
                if "pi_amplitude" in device_results:
                    pi_amp = device_results.get("pi_amplitude", 0.5)
                    freq = device_results.get("frequency", 1.0)
                    fit_amp = device_results.get("fit_amplitude", 0.5)
                    offset = device_results.get("offset", 0.5)

                    import numpy as np

                    x_fine = np.linspace(
                        device_df["amplitude"].min(), device_df["amplitude"].max(), 200
                    )
                    y_fine = fit_amp * np.sin(np.pi * x_fine * freq) ** 2 + offset

                    fig.add_trace(
                        go.Scatter(
                            x=x_fine,
                            y=y_fine,
                            mode="lines",
                            name="Fit",
                            line=dict(width=3, color=colors[0]),
                        )
                    )

            # Apply standard layout
            physical_qubit = getattr(self, "physical_qubit", "unknown")
            apply_experiment_layout(
                fig,
                title=f"Rabi oscillation : Q{physical_qubit}",
                xaxis_title="Drive amplitude (arb. unit)",
                yaxis_title="P(|1⟩)",
                height=400,
                width=600,
            )

            fig.update_yaxes(range=[0, 1])  # Probability range

            # Save image if requested
            if save_image:
                images_dir = "./images"
                if hasattr(self, "data_manager") and hasattr(
                    self.data_manager, "session_dir"
                ):
                    images_dir = f"{self.data_manager.session_dir}/plots"

                save_plotly_figure(
                    fig,
                    name=f"rabi_{physical_qubit}",
                    images_dir=images_dir,
                    width=600,
                    height=400,
                )

            # Show plot
            config = get_plotly_config(f"rabi_Q{physical_qubit}", width=600, height=400)
            show_plotly_figure(fig, config)

        except ImportError:
            print("⚠️  plotly not available, skipping interactive plot")
        except Exception as e:
            print(f"⚠️  Plot creation failed: {e}")
