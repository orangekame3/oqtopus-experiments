#!/usr/bin/env python3
"""
Parity Oscillation Experiment Class - Decoherence study of GHZ states
Based on "Decoherence of up to 8-qubit entangled states in a 16-qubit superconducting quantum processor"
by Asier Ozaeta and Peter L McMahon (2019)
"""

import time
from typing import Any

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..core.base_experiment import BaseExperiment

try:
    from qiskit import QuantumCircuit

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class ParityOscillation(BaseExperiment):
    """
    Parity oscillation experiment for studying GHZ state decoherence

    Measures the coherence C(N, τ) of N-qubit GHZ states as a function of:
    - Number of qubits N
    - Delay time τ
    - Rotation phase φ

    The coherence is extracted from parity oscillations amplitude.
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        num_qubits: int = 2,
        delay_us: float = 0.0,
        shots: int = 1000,
        **kwargs
    ):
        """Initialize Parity Oscillation experiment with simplified parameters"""

        super().__init__(experiment_name or "parity_oscillation")

        from ..models.parity_oscillation_models import ParityOscillationParameters

        # Calculate phase points automatically based on paper: 4N+1 points for N-qubit GHZ state
        phase_points = 4 * num_qubits + 1

        # Simplified parameters like other experiments
        self.params = ParityOscillationParameters(
            experiment_name=experiment_name,
            num_qubits_list=[num_qubits],  # Single qubit count
            delays_us=[delay_us],          # Single delay
            phase_points=phase_points,
            no_delay=delay_us == 0.0,      # Auto-detect no_delay mode
            shots_per_circuit=shots,
        )

        # Store simplified parameters
        self.num_qubits = num_qubits
        self.delay_us = delay_us
        self.phase_points = phase_points  # Auto-calculated from num_qubits
        self.no_delay = delay_us == 0.0
        self.shots = shots

        # Backward compatibility for internal use
        self.num_qubits_list = [num_qubits]
        self.delays_us = [delay_us]
        self.default_phase_points = phase_points

        # Handle mitigation options
        no_mitigation = kwargs.get("no_mitigation", False)
        if no_mitigation:
            self.mitigation_options = {}

        print("Parity Oscillation Experiment initialized")
        print(f"Qubits: {self.num_qubits}")
        print(f"Delay (μs): {self.delay_us}")
        print(f"Phase points: {self.phase_points} (4N+1 = 4×{self.num_qubits}+1)")

    def create_ghz_with_delay_and_rotation(
        self,
        num_qubits: int,
        delay_us: float = 0.0,
        phi: float = 0.0,
        no_delay: bool = False,
    ) -> Any:
        """
        Create GHZ state circuit with delay and rotation analysis

        Circuit structure:
        1. Generate N-qubit GHZ state
        2. Apply delay τ (using identity gates)
        3. Apply rotation U(φ) to each qubit
        4. Measure in computational basis

        Args:
            num_qubits: Number of qubits in GHZ state
            delay_us: Delay time in microseconds
            phi: Rotation phase φ
            no_delay: Skip delay insertion (for faster execution)

        Returns:
            Quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for parity oscillation circuits")

        # Create base GHZ state (without measurement)
        qc = QuantumCircuit(num_qubits, num_qubits)

        # Step 1: Generate GHZ state |0...0⟩ + |1...1⟩
        qc.h(0)  # Hadamard on first qubit
        for i in range(1, num_qubits):
            qc.cx(0, i)  # CNOT chain

        # Step 2: Apply delay directly in nanoseconds
        if delay_us > 0 and not no_delay:
            delay_ns = delay_us * 1000  # Convert μs to ns
            for qubit in range(num_qubits):
                qc.delay(delay_ns, qubit)

        # Step 3: Apply rotation U(φ) to each qubit
        # U(φ) = exp(-iφσy/2) rotation around Y-axis
        # Always apply rotation for continuity (even if phi=0)
        for qubit in range(num_qubits):
            # Using U gate: U(θ, φ, λ) where θ=π/2, φ=-φ-π/2, λ=-φ-π/2
            # This implements the rotation from the paper
            u_params = (np.pi / 2, -phi - np.pi / 2, -phi - np.pi / 2)
            qc.u(*u_params, qubit)
            
            # Debug: show U gate parameters for first circuit
            if qubit == 0 and hasattr(self, '_debug_u_params'):
                print(f"🔍 U gate params for φ={phi:.3f}: θ={u_params[0]:.3f}, φ={u_params[1]:.3f}, λ={u_params[2]:.3f}")
        
        # Set debug flag for first call
        if not hasattr(self, '_debug_u_params'):
            self._debug_u_params = True

        # Step 4: Measurement
        qc.measure_all()

        return qc

    def circuits(self, **kwargs) -> list[Any]:
        """
        Create parity oscillation experiment circuits for single qubit count and delay

        Returns:
            List of quantum circuits
        """
        circuits = []
        circuit_metadata = []

        # Use instance parameters (simplified single values)
        num_qubits = self.num_qubits
        delay_us = self.delay_us
        phase_points = self.phase_points
        no_delay = self.no_delay

        # Generate phase values from 0 to π (unified range)
        phase_values = np.linspace(0, np.pi, phase_points)

        for phi in phase_values:
            circuit = self.create_ghz_with_delay_and_rotation(
                num_qubits, delay_us, phi, no_delay
            )
            circuits.append(circuit)
            
            # Debug: show first circuit for comparison
            if len(circuits) <= 2:
                print(f"🔍 Circuit for φ={phi:.3f}:")
                print(circuit.draw())

            # Store metadata for analysis
            circuit_metadata.append(
                {
                    "num_qubits": num_qubits,
                    "delay_us": delay_us,
                    "phi": phi,
                    "circuit_index": len(circuits) - 1,
                }
            )

        # Store metadata for later analysis
        self.circuit_metadata = circuit_metadata

        print(f"Generated {len(circuits)} parity oscillation circuits")
        print(f"N={num_qubits} qubits, τ={delay_us}μs, {phase_points} phase points")
        if no_delay:
            print("⚠️ Delay gates skipped (τ=0 mode)")

        # Store circuits for later use by run() methods
        self._circuits = circuits
        return circuits

    def run(self, backend, shots: int = 1024, **kwargs):
        """
        Run Parity Oscillation experiment with unified interface
        
        Args:
            backend: Backend instance
            shots: Number of shots per circuit
            **kwargs: Additional arguments (num_qubits_list, delays_us, no_delay, etc.)
        """
        # Update shots if provided
        if shots != 1024:
            self.shots = shots

        # Use BaseExperiment's run method with current parameters
        return super().run(backend=backend, shots=shots, **kwargs)

    def calculate_parity(self, counts: dict[str | int, int]) -> float:
        """
        Calculate parity P_even - P_odd from measurement counts

        Args:
            counts: Measurement counts dictionary

        Returns:
            Parity value P_even - P_odd
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0

        even_count = 0
        odd_count = 0

        # Check if we need OQTOPUS decimal-to-binary conversion
        integer_keys = [k for k in counts.keys() if isinstance(k, int)]
        if integer_keys:
            # OQTOPUS decimal format - need to guess num_qubits from data
            max_value = max(integer_keys)
            num_qubits = max_value.bit_length() if max_value > 0 else 1

            # Debug info for OQTOPUS counts format
            print(f"🔍 Parity Raw decimal counts: {dict(counts)}")
            print(f"🔍 Detected {num_qubits} qubits from max value {max_value}")

            # Convert to binary format
            binary_counts = {}
            for key, count in counts.items():
                if isinstance(key, int):
                    binary_key = format(key, f"0{num_qubits}b")
                else:
                    binary_key = str(key)
                binary_counts[binary_key] = count
        else:
            binary_counts = counts

        for bitstring, count in binary_counts.items():
            # Ensure string format
            if not isinstance(bitstring, str):
                bitstring = str(bitstring)

            # Count number of 1s in bitstring
            num_ones = bitstring.count("1")
            if num_ones % 2 == 0:
                even_count += count
            else:
                odd_count += count

        p_even = even_count / total_shots
        p_odd = odd_count / total_shots

        return p_even - p_odd

    def fit_sinusoid(self, phase_values: np.ndarray, parity_values: np.ndarray) -> dict:
        """
        Fit sinusoid to parity oscillations: P(φ) = A*sin(Nφ + θ) + offset

        Args:
            phase_values: Array of phase values φ
            parity_values: Array of corresponding parity values

        Returns:
            Dictionary with fit parameters: amplitude, phase, offset, frequency
        """
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            print("Warning: scipy not available, using simple amplitude estimation")
            # Simple amplitude estimation: max - min
            amplitude = (np.max(parity_values) - np.min(parity_values)) / 2
            return {
                "amplitude": amplitude,
                "phase": 0.0,
                "offset": np.mean(parity_values),
                "frequency": 1.0,
                "r_squared": 0.0,
                "fit_success": False,
            }

        # Determine frequency from number of qubits
        # For N-qubit GHZ state, frequency should be N
        num_qubits = len(phase_values) // 4 - 1  # Approximate from 4N+1 points
        if num_qubits < 1:
            num_qubits = 1

        def sinusoid(phi, amplitude, phase, offset):
            return amplitude * np.sin(num_qubits * phi + phase) + offset

        try:
            # Initial guess
            amplitude_guess = (np.max(parity_values) - np.min(parity_values)) / 2
            offset_guess = np.mean(parity_values)
            phase_guess = 0.0

            popt, pcov = curve_fit(
                sinusoid,
                phase_values,
                parity_values,
                p0=[amplitude_guess, phase_guess, offset_guess],
                maxfev=2000,
            )

            amplitude, phase, offset = popt

            # Calculate R²
            y_pred = sinusoid(phase_values, *popt)
            ss_res = np.sum((parity_values - y_pred) ** 2)
            ss_tot = np.sum((parity_values - np.mean(parity_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                "amplitude": abs(amplitude),  # Coherence C(N,τ)
                "phase": phase,
                "offset": offset,
                "frequency": num_qubits,
                "r_squared": r_squared,
                "fit_success": True,
            }

        except Exception as e:
            print(f"Sinusoid fitting failed: {e}")
            amplitude = (np.max(parity_values) - np.min(parity_values)) / 2
            return {
                "amplitude": amplitude,
                "phase": 0.0,
                "offset": np.mean(parity_values),
                "frequency": num_qubits,
                "r_squared": 0.0,
                "fit_success": False,
            }

    def analyze(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> "pd.DataFrame":
        """
        Analyze parity oscillation results

        Args:
            results: Raw measurement results from quantum devices
            **kwargs: Optional parameters (plot, save_data, save_image)

        Returns:
            DataFrame with coherence data for interface consistency
        """
        plot = kwargs.get("plot", False)
        save_data = kwargs.get("save_data", False)
        save_image = kwargs.get("save_image", False)
        if not hasattr(self, "circuit_metadata"):
            raise ValueError("Circuit metadata not found. Run create_circuits first.")

        analysis_results = {}

        # Flatten all results into single list (no device separation) - like Rabi
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Collect all phase values and parity values from flattened results
        phase_values = []
        parity_values = []

        for i, result in enumerate(all_results):
            if result is None:
                continue
            
            # Check if result has valid counts
            counts = result.get("counts", {})
            if not counts:
                continue

            # Get metadata from OQTOPUS params or fallback to circuit_metadata
            params = result.get("params", {})
            if "phi" in params:
                # Use parameters embedded in OQTOPUS description
                phi = params["phi"]
            elif hasattr(self, "circuit_metadata") and i < len(self.circuit_metadata):
                # Fallback to stored metadata
                phi = self.circuit_metadata[i]["phi"]
            else:
                # Skip if no parameter information available
                continue
                
            parity = self.calculate_parity(counts)
            
            # Debug: show phi vs parity relationship
            print(f"🔍 φ={phi:.3f} → parity={parity:.3f}")

            phase_values.append(phi)
            parity_values.append(parity)

        if len(phase_values) < 5:  # Need sufficient points
            print(f"⚠ Insufficient data points: {len(phase_values)} < 5")
            return pd.DataFrame()

        # Sort by phase for proper fitting
        sorted_indices = np.argsort(phase_values)
        phase_array = np.array(phase_values)[sorted_indices]
        parity_array = np.array(parity_values)[sorted_indices]

        # Calculate coherence as amplitude of parity oscillations
        p_center = np.mean(parity_array)
        amplitude = np.max(np.abs(parity_array - p_center))
        coherence = amplitude

        # Simplified: no fitting, just coherence calculation
        fit_success = True
        r_squared = 0.0

        # Get device name from results (like Rabi does)
        device_name = "unknown"
        if all_results:
            device_name = all_results[0].get("backend", "unknown")

        print(f"N={self.num_qubits}, τ={self.delay_us}μs: C={coherence:.3f}")

        # Create simplified analysis results for DataFrame creation
        analysis_results = {
            device_name: {
                "coherence_data": [{
                    "num_qubits": self.num_qubits,
                    "delay_us": self.delay_us,
                    "coherence": coherence,
                    "fit_r_squared": r_squared,
                    "fit_success": fit_success,
                }]
            }
        }

        # Create DataFrame for interface consistency
        df = self._create_dataframe(analysis_results)

        # Optional plotting (like other experiments)
        if plot and len(phase_values) >= 5:
            self._create_plot(phase_array, parity_array, coherence, save_image)

        return df

    def _create_dataframe(self, analysis_results: dict[str, Any]) -> "pd.DataFrame":
        """
        Create DataFrame from analysis results for interface consistency
        
        Args:
            analysis_results: Raw analysis results dictionary
            
        Returns:
            DataFrame with coherence data
        """
        if not PANDAS_AVAILABLE:
            print("Warning: pandas not available, returning empty DataFrame")
            return None

        df_data = []

        for device, device_results in analysis_results.items():
            coherence_data = device_results.get("coherence_data", [])

            for data in coherence_data:
                df_data.append({
                    "device": device,
                    "num_qubits": data["num_qubits"],
                    "delay_us": data["delay_us"],
                    "coherence": data["coherence"],
                    "fit_r_squared": data["fit_r_squared"],
                    "fit_success": data["fit_success"],
                })

        return pd.DataFrame(df_data) if df_data else pd.DataFrame()

    def _create_plot(self, phase_array, parity_array, coherence, save_image=False):
        """
        Create parity oscillation plot using plotly (consistent with other experiments)
        
        Args:
            phase_array: Phase values
            parity_array: Parity values
            coherence: Calculated coherence
            save_image: Whether to save the plot
        """
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

            # Convert phase to π units for display
            phase_pi = phase_array / np.pi

            # Data points with lines (like other experiments)
            fig.add_trace(
                go.Scatter(
                    x=phase_pi,
                    y=parity_array,
                    mode="markers+lines",
                    name="Data",
                    marker=dict(
                        size=7,
                        color=colors[1],  # Green like Rabi
                        line=dict(width=1, color="white")
                    ),
                    line=dict(width=2, color=colors[1]),
                    showlegend=True
                )
            )

            # Removed fitting - just show data points

            # Apply standard experiment layout
            apply_experiment_layout(
                fig,
                title=f"Parity Oscillation: N={self.num_qubits}, τ={self.delay_us}μs",
                xaxis_title="Phase φ/π",
                yaxis_title="Parity (P_even - P_odd)"
            )

            # Custom x-axis ticks in π units (0 to π range)
            fig.update_xaxes(
                tickmode='array',
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['0', 'π/4', 'π/2', '3π/4', 'π'],
                range=[0, 1]
            )

            # Fix y-axis scale to -1 to +1 with padding like CHSH experiment
            fig.update_yaxes(range=[-1.1, 1.1])

            # Add coherence annotation
            fig.add_annotation(
                x=0.98,
                y=0.02,
                text=f"N={self.num_qubits} qubits<br>τ={self.delay_us}μs<br>Coherence = {coherence:.3f}",
                showarrow=False,
                xref="x domain",
                yref="y domain",
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )

            # Save if requested
            if save_image:
                images_dir = getattr(self, 'data_manager', None)
                if images_dir and hasattr(images_dir, 'session_dir'):
                    images_dir = f"{images_dir.session_dir}/plots"
                else:
                    images_dir = "."

                save_plotly_figure(
                    fig,
                    name=f"parity_oscillation_N{self.num_qubits}_tau{self.delay_us}us",
                    images_dir=images_dir,
                )

            # Show plot with standard config
            config = get_plotly_config(
                f"parity_oscillation_N{self.num_qubits}", width=800, height=500
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("plotly not available, skipping plot")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def save_experiment_data(
        self, results: dict[str, Any], metadata: dict[str, Any] = None
    ) -> str:
        """
        Save parity oscillation experiment data

        Args:
            results: Analyzed results
            metadata: Additional metadata

        Returns:
            Save path
        """
        save_data = {
            "experiment_type": "ParityOscillation",
            "results": results,
            "parameters": {
                "num_qubits_list": getattr(self, "default_num_qubits", []),
                "delays_us": getattr(self, "default_delays_us", []),
                "phase_points": getattr(self, "default_phase_points", 21),
            },
            "analysis_timestamp": time.time(),
            "metadata": metadata or {},
        }

        return self.data_manager.save_data(save_data, "parity_oscillation_results")

    def generate_parityoscillation_plot(
        self, results: dict[str, Any], save_plot: bool = True, show_plot: bool = False
    ) -> str | None:
        """
        Generate parity oscillation experiment plot following quantumlib standards

        Args:
            results: Complete experiment results
            save_plot: Save plot to file
            show_plot: Display plot interactively

        Returns:
            Plot file path if saved, None otherwise
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        analysis = results.get("analysis", {})
        if not analysis:
            print("No analysis results for plotting")
            return None

        # Get device information for plot labeling
        device_names = list(analysis.keys())
        device_name = device_names[0] if device_names else "unknown"

        # Create figure with subplots for different delay times
        device_data = list(analysis.values())[0]  # Get first device data
        oscillation_data = device_data.get("parity_oscillations", [])

        if not oscillation_data:
            print("No parity oscillation data for plotting")
            return None

        # Group by delay time
        delay_groups: dict[float, list[Any]] = {}
        for data in oscillation_data:
            delay = data["delay_us"]
            if delay not in delay_groups:
                delay_groups[delay] = []
            delay_groups[delay].append(data)

        n_delays = len(delay_groups)
        if n_delays == 0:
            return None

        # Create subplots
        if n_delays == 1:
            # Single subplot for one delay
            fig, ax = plt.subplots(figsize=(7, 4))
            axes = [ax]
        else:
            # Multiple subplots
            fig, axes = plt.subplots(
                nrows=(n_delays + 1) // 2,
                ncols=2,
                figsize=(14, 4 * ((n_delays + 1) // 2)),
            )
            if n_delays == 2:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

        # Colors for different qubit counts (matching existing style)
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (delay_us, delay_data) in enumerate(delay_groups.items()):
            ax = axes[i] if n_delays > 1 else axes[0]

            for j, data in enumerate(delay_data):
                phase_values = np.array(data["phase_values"])
                parity_values = np.array(data["parity_values"])
                num_qubits = data["num_qubits"]

                color = colors[j % len(colors)]
                ax.plot(
                    phase_values,
                    parity_values,
                    "o-",
                    color=color,
                    label=f"N={num_qubits}",
                    linewidth=2,
                    markersize=6,
                    alpha=0.8,
                )

                # Fit line removed for simplicity - show only actual data points

            # Formatting (following existing style)
            ax.set_xlabel("Phase φ [rad]", fontsize=12)
            ax.set_ylabel("Parity (P_even - P_odd)", fontsize=12)
            ax.set_title(
                f"Parity Oscillations (τ = {delay_us} μs) - {device_name}",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_xlim(0, np.pi)

            # X-axis labels in π units (following existing style)
            ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
            ax.set_xticklabels(["0", "π/4", "π/2", "3π/4", "π"])

        # Hide unused subplots
        for i in range(n_delays, len(axes)):
            axes[i].set_visible(False)

        # Main title with device information
        fig.suptitle(
            f"OQTOPUS Experiments Parity Oscillation (GHZ Decoherence) Experiment - {device_name}",
            fontsize=16,
            fontweight="bold",
        )

        plot_filename = None
        if save_plot:
            plt.tight_layout()
            plot_filename = f"parity_oscillation_plot_{device_name}_{self.experiment_name}_{int(time.time())}.png"

            # Save to experiment results directory (following existing pattern)
            if hasattr(self, "data_manager") and hasattr(
                self.data_manager, "session_dir"
            ):
                plot_path = f"{self.data_manager.session_dir}/plots/{plot_filename}"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved: {plot_path}")
                plot_filename = plot_path  # Return full path
            else:
                # Fallback: save in current directory
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                print(f"⚠️ Plot saved to current directory: {plot_filename}")
                print("   (data_manager not available)")

        # Display plot
        if show_plot:
            try:
                plt.show()
            except Exception:
                pass

        plt.close()
        return plot_filename

    def create_plots(
        self, analysis_results: dict[str, Any], save_dir: str = None
    ) -> None:
        """
        Create plots for parity oscillation experiment results (legacy method)

        Args:
            analysis_results: Results from analyze_results()
            save_dir: Directory to save plots (optional)
        """
        try:
            # Use the unified plot generation method
            self.generate_parityoscillation_plot(
                {"analysis": analysis_results}, save_plot=True, show_plot=False
            )
        except ImportError:
            print("Cannot create plots: matplotlib not available")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def save_complete_experiment_data(self, results: dict[str, Any]) -> str:
        """
        Save complete experiment data including plots and summary
        Required by the unified CLI framework

        Args:
            results: Complete experiment results

        Returns:
            Path to main results file
        """
        # Save main experiment data
        main_file = self.save_experiment_data(results["analysis"])

        # Generate and save plots using unified method
        try:
            plot_file = self.generate_parityoscillation_plot(
                results, save_plot=True, show_plot=False
            )
        except Exception as e:
            print(f"⚠️ Plot generation failed: {e}")
            plot_file = None

        # Create experiment summary
        summary = self._create_experiment_summary(results)
        summary_file = self.data_manager.save_data(summary, "experiment_summary")

        print("📊 Complete parity oscillation experiment data saved:")
        print(f"  • Main results: {main_file}")
        print(f"  • Plots: {plot_file if plot_file else 'Not generated'}")
        print(f"  • Summary: {summary_file}")

        return main_file

    def _create_experiment_summary(self, results: dict[str, Any]) -> dict:
        """
        Create experiment summary for unified framework

        Args:
            results: Complete experiment results

        Returns:
            Summary dictionary
        """
        analysis = results.get("analysis", {})

        # Count successful measurements
        total_measurements = 0
        successful_measurements = 0

        for _, device_results in analysis.items():
            coherence_data = device_results.get("coherence_data", [])
            total_measurements += len(coherence_data)
            successful_measurements += sum(
                1 for c in coherence_data if c.get("fit_success", False)
            )

        # Extract key findings
        coherence_summary = {}
        for device, device_results in analysis.items():
            coherence_data = device_results.get("coherence_data", [])
            if coherence_data:
                coherence_summary[device] = {
                    "measurements": len(coherence_data),
                    "successful_fits": sum(
                        1 for c in coherence_data if c.get("fit_success", False)
                    ),
                    "qubit_counts": sorted({c["num_qubits"] for c in coherence_data}),
                    "max_coherence": max(c["coherence"] for c in coherence_data),
                    "min_coherence": min(c["coherence"] for c in coherence_data),
                }

        return {
            "experiment_type": "ParityOscillation",
            "timestamp": time.time(),
            "parameters": getattr(self, "experiment_params", {}),
            "total_measurements": total_measurements,
            "successful_measurements": successful_measurements,
            "success_rate": (
                successful_measurements / total_measurements
                if total_measurements > 0
                else 0
            ),
            "devices": list(analysis.keys()),
            "coherence_summary": coherence_summary,
            "key_findings": self._extract_key_findings(analysis),
        }

    def _extract_key_findings(self, analysis: dict[str, Any]) -> dict:
        """
        Extract key scientific findings from the analysis

        Args:
            analysis: Analysis results

        Returns:
            Key findings dictionary
        """
        findings = {}

        for device, device_results in analysis.items():
            coherence_data = device_results.get("coherence_data", [])
            if not coherence_data:
                continue

            # Group by qubit count to analyze scaling
            qubit_groups: dict[int, list[Any]] = {}
            for data in coherence_data:
                n = data["num_qubits"]
                if n not in qubit_groups:
                    qubit_groups[n] = []
                qubit_groups[n].append(data)

            # Analyze initial coherence scaling
            initial_coherences = {}
            for n, group in qubit_groups.items():
                # Find minimum delay (closest to τ=0)
                min_delay_data = min(group, key=lambda x: x["delay_us"])
                initial_coherences[n] = min_delay_data["coherence"]

            if len(initial_coherences) >= 2:
                # Simple linear fit slope estimation
                qubits = list(initial_coherences.keys())
                coherences = list(initial_coherences.values())

                if len(qubits) >= 2:
                    import numpy as np

                    slope = np.polyfit(qubits, coherences, 1)[0]

                    findings[device] = {
                        "initial_coherence_scaling": {
                            "slope": slope,
                            "interpretation": (
                                "Linear decrease"
                                if slope < -0.05
                                else "Approximately constant"
                            ),
                        },
                        "coherence_range": {
                            "max": max(coherences),
                            "min": min(coherences),
                        },
                        "tested_qubit_counts": qubits,
                    }

        return findings

    def display_results(self, results: dict[str, Any], use_rich: bool = True) -> None:
        """
        Display parity oscillation experiment results in formatted table

        Args:
            results: Complete experiment results
            use_rich: Use rich formatting if available
        """
        analysis = results.get("analysis", {})

        if not analysis:
            print("No analysis results found")
            return

        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(
                    title="Parity Oscillation (GHZ Decoherence) Results",
                    show_header=True,
                    header_style="bold blue",
                )
                table.add_column("Device", style="cyan")
                table.add_column("N (Qubits)", justify="right")
                table.add_column("τ (μs)", justify="right")
                table.add_column("C(N,τ)", justify="right")
                table.add_column("R²", justify="right")
                table.add_column("Fit Success", justify="center")

                for device, device_results in analysis.items():
                    coherence_data = device_results.get("coherence_data", [])

                    if not coherence_data:
                        table.add_row(device, "—", "—", "—", "—", "❌")
                        continue

                    # Sort by qubit count, then by delay
                    sorted_data = sorted(
                        coherence_data, key=lambda x: (x["num_qubits"], x["delay_us"])
                    )

                    for i, data in enumerate(sorted_data):
                        device_name = device if i == 0 else ""

                        n_qubits = str(data["num_qubits"])
                        delay = f"{data['delay_us']:.1f}"
                        coherence = f"{data['coherence']:.3f}"
                        r_squared = f"{data['fit_r_squared']:.3f}"
                        fit_success = "✓" if data.get("fit_success", False) else "❌"

                        table.add_row(
                            device_name,
                            n_qubits,
                            delay,
                            coherence,
                            r_squared,
                            fit_success,
                        )

                console.print(table)

                # Additional summary
                for device, device_results in analysis.items():
                    coherence_data = device_results.get("coherence_data", [])
                    if coherence_data:
                        successful_fits = sum(
                            1 for c in coherence_data if c.get("fit_success", False)
                        )
                        total_measurements = len(coherence_data)
                        success_rate = successful_fits / total_measurements * 100

                        console.print(f"\n[bold cyan]{device}[/bold cyan] Summary:")
                        console.print(f"  • Measurements: {total_measurements}")
                        console.print(
                            f"  • Successful fits: {successful_fits} ({success_rate:.1f}%)"
                        )

                        # Show coherence range
                        coherences = [
                            c["coherence"]
                            for c in coherence_data
                            if c.get("fit_success", False)
                        ]
                        if coherences:
                            console.print(
                                f"  • Coherence range: {min(coherences):.3f} - {max(coherences):.3f}"
                            )

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to plain text
            print("=== Parity Oscillation (GHZ Decoherence) Results ===")

            for device, device_results in analysis.items():
                print(f"\n{device} Results:")
                print("N\tτ(μs)\tC(N,τ)\tR²\tFit")
                print("-" * 40)

                coherence_data = device_results.get("coherence_data", [])
                if not coherence_data:
                    print("No coherence data available")
                    continue

                # Sort by qubit count, then by delay
                sorted_data = sorted(
                    coherence_data, key=lambda x: (x["num_qubits"], x["delay_us"])
                )

                for data in sorted_data:
                    fit_symbol = "✓" if data.get("fit_success", False) else "✗"
                    print(
                        f"{data['num_qubits']}\t"
                        f"{data['delay_us']:.1f}\t"
                        f"{data['coherence']:.3f}\t"
                        f"{data['fit_r_squared']:.3f}\t"
                        f"{fit_symbol}"
                    )

                # Summary
                successful_fits = sum(
                    1 for c in coherence_data if c.get("fit_success", False)
                )
                total_measurements = len(coherence_data)
                success_rate = successful_fits / total_measurements * 100

                print("\nSummary:")
                print(f"  Measurements: {total_measurements}")
                print(f"  Successful fits: {successful_fits} ({success_rate:.1f}%)")

                coherences = [
                    c["coherence"]
                    for c in coherence_data
                    if c.get("fit_success", False)
                ]
                if coherences:
                    print(
                        f"  Coherence range: {min(coherences):.3f} - {max(coherences):.3f}"
                    )

    def _get_circuit_params(self) -> list[dict[str, Any]] | None:
        """Get circuit parameters for OQTOPUS description embedding"""
        if not hasattr(self, "circuit_metadata"):
            return None
        
        # Return the metadata as circuit parameters
        return [metadata.copy() for metadata in self.circuit_metadata]
