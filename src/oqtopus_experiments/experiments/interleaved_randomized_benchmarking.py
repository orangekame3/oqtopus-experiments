#!/usr/bin/env python3
"""
Interleaved Randomized Benchmarking Experiment

This module provides a specialized class for interleaved randomized benchmarking,
which allows for gate-specific error characterization by comparing standard RB
with interleaved RB that includes a target gate between each Clifford.
"""

from typing import Any

import pandas as pd

from ..models.results import ExperimentResult
from .randomized_benchmarking import RandomizedBenchmarking


class InterleavedRandomizedBenchmarking(RandomizedBenchmarking):
    """
    Interleaved Randomized Benchmarking experiment for gate-specific error characterization.

    This class runs both standard and interleaved RB experiments, compares their results,
    and calculates gate-specific fidelity metrics.
    """

    def __init__(
        self,
        experiment_name: str,
        physical_qubit: int = 0,
        max_sequence_length: int = 128,
        num_lengths: int = 8,
        num_samples: int = 30,
        interleaved_gate: str = "x",
        **kwargs,
    ):
        """
        Initialize Interleaved Randomized Benchmarking experiment.

        Args:
            experiment_name: Name for this experiment
            physical_qubit: Physical qubit index to use
            max_sequence_length: Maximum Clifford sequence length
            num_lengths: Number of different sequence lengths to test
            num_samples: Number of random sequences per length
            interleaved_gate: Gate to interleave (e.g., 'x', 'y', 'z', 'h')
            **kwargs: Additional arguments passed to RandomizedBenchmarking
        """
        # Initialize base RandomizedBenchmarking with interleaved type
        super().__init__(
            experiment_name=experiment_name,
            physical_qubit=physical_qubit,
            max_sequence_length=max_sequence_length,
            num_lengths=num_lengths,
            num_samples=num_samples,
            rb_type="interleaved",
            interleaved_gate=interleaved_gate,
            **kwargs,
        )

        self.interleaved_gate = interleaved_gate

        # Create standard RB experiment for comparison
        self.standard_rb = RandomizedBenchmarking(
            experiment_name=f"{experiment_name}_standard",
            physical_qubit=physical_qubit,
            max_sequence_length=max_sequence_length,
            num_lengths=num_lengths,
            num_samples=num_samples,
            rb_type="standard",
            **kwargs,
        )

        # Store results for analysis
        self.standard_result: ExperimentResult | None = None
        self.standard_df: pd.DataFrame | None = None

    def run(self, backend, shots: int = 1000, **kwargs):
        """
        Run both standard and interleaved RB experiments.

        Args:
            backend: Quantum backend to run on
            shots: Number of shots per circuit
            **kwargs: Additional arguments for experiment execution

        Returns:
            ExperimentResult for the interleaved RB experiment
        """
        print(f"Running Interleaved RB for {self.interleaved_gate} gate...")
        print("=" * 60)

        # Run standard RB first
        print("1. Running Standard RB (reference)...")
        self.standard_result = self.standard_rb.run(
            backend=backend, shots=shots, **kwargs
        )
        self.standard_df = self.standard_result.analyze(
            plot=False, save_data=True, save_image=False
        )

        # Run interleaved RB (this instance)
        print("\n2. Running Interleaved RB...")
        interleaved_result = super().run(backend=backend, shots=shots, **kwargs)

        return interleaved_result

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Analyze interleaved RB results and calculate gate fidelity.

        Args:
            results: Experiment results dictionary
            plot: Whether to create plots
            save_data: Whether to save analysis data
            save_image: Whether to save plot images
            **kwargs: Additional arguments

        Returns:
            DataFrame with interleaved RB analysis and gate fidelity metrics
        """
        # First get the standard interleaved RB analysis
        interleaved_df = super().analyze(
            results, plot=False, save_data=save_data, save_image=False, **kwargs
        )

        if self.standard_df is None:
            raise ValueError("Must run standard RB first. Call run() method.")

        assert self.standard_df is not None  # Type narrowing

        # Calculate gate fidelity metrics
        fidelity_results = self._calculate_gate_fidelity(
            self.standard_df, interleaved_df
        )

        # Print detailed analysis
        self._print_analysis_results(fidelity_results, interleaved_df)

        # Create combined plot if requested
        if plot:
            self._create_combined_plot(
                self.standard_df, interleaved_df, save_image=save_image
            )

        # Save combined analysis data using standard experiment data saving
        if save_data:
            self._save_combined_results(fidelity_results, interleaved_df)

        return interleaved_df

    def _calculate_gate_fidelity(
        self, standard_df: pd.DataFrame, interleaved_df: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate gate-specific fidelity from standard vs interleaved RB."""
        try:
            from scipy.optimize import curve_fit

            def exponential_decay(m, A, r, B):
                return A * (r**m) + B

            # Fit standard RB
            std_lengths = standard_df["sequence_length"].values
            std_probs = standard_df["mean_survival_probability"].values
            popt_std, _ = curve_fit(
                exponential_decay, std_lengths, std_probs, bounds=([0, 0, 0], [1, 1, 1])
            )
            A_std, r_std, B_std = popt_std

            # Fit interleaved RB
            int_lengths = interleaved_df["sequence_length"].values
            int_probs = interleaved_df["mean_survival_probability"].values
            popt_int, _ = curve_fit(
                exponential_decay, int_lengths, int_probs, bounds=([0, 0, 0], [1, 1, 1])
            )
            A_int, r_int, B_int = popt_int

            # Calculate gate-specific metrics
            gate_error = (1 - r_int / r_std) / 2
            gate_fidelity = 1 - gate_error

            # Calculate individual errors per Clifford
            epsilon_clifford = (1 - r_std) / 2
            epsilon_interleaved = (1 - r_int) / 2

            return {
                "r_standard": r_std,
                "r_interleaved": r_int,
                "A_standard": A_std,
                "A_interleaved": A_int,
                "B_standard": B_std,
                "B_interleaved": B_int,
                "gate_error": gate_error,
                "gate_fidelity": gate_fidelity,
                "epsilon_clifford": epsilon_clifford,
                "epsilon_interleaved": epsilon_interleaved,
            }

        except Exception as e:
            print(f"Gate fidelity calculation failed: {e}")
            return {
                "r_standard": 1.0,
                "r_interleaved": 1.0,
                "gate_error": 0.0,
                "gate_fidelity": 1.0,
                "epsilon_clifford": 0.0,
                "epsilon_interleaved": 0.0,
            }

    def _print_analysis_results(
        self, fidelity_results: dict[str, float], interleaved_df: pd.DataFrame = None
    ):
        """Print detailed analysis results."""
        print("\n3. Gate Fidelity Analysis:")
        print("-" * 60)

        print(f"Standard RB decay rate (r_std): {fidelity_results['r_standard']:.6f}")
        print(
            f"Interleaved RB decay rate (r_int): {fidelity_results['r_interleaved']:.6f}"
        )

        print(
            f"\nError per Clifford gate (standard): {fidelity_results['epsilon_clifford']:.6f}"
        )
        print(
            f"Error per gate (interleaved): {fidelity_results['epsilon_interleaved']:.6f}"
        )

        print(f"\n{self.interleaved_gate.upper()} Gate Analysis:")
        print(
            f"Gate-specific error (ε_{self.interleaved_gate}): {fidelity_results['gate_error']:.6f}"
        )
        print(
            f"{self.interleaved_gate.upper()} Gate Fidelity: {fidelity_results['gate_fidelity']:.6f} ({fidelity_results['gate_fidelity'] * 100:.4f}%)"
        )

        # Compare final survival probabilities
        std_final = self.standard_df.iloc[-1]["mean_survival_probability"]
        int_final = interleaved_df.iloc[-1]["mean_survival_probability"]

        print(f"\nFinal Survival Probabilities ({self.max_sequence_length} gates):")
        print(f"Standard RB: {std_final:.4f}")
        print(f"Interleaved RB: {int_final:.4f}")
        print(
            f"Additional decay from {self.interleaved_gate.upper()} gate: {std_final - int_final:.4f}"
        )

        # Check convergence to 0.5
        print("\nConvergence Analysis:")
        print(f"Standard RB distance from 0.5: {abs(std_final - 0.5):.4f}")
        print(f"Interleaved RB distance from 0.5: {abs(int_final - 0.5):.4f}")

        if std_final < 0.7 and int_final < 0.7:
            print("✅ Both experiments show significant decay toward 0.5")

    def _create_combined_plot(
        self,
        standard_df: pd.DataFrame,
        interleaved_df: pd.DataFrame,
        save_image: bool = True,
    ):
        """Create combined plot showing both RB curves."""
        try:
            import numpy as np
            import plotly.graph_objects as go
            from scipy.optimize import curve_fit

            # Import standard visualization utilities
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

            def exponential_decay(m, A, r, B):
                return A * (r**m) + B

            # Get data
            std_lengths = standard_df["sequence_length"].values
            std_probs = standard_df["mean_survival_probability"].values
            std_errors = standard_df["std_survival_probability"].values

            int_lengths = interleaved_df["sequence_length"].values
            int_probs = interleaved_df["mean_survival_probability"].values
            int_errors = interleaved_df["std_survival_probability"].values

            # Fit both datasets
            popt_std, _ = curve_fit(
                exponential_decay, std_lengths, std_probs, bounds=([0, 0, 0], [1, 1, 1])
            )
            A_std, r_std, B_std = popt_std

            popt_int, _ = curve_fit(
                exponential_decay, int_lengths, int_probs, bounds=([0, 0, 0], [1, 1, 1])
            )
            A_int, r_int, B_int = popt_int

            # Create smooth curves for fitting
            x_smooth = np.linspace(1, max(max(std_lengths), max(int_lengths)), 100)
            y_std_fit = exponential_decay(x_smooth, A_std, r_std, B_std)
            y_int_fit = exponential_decay(x_smooth, A_int, r_int, B_int)

            # Create figure following project standards
            fig = go.Figure()

            # Add standard RB data (green for data points - colors[1])
            fig.add_trace(
                go.Scatter(
                    x=std_lengths,
                    y=std_probs,
                    error_y={"type": "data", "array": std_errors, "visible": True},
                    mode="markers",
                    name="Standard RB",
                    marker={"color": colors[1], "size": 8},
                    legendgroup="standard",
                )
            )

            # Add standard RB fit (blue for fit line - colors[0])
            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=y_std_fit,
                    mode="lines",
                    name=f"Standard Fit: r={r_std:.4f}",
                    line={"color": colors[0], "width": 2},
                    legendgroup="standard",
                )
            )

            # Add interleaved RB data (orange for data points - colors[2])
            fig.add_trace(
                go.Scatter(
                    x=int_lengths,
                    y=int_probs,
                    error_y={"type": "data", "array": int_errors, "visible": True},
                    mode="markers",
                    name=f"Interleaved RB ({self.interleaved_gate.upper()} gate)",
                    marker={"color": colors[2], "size": 8},
                    legendgroup="interleaved",
                )
            )

            # Add interleaved RB fit (red for fit line - colors[3])
            fig.add_trace(
                go.Scatter(
                    x=x_smooth,
                    y=y_int_fit,
                    mode="lines",
                    name=f"Interleaved Fit: r={r_int:.4f}",
                    line={"color": colors[3], "width": 2},
                    legendgroup="interleaved",
                )
            )

            # Calculate gate fidelity
            epsilon_gate = (1 - r_int / r_std) / 2
            gate_fidelity = 1 - epsilon_gate

            # Apply standard experiment layout
            apply_experiment_layout(
                fig,
                title=f"Interleaved Randomized Benchmarking: {self.interleaved_gate.upper()} Gate<br>Gate Fidelity: {gate_fidelity:.4f} ({gate_fidelity * 100:.2f}%)",
                xaxis_title="Sequence Length (# Clifford gates)",
                yaxis_title="Survival Probability",
                width=1000,
                height=500,
            )

            # Add horizontal line at 0.5 using project color scheme
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color=colors[6],  # Light gray
                annotation_text="Complete decoherence limit (0.5)",
            )

            # Add statistics box following project format
            stats_text = f"""{self.interleaved_gate.upper()} Gate Fidelity: {gate_fidelity:.4f} ({gate_fidelity * 100:.2f}%)
Standard RB: r={r_std:.4f}
Interleaved RB: r={r_int:.4f}
Gate Error: {epsilon_gate:.2e}"""

            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#CCCCCC",
                borderwidth=1,
                font={"size": 10},
                align="left",
                valign="top",
            )

            # Save using standard utility with proper session directory
            if save_image:
                # Use data manager session directory for proper organization
                images_dir = "./plots"  # Fallback
                if hasattr(self, "data_manager") and hasattr(
                    self.data_manager, "session_dir"
                ):
                    images_dir = f"{self.data_manager.session_dir}/plots"

                save_plotly_figure(
                    fig,
                    name=f"interleaved_rb_{self.interleaved_gate}_gate",
                    images_dir=images_dir,
                    width=1000,
                    height=500,
                )

            # Display using standard utility
            config = get_plotly_config(
                f"interleaved_rb_{self.interleaved_gate}", width=1000, height=500
            )
            show_plotly_figure(fig, config)

        except ImportError as e:
            print(f"Could not create combined plot: {e}")
            print("Install plotly and kaleido: pip install plotly kaleido")
        except Exception as e:
            print(f"Error creating combined plot: {e}")

    def _save_combined_results(
        self, fidelity_results: dict[str, float], interleaved_df: pd.DataFrame
    ):
        """Save combined analysis results using standard experiment data saving."""
        try:
            # Create combined analysis data in standard format
            combined_data = interleaved_df.to_dict(orient="records")

            # Use standard experiment data saving with extended metadata
            saved_path = self.save_experiment_data(
                combined_data,
                metadata={
                    "gate_analysis": {
                        "target_gate": self.interleaved_gate,
                        "gate_fidelity": fidelity_results["gate_fidelity"],
                        "gate_error": fidelity_results["gate_error"],
                        "standard_decay_rate": fidelity_results["r_standard"],
                        "interleaved_decay_rate": fidelity_results["r_interleaved"],
                        "clifford_error": fidelity_results["epsilon_clifford"],
                        "interleaved_error": fidelity_results["epsilon_interleaved"],
                    },
                    "experiment_type": "interleaved_randomized_benchmarking",
                    "physical_qubit": self.physical_qubit,
                },
                experiment_type="interleaved_randomized_benchmarking",
            )
            print(f"📊 Saved interleaved RB analysis: {saved_path}")

        except Exception as e:
            print(f"Failed to save combined results: {e}")

    # Remove the property to avoid conflicts with parent class
    # sequence_lengths is already inherited from RandomizedBenchmarking
