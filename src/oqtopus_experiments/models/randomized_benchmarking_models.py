#!/usr/bin/env python3
"""
Pydantic models for Randomized Benchmarking experiment
Provides structured data validation and serialization
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .experiment_result import ExperimentResult


class RandomizedBenchmarkingFittingResult(BaseModel):
    """Fitting results for Randomized Benchmarking decay"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    error_per_clifford: float = Field(description="Average error per Clifford gate")
    decay_rate: float = Field(description="Exponential decay rate (r)")
    initial_fidelity: float = Field(description="Initial fidelity (A)")
    offset: float = Field(description="Fitting offset (B)")
    r_squared: float = Field(description="R-squared goodness of fit")
    sequence_lengths: list[int] = Field(description="Clifford sequence lengths")
    survival_probabilities: list[float] = Field(
        description="Measured survival probabilities"
    )
    error_info: str | None = Field(
        default=None, description="Error information if fitting failed"
    )


class RandomizedBenchmarkingParameters(BaseModel):
    """Randomized Benchmarking experiment parameters"""

    experiment_name: str | None = Field(default=None, description="Experiment name")
    physical_qubit: int = Field(default=0, description="Target physical qubit")
    max_sequence_length: int = Field(
        default=100, description="Maximum Clifford sequence length"
    )
    num_lengths: int = Field(
        default=10, description="Number of different sequence lengths"
    )
    num_samples: int = Field(
        default=50, description="Number of random sequences per length"
    )
    rb_type: str = Field(
        default="standard", description="Type of RB: 'standard' or 'interleaved'"
    )
    interleaved_gate: str | None = Field(
        default=None, description="Gate to interleave (for interleaved RB)"
    )


class RandomizedBenchmarkingData(BaseModel):
    """Data structure for Randomized Benchmarking measurements"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequence_lengths: list[int] = Field(description="Clifford sequence lengths")
    survival_probabilities: list[list[float]] = Field(
        description="Survival probabilities for each length and sample"
    )
    mean_survival_probabilities: list[float] = Field(
        description="Mean survival probability for each length"
    )
    std_survival_probabilities: list[float] = Field(
        description="Standard deviation for each length"
    )
    num_samples: int = Field(description="Number of samples per length")
    fitting_result: RandomizedBenchmarkingFittingResult | None = Field(
        default=None, description="Exponential decay fitting results"
    )


class RandomizedBenchmarkingResult(ExperimentResult):
    """Complete Randomized Benchmarking experiment result"""

    def __init__(self, data: RandomizedBenchmarkingData, **kwargs):
        """Initialize with RB data"""
        super().__init__(**kwargs)
        self.data = data

    def analyze(
        self,
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze Randomized Benchmarking results with exponential decay fitting

        Args:
            plot: Whether to generate plots using plot_settings.md guidelines
            save_data: Whether to save analysis results
            save_image: Whether to save generated plots

        Returns:
            DataFrame with analysis results including fitted parameters
        """
        try:
            # Extract measurement data
            data = self.data
            lengths = data.sequence_lengths
            mean_probs = data.mean_survival_probabilities
            std_probs = data.std_survival_probabilities

            # Check if we have enough data for fitting
            if len(lengths) < 2:
                # Not enough data points for fitting
                fitting_result = RandomizedBenchmarkingFittingResult(
                    error_per_clifford=0.0,
                    decay_rate=1.0,
                    initial_fidelity=1.0,
                    offset=0.0,
                    r_squared=0.0,
                    sequence_lengths=lengths,
                    survival_probabilities=mean_probs,
                    error_info="Insufficient data points for fitting (need at least 2)",
                )
            else:
                # Fit decay curve: P(m) = A * r^m + B
                fitting_result = self._fit_exponential_decay(
                    lengths, mean_probs, std_probs
                )

            # Create DataFrame
            df_data = []
            for i, length in enumerate(lengths):
                df_data.append(
                    {
                        "sequence_length": length,
                        "mean_survival_probability": mean_probs[i],
                        "std_survival_probability": std_probs[i],
                        "num_samples": data.num_samples,
                    }
                )

            df = pd.DataFrame(df_data)

            # Add fitted curve if successful
            if not fitting_result.error_info:
                df["fitted_survival_probability"] = self._rb_decay_function(
                    df["sequence_length"].values,
                    fitting_result.initial_fidelity,
                    fitting_result.decay_rate,
                    fitting_result.offset,
                )

            # Store fitting result
            self.data.fitting_result = fitting_result

            # Generate plots following plot_settings.md
            if plot:
                self._plot_rb_results(df, fitting_result, save_image, plot)

            # Save data if requested
            if save_data:
                self._save_results(df, fitting_result)

            return df

        except Exception as e:
            print(f"Analysis failed: {e}")
            # Return DataFrame with error info
            return pd.DataFrame(
                {
                    "sequence_length": data.sequence_lengths,
                    "error": [str(e)] * len(data.sequence_lengths),
                }
            )

    def _rb_decay_function(
        self, sequence_length: list[int], A: float, r: float, B: float
    ) -> list[float]:
        """RB decay function: P(m) = A * r^m + B"""
        import numpy as np

        result = A * (r ** np.array(sequence_length)) + B
        return result.tolist()  # type: ignore

    def _fit_exponential_decay(
        self, lengths: list[int], mean_probs: list[float], std_probs: list[float]
    ) -> RandomizedBenchmarkingFittingResult:
        """Fit exponential decay to RB data"""
        try:
            import numpy as np
            from scipy.optimize import curve_fit

            # Initial guess: A=0.5, r=0.99, B=0.5
            initial_guess = [0.5, 0.99, 0.5]
            bounds = ([0, 0, 0], [1, 1, 1])  # Physical bounds

            def decay_func(m, A, r, B):
                return A * (r**m) + B

            popt, pcov = curve_fit(
                decay_func,
                np.array(lengths),
                np.array(mean_probs),
                p0=initial_guess,
                bounds=bounds,
                sigma=std_probs,
                absolute_sigma=True,
            )

            A, r, B = popt

            # Calculate error per Clifford (for single qubit)
            error_per_clifford = (1 - r) / 2

            # Calculate R-squared
            y_pred = decay_func(np.array(lengths), A, r, B)
            ss_res = np.sum((np.array(mean_probs) - y_pred) ** 2)
            ss_tot = np.sum((np.array(mean_probs) - np.mean(mean_probs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            return RandomizedBenchmarkingFittingResult(
                error_per_clifford=error_per_clifford,
                decay_rate=r,
                initial_fidelity=A,
                offset=B,
                r_squared=r_squared,
                sequence_lengths=lengths,
                survival_probabilities=mean_probs,
            )

        except Exception as e:
            return RandomizedBenchmarkingFittingResult(
                error_per_clifford=0.0,
                decay_rate=1.0,
                initial_fidelity=1.0,
                offset=0.0,
                r_squared=0.0,
                sequence_lengths=lengths,
                survival_probabilities=mean_probs,
                error_info=f"Fitting failed: {str(e)}",
            )

    def _plot_rb_results(
        self,
        df: pd.DataFrame,
        fitting_result: RandomizedBenchmarkingFittingResult,
        save_image: bool = False,
        plot: bool = True,
    ):
        """Plot RB decay curve following plot_settings.md guidelines"""
        try:
            import plotly.graph_objects as go

            from ..utils.visualization import (
                apply_experiment_layout,
                get_experiment_colors,
                get_plotly_config,
                setup_plotly_environment,
                show_plotly_figure,
            )

            setup_plotly_environment()
            colors = get_experiment_colors()

            # Create figure with plot_settings.md specifications
            fig = go.Figure()

            # Plot data points with error bars (primary data in green - colors[1])
            fig.add_trace(
                go.Scatter(
                    x=df["sequence_length"],
                    y=df["mean_survival_probability"],
                    error_y={"array": df["std_survival_probability"]},
                    mode="markers",
                    name="Data",
                    marker={"color": colors[1], "size": 8},
                    line={"color": colors[1]},
                )
            )

            # Plot fitted curve if available (fit in blue - colors[0])
            if "fitted_survival_probability" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["sequence_length"],
                        y=df["fitted_survival_probability"],
                        mode="lines",
                        name=f"Fit: r={fitting_result.decay_rate:.4f}",
                        line={"color": colors[0], "width": 2},
                    )
                )

            # Apply layout using standard utility
            apply_experiment_layout(
                fig,
                title="Randomized Benchmarking Decay",
                xaxis_title="Sequence Length (# Clifford gates)",
                yaxis_title="Survival Probability",
                width=1000,
                height=500,
            )

            # Add statistics box if fitting succeeded
            if not fitting_result.error_info:
                stats_text = f"""Error per Clifford: {fitting_result.error_per_clifford:.2e}
Decay rate (r): {fitting_result.decay_rate:.6f}
R²: {fitting_result.r_squared:.4f}"""

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

            # Show plot - saving handled by experiment classes
            if save_image:
                # Plot saving handled by experiment classes via data manager
                pass

            config = get_plotly_config(
                "randomized_benchmarking", width=1000, height=500
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("Plotly not available. Skipping plot.")
        except Exception as e:
            print(f"Plotting failed: {e}")

    def _save_results(
        self, df: pd.DataFrame, fitting_result: RandomizedBenchmarkingFittingResult
    ):
        """Deprecated method - data saving handled by experiment classes"""
        pass
