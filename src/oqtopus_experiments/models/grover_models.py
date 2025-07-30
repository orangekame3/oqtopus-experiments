#!/usr/bin/env python3
"""
Pydantic models for Grover's quantum search algorithm experiment
Provides structured data validation and serialization
"""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .experiment_result import ExperimentResult


class GroverAnalysisResult(BaseModel):
    """Analysis results for Grover's algorithm performance"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success_probability: float = Field(description="Measured success probability")
    theoretical_success_probability: float = Field(
        description="Theoretical success probability"
    )
    marked_states: list[int] = Field(description="List of marked state indices")
    unmarked_states: list[int] = Field(description="List of unmarked state indices")
    measurement_counts: dict[str, int] = Field(description="Raw measurement counts")
    total_shots: int = Field(description="Total number of measurements")
    success_rate_error: float = Field(
        description="Difference between theoretical and measured success rates"
    )
    optimal_iterations: int = Field(description="Theoretical optimal iterations")
    actual_iterations: int = Field(description="Actual iterations used")


class GroverParameters(BaseModel):
    """Grover's algorithm experiment parameters"""

    experiment_name: str | None = Field(default=None, description="Experiment name")
    n_qubits: int = Field(default=2, description="Number of qubits in search space")
    marked_states: list[int] | str = Field(
        default="random", description="Marked state indices or 'random'"
    )
    num_iterations: int | None = Field(
        default=None, description="Number of Grover iterations (None for optimal)"
    )


class GroverData(BaseModel):
    """Data structure for Grover's algorithm measurements"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_qubits: int = Field(description="Number of qubits")
    marked_states: list[int] = Field(description="Indices of marked states")
    num_iterations: int = Field(description="Number of Grover iterations performed")
    measurement_counts: dict[str, int] = Field(description="Measurement results")
    total_shots: int = Field(description="Total number of shots")
    search_space_size: int = Field(description="Total number of states (2^n)")
    analysis_result: GroverAnalysisResult | None = Field(
        default=None, description="Analysis results"
    )


class GroverResult(ExperimentResult):
    """Complete Grover's algorithm experiment result"""

    def __init__(self, data: GroverData, **kwargs):
        """Initialize with Grover data"""
        super().__init__(**kwargs)
        self.data = data

    def analyze(
        self,
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze Grover's algorithm results

        Args:
            plot: Whether to generate plots using plot_settings.md guidelines
            save_data: Whether to save analysis results
            save_image: Whether to save generated plots

        Returns:
            DataFrame with analysis results including success probabilities
        """
        try:
            import math

            # Extract measurement data
            data = self.data
            counts = data.measurement_counts
            total_shots = data.total_shots
            marked_states = data.marked_states
            n_qubits = data.n_qubits
            search_space_size = data.search_space_size
            num_iterations = data.num_iterations

            # Calculate success probability (measured)
            marked_counts = sum(
                counts.get(format(state, f"0{n_qubits}b"), 0) for state in marked_states
            )
            success_probability = marked_counts / total_shots if total_shots > 0 else 0

            # Calculate theoretical success probability
            num_marked = len(marked_states)
            if num_marked == 0:
                theoretical_success_probability = 0.0
                optimal_iterations = 0
            else:
                # Theoretical optimal number of iterations
                optimal_iterations = int(
                    (math.pi / 4) * math.sqrt(search_space_size / num_marked)
                )

                # Theoretical success probability after num_iterations
                theta = math.asin(math.sqrt(num_marked / search_space_size))
                theoretical_success_probability = (
                    math.sin((2 * num_iterations + 1) * theta) ** 2
                )

            # Identify unmarked states
            all_states = list(range(search_space_size))
            unmarked_states = [s for s in all_states if s not in marked_states]

            # Calculate analysis results
            analysis_result = GroverAnalysisResult(
                success_probability=success_probability,
                theoretical_success_probability=theoretical_success_probability,
                marked_states=marked_states,
                unmarked_states=unmarked_states,
                measurement_counts=counts,
                total_shots=total_shots,
                success_rate_error=abs(
                    theoretical_success_probability - success_probability
                ),
                optimal_iterations=optimal_iterations,
                actual_iterations=num_iterations,
            )

            # Store analysis result
            self.data.analysis_result = analysis_result

            # Create DataFrame with state-by-state results
            df_data = []
            for state in range(search_space_size):
                state_str = format(state, f"0{n_qubits}b")
                count = counts.get(state_str, 0)
                probability = count / total_shots if total_shots > 0 else 0
                is_marked = state in marked_states

                df_data.append(
                    {
                        "state": state,
                        "state_binary": state_str,
                        "count": count,
                        "probability": probability,
                        "is_marked": is_marked,
                        "theoretical_probability": (
                            theoretical_success_probability / num_marked
                            if is_marked and num_marked > 0
                            else (
                                (1 - theoretical_success_probability)
                                / (search_space_size - num_marked)
                                if not is_marked and num_marked < search_space_size
                                else 0
                            )
                        ),
                    }
                )

            df = pd.DataFrame(df_data)

            # Generate plots following plot_settings.md
            if plot:
                self._plot_grover_results(df, analysis_result, save_image)

            # Save data - handled by experiment classes per architecture
            if save_data:
                # Data saving handled by experiment classes
                pass

            return df

        except Exception as e:
            print(f"Analysis failed: {e}")
            # Return DataFrame with error info
            return pd.DataFrame(
                {
                    "state": list(range(data.search_space_size)),
                    "error": [str(e)] * data.search_space_size,
                }
            )

    def _plot_grover_results(
        self,
        df: pd.DataFrame,
        analysis_result: GroverAnalysisResult,
        save_image: bool = False,
    ):
        """Plot Grover's algorithm results following plot_settings.md guidelines"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            from ..utils.visualization import (
                get_experiment_colors,
                get_plotly_config,
                setup_plotly_environment,
                show_plotly_figure,
            )

            setup_plotly_environment()
            colors = get_experiment_colors()

            # Create subplots: measurement histogram and success probability comparison
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=[
                    "Measurement Distribution",
                    "Success Probability Comparison",
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}]],
            )

            # Plot 1: Measurement histogram with marked vs unmarked highlighting
            marked_mask = df["is_marked"]
            unmarked_mask = ~marked_mask

            # Marked states (green - colors[1])
            if marked_mask.any():
                fig.add_trace(
                    go.Bar(
                        x=df[marked_mask]["state_binary"],
                        y=df[marked_mask]["probability"],
                        name="Marked States",
                        marker_color=colors[1],
                        opacity=0.8,
                    ),
                    row=1,
                    col=1,
                )

            # Unmarked states (blue - colors[0])
            if unmarked_mask.any():
                fig.add_trace(
                    go.Bar(
                        x=df[unmarked_mask]["state_binary"],
                        y=df[unmarked_mask]["probability"],
                        name="Unmarked States",
                        marker_color=colors[0],
                        opacity=0.6,
                    ),
                    row=1,
                    col=1,
                )

            # Plot 2: Success probability comparison
            categories = ["Measured", "Theoretical"]
            values = [
                analysis_result.success_probability,
                analysis_result.theoretical_success_probability,
            ]

            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    name="Success Probability",
                    marker_color=[colors[1], colors[0]],
                    text=[f"{v:.3f}" for v in values],
                    textposition="auto",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # Update subplot layouts
            fig.update_xaxes(title_text="Quantum States", row=1, col=1)
            fig.update_yaxes(title_text="Probability", row=1, col=1)
            fig.update_xaxes(title_text="Measurement Type", row=1, col=2)
            fig.update_yaxes(title_text="Success Probability", row=1, col=2)

            # Update overall layout following plot_settings.md
            fig.update_layout(
                title_text="Grover's Algorithm Results",
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=True,
                width=1000,
                height=500,
                font={"size": 12},
            )

            # Add grid lines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

            # Add statistics annotation
            stats_text = f"""Iterations: {analysis_result.actual_iterations} (optimal: {analysis_result.optimal_iterations})
Success Rate: {analysis_result.success_probability:.3f}
Theoretical: {analysis_result.theoretical_success_probability:.3f}
Error: {analysis_result.success_rate_error:.3f}
Marked States: {len(analysis_result.marked_states)}/{2**self.data.n_qubits}"""

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

            # Show plot
            config = get_plotly_config("grover", width=1000, height=500)
            show_plotly_figure(fig, config)

        except ImportError:
            print("Plotly not available. Skipping plot.")
        except Exception as e:
            print(f"Plotting failed: {e}")
