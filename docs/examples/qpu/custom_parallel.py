#!/usr/bin/env python3
"""
Simple parallel execution example using OQTOPUS backend.
Demonstrates how to define custom circuits and execute them in parallel.
"""

import pandas as pd
import plotly.graph_objects as go
from qiskit import QuantumCircuit

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.core.base_experiment import BaseExperiment
from oqtopus_experiments.utils.visualization import (
    apply_experiment_layout,
    get_experiment_colors,
    get_plotly_config,
    save_plotly_figure,
    setup_plotly_environment,
    show_plotly_figure,
)


class CustomExperiment(BaseExperiment):
    """Custom experiment class for Bell state measurements."""

    def __init__(self, experiment_name: str | None = None, num_circuits: int = 10):
        self.num_circuits = num_circuits
        super().__init__(experiment_name or "bell_state_parallel")

    def circuits(self, **kwargs) -> list[QuantumCircuit]:
        """Generate Bell state circuits for parallel execution."""
        circuits = []
        for _ in range(self.num_circuits):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            circuits.append(qc)
        return circuits

    def analyze(
        self,
        results: dict[str, list[dict]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """Analyze Bell state measurement results."""
        # Combine all counts from parallel executions
        combined_counts = {"00": 0, "01": 0, "10": 0, "11": 0}

        # Flatten results from all circuits
        for circuit_results in results.values():
            for result in circuit_results:
                if "counts" in result:
                    for bitstring, count in result["counts"].items():
                        if bitstring in combined_counts:
                            combined_counts[bitstring] += count

        total_counts = sum(combined_counts.values())

        # Create analysis DataFrame with proper state ordering
        state_order = ["00", "01", "10", "11"]
        df = pd.DataFrame(
            [
                {
                    "state": state,
                    "count": combined_counts[state],
                    "probability": (
                        combined_counts[state] / total_counts
                        if total_counts > 0
                        else 0.0
                    ),
                }
                for state in state_order
            ]
        )

        if plot:
            self._create_histogram(df, save_image)

        return df

    def _create_histogram(self, df: pd.DataFrame, save_image: bool = True) -> None:
        """Create histogram visualization of Bell state measurements."""
        setup_plotly_environment()
        colors = get_experiment_colors()

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df["state"],
                y=df["count"],
                text=[f"{prob:.3f}" for prob in df["probability"]],
                textposition="outside",
                marker_color=colors[1],
                name="Counts",
            )
        )

        apply_experiment_layout(
            fig,
            title="Bell State Measurement Results",
            xaxis_title="Measurement Outcome",
            yaxis_title="Counts",
            width=700,
            height=500,
        )

        # Ensure x-axis shows state labels as strings, not numbers
        fig.update_xaxes(type="category")

        fig.add_annotation(
            text="Ideal Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2<br>Expected: 50% |00⟩, 50% |11⟩",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.95,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#CCCCCC",
            borderwidth=1,
            font={"size": 12},
        )

        if save_image:
            save_plotly_figure(fig, name="bell_state_histogram", images_dir="./plots")

        config = get_plotly_config("bell_state_histogram", 700, 500)
        show_plotly_figure(fig, config)


def main():
    """Execute Bell state experiment following standard pattern."""
    # Setup backend
    backend = OqtopusBackend(device="urchin")

    # Create experiment instance
    exp = CustomExperiment(num_circuits=10)

    # Run experiment (returns ExperimentResult)
    result = exp.run_parallel(backend, shots=1000, workers=10, mitigation_info={})

    # Analyze results (returns DataFrame)
    df = result.analyze(plot=True, save_data=True, save_image=True)

    # Display summary
    print("\nBell State Measurement Summary:")
    print(df)
    print(f"\nTotal measurements: {df['count'].sum()}")

    # Calculate fidelity with ideal Bell state
    ideal_fidelity = (
        df.loc[df["state"] == "00", "probability"].iloc[0]
        + df.loc[df["state"] == "11", "probability"].iloc[0]
    )
    print(f"Bell state fidelity: {ideal_fidelity:.3f}")


if __name__ == "__main__":
    main()
