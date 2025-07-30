#!/usr/bin/env python3
"""
Deutsch-Jozsa Algorithm Experiment Class
"""

from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from ..core.base_experiment import BaseExperiment
from ..models.deutsch_jozsa_models import (
    DeutschJozsaAnalysisResult,
    DeutschJozsaCircuitParams,
    DeutschJozsaParameters,
    DeutschJozsaResult,
)


class DeutschJozsa(BaseExperiment):
    """Deutsch-Jozsa algorithm experiment

    Determines whether a black-box Boolean function is constant (returns the same
    value for all inputs) or balanced (returns 0 for half the inputs and 1 for
    the other half) with a single oracle query.

    Classical algorithms require up to 2^(n-1) + 1 queries, while this quantum
    algorithm needs only 1.
    """

    def __init__(
        self,
        n_qubits: int = 3,
        oracle_type: str = "balanced_random",
        experiment_name: str | None = None,
    ):
        """Initialize Deutsch-Jozsa experiment

        Args:
            n_qubits: Number of input qubits (typically 2-6)
            oracle_type: Type of oracle function to use
                Options: "constant_0", "constant_1", "balanced_random", "balanced_alternating"
            experiment_name: Optional experiment name
        """
        self.params = DeutschJozsaParameters(
            experiment_name=experiment_name,
            n_qubits=n_qubits,
            oracle_type=oracle_type,
        )
        super().__init__(self.params.experiment_name or "deutsch_jozsa_experiment")

        self.n_qubits = self.params.n_qubits
        self.oracle_type = self.params.oracle_type

        # Generate oracle function based on type
        self._generate_oracle()

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """Analyze Deutsch-Jozsa results"""

        if not results:
            return pd.DataFrame()

        # Flatten all results into single list
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Process the results
        dj_result = self._process_results(all_results)
        if not dj_result:
            return pd.DataFrame()

        # Get device name from results
        device_name = "unknown"
        if all_results:
            device_name = all_results[0].get("backend", "unknown")

        # Create DataFrame
        df = self._create_dataframe(dj_result, device_name)

        # Create analysis result
        analysis_result = DeutschJozsaAnalysisResult(
            result=dj_result,
            dataframe=df,
            metadata={
                "experiment_type": "deutsch_jozsa",
                "n_qubits": self.n_qubits,
                "oracle_type": self.oracle_type,
            },
        )

        # Optional actions
        if plot:
            self._create_plot(analysis_result, save_image)
        if save_data:
            self._save_results(analysis_result)

        return df

    def circuits(self, **kwargs: Any) -> list["QuantumCircuit"]:
        """Generate Deutsch-Jozsa circuit

        The circuit structure:
        1. Initialize ancilla in |1⟩
        2. Apply Hadamard to all qubits
        3. Apply oracle function
        4. Apply Hadamard to input qubits
        5. Measure input qubits
        """
        # Create single circuit for DJ algorithm
        qc = self._create_dj_circuit()
        circuits = [qc]

        # Store parameters for analysis
        self.experiment_params = {
            "n_qubits": self.n_qubits,
            "oracle_type": self.oracle_type,
        }

        return circuits

    def _generate_oracle(self):
        """Generate oracle function based on type"""
        if self.oracle_type == "constant_0":
            # Function always returns 0
            self.oracle_function = lambda x: 0
            self.is_constant = True
        elif self.oracle_type == "constant_1":
            # Function always returns 1
            self.oracle_function = lambda x: 1
            self.is_constant = True
        elif self.oracle_type == "balanced_random":
            # Random balanced function
            n = 2**self.n_qubits
            values = [0] * (n // 2) + [1] * (n // 2)
            np.random.shuffle(values)
            self.oracle_map = {i: values[i] for i in range(n)}
            self.oracle_function = lambda x: self.oracle_map[x]
            self.is_constant = False
        elif self.oracle_type == "balanced_alternating":
            # Alternating balanced function (e.g., XOR of all bits)
            self.oracle_function = lambda x: bin(x).count("1") % 2
            self.is_constant = False
        else:
            raise ValueError(f"Unknown oracle type: {self.oracle_type}")

    def _create_dj_circuit(self) -> QuantumCircuit:
        """Create Deutsch-Jozsa quantum circuit"""
        # n input qubits + 1 ancilla qubit
        total_qubits = self.n_qubits + 1
        qc = QuantumCircuit(total_qubits, self.n_qubits)

        # Step 1: Initialize ancilla to |1⟩
        qc.x(self.n_qubits)  # Ancilla is the last qubit

        # Step 2: Apply Hadamard gates to all qubits
        for i in range(total_qubits):
            qc.h(i)

        # Step 3: Apply oracle
        self._apply_oracle(qc)

        # Step 4: Apply Hadamard gates to input qubits (not ancilla)
        for i in range(self.n_qubits):
            qc.h(i)

        # Step 5: Measure input qubits
        for i in range(self.n_qubits):
            qc.measure(i, i)

        return qc

    def _apply_oracle(self, qc: QuantumCircuit):
        """Apply oracle to quantum circuit"""
        ancilla_idx = self.n_qubits

        if self.oracle_type == "constant_0":
            # Identity operation (do nothing)
            pass
        elif self.oracle_type == "constant_1":
            # Apply X to ancilla
            qc.x(ancilla_idx)
        elif self.oracle_type == "balanced_random":
            # Apply controlled operations based on oracle map
            for x, f_x in self.oracle_map.items():
                if f_x == 1:
                    # Create multi-controlled X gate
                    control_bits = []
                    for i in range(self.n_qubits):
                        if (x >> i) & 1:
                            control_bits.append(i)
                        else:
                            qc.x(i)  # Flip bit to match pattern

                    # Apply multi-controlled X
                    if len(control_bits) == 0:
                        qc.x(ancilla_idx)
                    else:
                        qc.mcx(control_bits, ancilla_idx)

                    # Unflip bits
                    for i in range(self.n_qubits):
                        if not ((x >> i) & 1):
                            qc.x(i)
        elif self.oracle_type == "balanced_alternating":
            # XOR of all input bits - apply cascading CNOTs
            for i in range(self.n_qubits):
                qc.cx(i, ancilla_idx)

    def _process_results(
        self, all_results: list[dict[str, Any]]
    ) -> DeutschJozsaResult | None:
        """Process measurement results to determine if function is constant or balanced"""
        try:
            # Extract counts from results
            total_counts: dict[str, int] = {}
            total_shots = 0

            for result in all_results:
                counts = result.get("counts", {})
                shots = sum(counts.values())
                total_shots += shots

                # Aggregate counts
                for outcome, count in counts.items():
                    outcome_str = str(outcome)
                    total_counts[outcome_str] = total_counts.get(outcome_str, 0) + count

            if total_shots == 0:
                return None

            # Count measurements of all zeros
            all_zeros = "0" * self.n_qubits
            all_zeros_count = total_counts.get(all_zeros, 0)
            all_zeros_probability = all_zeros_count / total_shots

            # Determine if function is constant based on measurements
            # In ideal case: constant -> measure all zeros, balanced -> never measure all zeros
            # With noise, use threshold
            measured_constant = all_zeros_probability > 0.5
            is_correct = measured_constant == self.is_constant

            # Calculate distribution
            distribution = {k: v / total_shots for k, v in total_counts.items()}

            return DeutschJozsaResult(
                oracle_type=self.oracle_type,
                is_constant_actual=self.is_constant,
                is_constant_measured=measured_constant,
                all_zeros_probability=all_zeros_probability,
                is_correct=is_correct,
                counts=total_counts,
                distribution=distribution,
                total_shots=total_shots,
            )

        except Exception as e:
            print(f"Error processing results: {e}")
            return None

    def _create_dataframe(
        self, result: DeutschJozsaResult, device_name: str = "unknown"
    ) -> pd.DataFrame:
        """Create DataFrame from results"""
        df_data = []

        # Add a row for each measurement outcome
        for outcome, probability in result.distribution.items():
            df_data.append(
                {
                    "device": device_name,
                    "outcome": outcome,
                    "probability": probability,
                    "counts": result.counts.get(outcome, 0),
                    "is_all_zeros": outcome == "0" * self.n_qubits,
                    "oracle_type": self.oracle_type,
                    "is_constant_actual": result.is_constant_actual,
                    "is_constant_measured": result.is_constant_measured,
                    "all_zeros_probability": result.all_zeros_probability,
                    "is_correct": result.is_correct,
                }
            )

        # Sort by probability descending
        df = pd.DataFrame(df_data)
        if not df.empty:
            df = df.sort_values("probability", ascending=False)

        return df

    def _create_plot(
        self, analysis_result: DeutschJozsaAnalysisResult, save_image: bool = False
    ):
        """Create visualization for Deutsch-Jozsa results"""
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

            result = analysis_result.result
            df = analysis_result.dataframe

            # Get device name
            device_name = "unknown"
            if not df.empty and "device" in df.columns:
                device_name = df["device"].iloc[0]

            # Create histogram of measurement outcomes
            fig = go.Figure()

            # Sort dataframe by outcome as binary numbers
            df_sorted = df.copy()
            df_sorted["outcome_int"] = df_sorted["outcome"].apply(lambda x: int(x, 2))
            df_sorted = (
                df_sorted.sort_values("outcome_int")
                .drop("outcome_int", axis=1)
                .reset_index(drop=True)
            )

            # Create separate traces for all-zeros and other outcomes
            all_zeros_mask = df_sorted["is_all_zeros"]

            # Add trace for non-zero outcomes
            if not all_zeros_mask.all():
                other_data = df_sorted[~all_zeros_mask]
                fig.add_trace(
                    go.Bar(
                        x=other_data["outcome"],
                        y=other_data["counts"],
                        name="Non-zero outcomes",
                        marker={
                            "color": (
                                colors[2] if result.is_constant_actual else colors[1]
                            ),
                            "line": {"width": 1, "color": "white"},
                        },
                        text=[f"{int(c)}" for c in other_data["counts"]],
                        textposition="outside",
                    )
                )

            # Add trace for all-zeros outcome
            if all_zeros_mask.any():
                zeros_data = df_sorted[all_zeros_mask]
                fig.add_trace(
                    go.Bar(
                        x=zeros_data["outcome"],
                        y=zeros_data["counts"],
                        name="All zeros (|0...0⟩)",
                        marker={
                            "color": (
                                colors[0] if result.is_constant_actual else colors[3]
                            ),
                            "line": {
                                "width": 3,
                                "color": (
                                    "darkblue"
                                    if result.is_constant_actual
                                    else "darkred"
                                ),
                            },
                        },
                        text=[f"{int(c)}" for c in zeros_data["counts"]],
                        textposition="outside",
                    )
                )

            # Apply layout
            oracle_desc = self.oracle_type.replace("_", " ").title()
            apply_experiment_layout(
                fig,
                title=f"Deutsch-Jozsa Results: {oracle_desc} Oracle, {self.n_qubits} qubits ({device_name})",
                xaxis_title="Measurement Outcome",
                yaxis_title="Count",
                height=500,
                width=1000,
            )

            # Update layout
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=True,
                legend={
                    "x": 0.7,
                    "y": 0.95,
                    "bgcolor": "rgba(255,255,255,0.8)",
                    "bordercolor": "black",
                    "borderwidth": 1,
                },
            )

            # Update axes
            sorted_outcomes = df_sorted["outcome"].tolist()
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="LightGray",
                tickangle=-45 if self.n_qubits > 3 else 0,
                type="category",
                categoryorder="array",
                categoryarray=sorted_outcomes,
            )
            fig.update_yaxes(
                range=[0, max(df_sorted["counts"]) * 1.1],
                showgrid=True,
                gridwidth=1,
                gridcolor="LightGray",
            )

            # Add annotation with results
            function_type = "Constant" if result.is_constant_actual else "Balanced"
            measured_type = "Constant" if result.is_constant_measured else "Balanced"
            annotation_text = (
                f"<b>Oracle Type:</b> {oracle_desc}<br>"
                f"<b>Actual Function:</b> {function_type}<br>"
                f"<b>Measured as:</b> {measured_type}<br>"
                f"<b>P(|0...0⟩):</b> {result.all_zeros_probability:.1%}<br>"
                f"<b>Total Shots:</b> {result.total_shots}<br>"
                f"<b>Result:</b> {'✓ Correct' if result.is_correct else '✗ Incorrect'}"
            )

            fig.add_annotation(
                x=0.02,
                y=0.98,
                text=annotation_text,
                xref="paper",
                yref="paper",
                showarrow=False,
                font={"size": 12, "color": "#333333"},
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#CCCCCC",
                borderwidth=1,
                align="left",
            )

            # Add threshold line at 50% for constant detection
            fig.add_hline(
                y=result.total_shots * 0.5,
                line_dash="dash",
                line_color="gray",
                annotation_text="50% threshold",
                annotation_position="right",
            )

            # Save and show
            if save_image:
                images_dir = (
                    getattr(self.data_manager, "session_dir", "./images") + "/plots"
                )
                save_plotly_figure(
                    fig,
                    name=f"deutsch_jozsa_{self.n_qubits}qubits_{self.oracle_type}",
                    images_dir=images_dir,
                    width=1000,
                    height=500,
                )

            config = get_plotly_config(
                f"deutsch_jozsa_{self.n_qubits}qubits", width=1000, height=500
            )
            show_plotly_figure(fig, config)

        except Exception as e:
            print(f"Failed to create plot: {e}")

    def _save_results(self, analysis_result: DeutschJozsaAnalysisResult):
        """Save analysis results to CSV file"""
        try:
            # Save to CSV using the DataFrame
            filename = (
                f"deutsch_jozsa_{self.n_qubits}qubits_{self.oracle_type}_results.csv"
            )
            analysis_result.dataframe.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Failed to save results: {e}")

    def _get_circuit_params(self) -> list[DeutschJozsaCircuitParams] | None:
        """Get circuit parameters for backend"""
        return [
            DeutschJozsaCircuitParams(
                n_qubits=self.n_qubits,
                oracle_type=self.oracle_type,
            )
        ]
