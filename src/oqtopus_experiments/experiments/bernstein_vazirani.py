#!/usr/bin/env python3
"""
Bernstein-Vazirani Algorithm Experiment Class
"""

from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from ..core.base_experiment import BaseExperiment
from ..models.bernstein_vazirani_models import (
    BernsteinVaziraniAnalysisResult,
    BernsteinVaziraniCircuitParams,
    BernsteinVaziraniParameters,
    BernsteinVaziraniResult,
)


class BernsteinVazirani(BaseExperiment):
    """Bernstein-Vazirani algorithm experiment

    Finds an n-bit secret string s with a single quantum query to an oracle.
    The oracle computes f_s(x) = s·x (dot product mod 2).

    Classical algorithms require n queries, while this quantum algorithm needs only 1.
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        secret_string: str | None = None,
        n_bits: int = 4,
    ):
        """Initialize Bernstein-Vazirani experiment

        Args:
            experiment_name: Optional experiment name
            secret_string: Binary string to find (e.g., "1011"). If None, random string is generated
            n_bits: Number of bits in the secret string (used if secret_string is None)
        """
        # Generate random secret if not provided
        if secret_string is None:
            secret_string = "".join(str(np.random.randint(0, 2)) for _ in range(n_bits))

        # Validate secret string
        if not all(bit in "01" for bit in secret_string):
            raise ValueError(
                f"Secret string must contain only 0s and 1s, got: {secret_string}"
            )

        self.params = BernsteinVaziraniParameters(
            experiment_name=experiment_name,
            secret_string=secret_string,
            n_bits=len(secret_string),
        )
        super().__init__(self.params.experiment_name or "bernstein_vazirani_experiment")

        self.secret_string = self.params.secret_string
        self.n_bits = self.params.n_bits
        # Convert string to list of integers with proper bit ordering
        # Human input is big-endian, but Qiskit uses little-endian indexing
        # So we reverse the string to match qubit indices
        self.secret_bits = [int(bit) for bit in self.secret_string[::-1]]

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """Analyze Bernstein-Vazirani results"""

        if not results:
            return pd.DataFrame()

        # Flatten all results into single list
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Process the results
        bv_result = self._process_results(all_results)
        if not bv_result:
            return pd.DataFrame()

        # Get device name from results
        device_name = "unknown"
        if all_results:
            device_name = all_results[0].get("backend", "unknown")

        # Create DataFrame
        df = self._create_dataframe(bv_result, device_name)

        # Create analysis result
        analysis_result = BernsteinVaziraniAnalysisResult(
            result=bv_result,
            dataframe=df,
            metadata={
                "experiment_type": "bernstein_vazirani",
                "secret_string": self.secret_string,
                "n_bits": self.n_bits,
            },
        )

        # Optional actions
        if plot:
            self._create_plot(analysis_result, save_image)
        if save_data:
            self._save_results(analysis_result)

        return df

    def circuits(self, **kwargs: Any) -> list["QuantumCircuit"]:
        """Generate Bernstein-Vazirani circuit

        The circuit structure:
        1. Prepare ancilla in |1⟩
        2. Apply Hadamard to all qubits
        3. Apply oracle (CNOTs controlled by bits where s_i = 1)
        4. Apply Hadamard to input qubits
        5. Measure input qubits
        """
        # Create single circuit for BV algorithm
        qc = self._create_bv_circuit()
        circuits = [qc]

        # Store parameters for analysis
        self.experiment_params = {
            "secret_string": self.secret_string,
            "n_bits": self.n_bits,
        }

        return circuits

    def _create_bv_circuit(self) -> QuantumCircuit:
        """Create Bernstein-Vazirani quantum circuit"""
        # n input qubits + 1 ancilla qubit
        n_qubits = self.n_bits + 1
        qc = QuantumCircuit(n_qubits, self.n_bits)

        # Step 1: Apply Hadamard gates to all qubits
        for i in range(n_qubits):
            qc.h(i)

        # Step 2: Apply Z to ancilla to create |-⟩ state
        # H|0⟩ = |+⟩, Z|+⟩ = |-⟩ (equivalent to H|1⟩)
        qc.z(self.n_bits)

        # Step 3: Apply oracle - CNOT for each bit s_i = 1
        # Note: secret_bits is already in little-endian order (reversed from human input)
        # so qubit index i correctly corresponds to bit position i in Qiskit's convention
        for i, bit in enumerate(self.secret_bits):
            if bit == 1:
                qc.cx(i, self.n_bits)  # Control on input qubit i, target on ancilla

        # Step 4: Apply Hadamard gates to input qubits (not ancilla)
        for i in range(self.n_bits):
            qc.h(i)

        # Step 5: Measure input qubits
        for i in range(self.n_bits):
            qc.measure(i, i)

        return qc

    def _process_results(
        self, all_results: list[dict[str, Any]]
    ) -> BernsteinVaziraniResult | None:
        """Process measurement results to extract the secret string"""
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
                    # Convert to string if needed
                    outcome_str = str(outcome)
                    total_counts[outcome_str] = total_counts.get(outcome_str, 0) + count

            if total_shots == 0:
                return None

            # Find the most frequent outcome
            measured_string = max(total_counts, key=total_counts.get)

            # Calculate success probability based on actual secret string measurements
            # Use secret string directly since measurement results match the input format
            secret_count = total_counts.get(self.secret_string, 0)
            success_probability = secret_count / total_shots

            # Consider correct if success rate > 50%
            # This is based on the actual measurement rate of the secret string
            is_correct = success_probability > 0.5

            # Calculate distribution metrics
            distribution = {k: v / total_shots for k, v in total_counts.items()}

            return BernsteinVaziraniResult(
                secret_string=self.secret_string,
                measured_string=measured_string,
                success_probability=success_probability,
                is_correct=is_correct,
                counts=total_counts,
                distribution=distribution,
                total_shots=total_shots,
            )

        except Exception as e:
            print(f"Error processing results: {e}")
            return None

    def _create_dataframe(
        self, result: BernsteinVaziraniResult, device_name: str = "unknown"
    ) -> pd.DataFrame:
        """Create DataFrame from results"""
        df_data = []

        # Add a row for each measurement outcome
        for outcome, probability in result.distribution.items():
            # Use outcome directly since oracle construction already handles bit ordering
            outcome_display = outcome
            df_data.append(
                {
                    "device": device_name,
                    "outcome": outcome_display,
                    "probability": probability,
                    "counts": result.counts.get(outcome, 0),
                    "is_secret": outcome_display == self.secret_string,
                    "secret_string": self.secret_string,
                    "measured_string": result.measured_string,
                    "success_probability": result.success_probability,
                    "is_correct": result.is_correct,
                }
            )

        # Sort by probability descending
        df = pd.DataFrame(df_data)
        if not df.empty:
            df = df.sort_values("probability", ascending=False)

        return df

    def _create_plot(
        self, analysis_result: BernsteinVaziraniAnalysisResult, save_image: bool = False
    ):
        """Create visualization for Bernstein-Vazirani results"""
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

            # Create histogram of measurement outcomes (counts)
            fig = go.Figure()

            # Sort dataframe by outcome as binary numbers for consistent display
            # Convert binary strings to integers for proper numerical sorting
            df_sorted = df.copy()
            df_sorted["outcome_int"] = df_sorted["outcome"].apply(lambda x: int(x, 2))
            df_sorted = (
                df_sorted.sort_values("outcome_int")
                .drop("outcome_int", axis=1)
                .reset_index(drop=True)
            )

            # Create separate traces for correct and incorrect outcomes for better visualization
            correct_mask = df_sorted["is_secret"]

            # Add trace for incorrect outcomes (noise)
            if not correct_mask.all():
                incorrect_data = df_sorted[~correct_mask]
                fig.add_trace(
                    go.Bar(
                        x=incorrect_data["outcome"],
                        y=incorrect_data["counts"],
                        name="Noise/Error",
                        marker={
                            "color": colors[2],  # Different color for noise
                            "line": {"width": 1, "color": "white"},
                        },
                        text=[f"{int(c)}" for c in incorrect_data["counts"]],
                        textposition="outside",
                    )
                )

            # Add trace for correct outcome (secret string)
            if correct_mask.any():
                correct_data = df_sorted[correct_mask]
                fig.add_trace(
                    go.Bar(
                        x=correct_data["outcome"],
                        y=correct_data["counts"],
                        name=f"Secret String ({self.secret_string})",
                        marker={
                            "color": colors[0],  # Highlight color for correct answer
                            "line": {
                                "width": 3,
                                "color": "darkgreen",
                            },  # Thick green border
                            "pattern": {"shape": ""},  # Solid fill
                        },
                        text=[f"{int(c)}" for c in correct_data["counts"]],
                        textposition="outside",
                    )
                )

            # If no correct measurements found, still show all data as noise
            if not correct_mask.any():
                fig.add_trace(
                    go.Bar(
                        x=df_sorted["outcome"],
                        y=df_sorted["counts"],
                        name="All measurements (no correct detection)",
                        marker={
                            "color": colors[2],
                            "line": {"width": 1, "color": "white"},
                        },
                        text=[f"{int(c)}" for c in df_sorted["counts"]],
                        textposition="outside",
                    )
                )

            # Apply layout
            apply_experiment_layout(
                fig,
                title=f"Bernstein-Vazirani Histogram: {result.total_shots} shots, secret='{result.secret_string}' ({device_name})",
                xaxis_title="Measurement Outcome (bit strings)",
                yaxis_title="Count (number of shots)",
                height=500,
                width=1000,
            )

            # Update layout for better display
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
            # Explicitly set the x-axis category order to maintain binary numerical sorting
            sorted_outcomes = df_sorted["outcome"].tolist()
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor="LightGray",
                tickangle=-45 if self.n_bits > 4 else 0,
                type="category",  # Force categorical x-axis to show bit strings properly
                categoryorder="array",
                categoryarray=sorted_outcomes,
            )
            fig.update_yaxes(
                range=[0, max(df_sorted["counts"]) * 1.1],
                showgrid=True,
                gridwidth=1,
                gridcolor="LightGray",
            )

            # Add annotation with results (emphasizing statistical nature)
            success_count = result.counts.get(
                result.secret_string[::-1], result.counts.get(result.secret_string, 0)
            )
            annotation_text = (
                f"<b>Secret String:</b> {result.secret_string}<br>"
                f"<b>Most Frequent:</b> {result.measured_string}<br>"
                f"<b>Success Count:</b> {success_count}/{result.total_shots}<br>"
                f"<b>Success Rate:</b> {result.success_probability:.1%}<br>"
                f"<b>Status:</b> {'✓ Correct' if result.is_correct else '✗ Incorrect'}<br>"
                f"<b>Backend:</b> {device_name}"
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

            # Add ideal marker line at total_shots for the secret string
            secret_mask = df_sorted["outcome"] == self.secret_string
            if secret_mask.any():
                secret_idx = secret_mask.idxmax()
                fig.add_shape(
                    type="line",
                    x0=secret_idx - 0.4,
                    x1=secret_idx + 0.4,
                    y0=result.total_shots,
                    y1=result.total_shots,
                    line={"color": "red", "width": 2, "dash": "dash"},
                )
                fig.add_annotation(
                    x=secret_idx,
                    y=result.total_shots * 1.05,
                    text="Ideal (all shots)",
                    showarrow=False,
                    font={"size": 10, "color": "red"},
                )

            # Save and show
            if save_image:
                images_dir = (
                    getattr(self.data_manager, "session_dir", "./images") + "/plots"
                )
                save_plotly_figure(
                    fig,
                    name=f"bernstein_vazirani_{self.n_bits}bits",
                    images_dir=images_dir,
                    width=1000,
                    height=500,
                )

            config = get_plotly_config(
                f"bernstein_vazirani_{self.n_bits}bits",
                width=1000,
                height=500,
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("plotly not available, skipping plot")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def _save_results(self, analysis_result: BernsteinVaziraniAnalysisResult):
        """Save analysis results"""
        try:
            result = analysis_result.result
            saved_path = self.save_experiment_data(
                analysis_result.dataframe.to_dict(orient="records"),
                metadata={
                    "result_summary": {
                        "secret_string": result.secret_string,
                        "measured_string": result.measured_string,
                        "success_probability": result.success_probability,
                        "is_correct": result.is_correct,
                        "total_shots": result.total_shots,
                    },
                    **analysis_result.metadata,
                },
                experiment_type="bernstein_vazirani",
            )
            print(f"Analysis data saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: Could not save analysis data: {e}")

    def _get_circuit_params(self) -> list[dict[str, Any]] | None:
        """Get circuit parameters for OQTOPUS"""
        if not hasattr(self, "experiment_params"):
            return None

        # BV has only one circuit
        param_model = BernsteinVaziraniCircuitParams(
            secret_string=self.experiment_params["secret_string"],
            n_bits=self.experiment_params["n_bits"],
        )

        return [param_model.model_dump()]
