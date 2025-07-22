#!/usr/bin/env python3
"""
CHSH (Bell Inequality) Experiment Class
"""

import math
from typing import Any

import pandas as pd
from qiskit import QuantumCircuit

from ..core.base_experiment import BaseExperiment
from ..models.chsh_models import (
    CHSHAnalysisResult,
    CHSHCircuitParams,
    CHSHExperimentResult,
    CHSHParameters,
)


class CHSH(BaseExperiment):
    """CHSH Bell inequality experiment for testing quantum nonlocality"""

    def __init__(
        self,
        experiment_name: str | None = None,
        physical_qubit_0: int | None = None,
        physical_qubit_1: int | None = None,
        shots_per_circuit: int = 1000,
        measurement_angles: dict[str, float] | None = None,
        theta: float = 0.0,
    ):
        """Initialize CHSH experiment with explicit parameters"""
        # Track if physical qubits were explicitly specified
        self._physical_qubits_specified = (
            physical_qubit_0 is not None and physical_qubit_1 is not None
        )
        actual_physical_qubit_0 = (
            physical_qubit_0 if physical_qubit_0 is not None else 0
        )
        actual_physical_qubit_1 = (
            physical_qubit_1 if physical_qubit_1 is not None else 1
        )

        # Default measurement angles for optimal CHSH violation
        default_angles = {
            "alice_0": 0.0,  # θ_A0 = 0°
            "alice_1": 45.0,  # θ_A1 = 45°
            "bob_0": 22.5,  # θ_B0 = 22.5°
            "bob_1": 67.5,  # θ_B1 = 67.5°
        }

        self.params = CHSHParameters(
            experiment_name=experiment_name,
            physical_qubit_0=actual_physical_qubit_0,
            physical_qubit_1=actual_physical_qubit_1,
            shots_per_circuit=shots_per_circuit,
            theta=theta,
            measurement_angles=measurement_angles or default_angles,
        )
        super().__init__(self.params.experiment_name or "chsh_experiment")

        self.physical_qubit_0 = self.params.physical_qubit_0
        self.physical_qubit_1 = self.params.physical_qubit_1
        self.shots_per_circuit = self.params.shots_per_circuit
        self.measurement_angles = self.params.measurement_angles
        self.theta = theta

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """Analyze CHSH results and calculate Bell inequality violation"""

        if not results:
            return pd.DataFrame()

        # Flatten all results into single list (no device separation)
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Analyze CHSH data
        analysis_result = self._analyze_chsh_data(all_results)
        if not analysis_result:
            return pd.DataFrame()

        # Create DataFrame
        df = self._create_dataframe(analysis_result)

        # Create experiment result
        experiment_result = CHSHExperimentResult(
            analysis_result=analysis_result,
            dataframe=df,
            metadata={
                "experiment_type": "chsh",
                "physical_qubit_0": self.physical_qubit_0,
                "physical_qubit_1": self.physical_qubit_1,
            },
        )

        # Optional actions
        if plot:
            self._create_plot(experiment_result, save_image)
        if save_data:
            self._save_results(experiment_result)

        return df

    def circuits(self, **kwargs: Any) -> list[Any]:
        """Generate CHSH circuits for sampling-based measurement"""
        circuits = []

        # Use theta from constructor (can be overridden by kwargs for backward compatibility)
        theta = kwargs.get("theta", self.theta)

        # The four measurement bases for CHSH: ZZ, ZX, XZ, XX
        measurement_bases = [
            ("ZZ", False, False),  # No additional rotations
            ("ZX", False, True),  # H gate on qubit 1 (Bob)
            ("XZ", True, False),  # H gate on qubit 0 (Alice)
            ("XX", True, True),  # H gates on both qubits
        ]

        for basis_name, alice_x, bob_x in measurement_bases:
            qc = QuantumCircuit(2, 2)

            # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
            qc.h(0)  # Put first qubit in superposition
            qc.cx(0, 1)  # Entangle with second qubit

            # Apply parameterized rotation to Alice (qubit 0)
            qc.ry(theta, 0)

            # Apply measurement basis rotations
            if alice_x:  # Measure Alice in X basis
                qc.h(0)
            if bob_x:  # Measure Bob in X basis
                qc.h(1)

            # Measurements
            qc.measure(0, 0)  # Alice's measurement
            qc.measure(1, 1)  # Bob's measurement

            circuits.append(qc)

        # Store parameters for analysis and OQTOPUS
        self.experiment_params = {
            "measurement_bases": measurement_bases,
            "theta": theta,
            "logical_qubit_0": 0,
            "logical_qubit_1": 1,
            "physical_qubit_0": self.physical_qubit_0,
            "physical_qubit_1": self.physical_qubit_1,
        }

        # Auto-transpile if physical qubits explicitly specified
        if self._physical_qubits_specified:
            # For 2-qubit experiments, we need a different transpilation approach
            # This would require extending the base class method or using a different approach
            # For now, keep circuits as-is for local simulation
            pass

        return circuits  # type: ignore

    def run(self, backend, shots: int = 1024, theta: float = math.pi / 4, **kwargs):
        """
        Run CHSH experiment with specified theta angle

        Args:
            backend: Backend instance
            shots: Number of shots per circuit
            theta: Measurement angle for Alice (radians), default π/4 for optimal violation
            **kwargs: Additional arguments
        """
        # Override circuits to use the specified theta
        original_circuits = self.circuits
        self.circuits = lambda **kw: original_circuits(theta=theta, **kw)

        try:
            # Use BaseExperiment's run method
            result = super().run(backend=backend, shots=shots, **kwargs)
            return result
        finally:
            # Restore original circuits method
            self.circuits = original_circuits

    def _analyze_chsh_data(
        self, all_results: list[dict[str, Any]]
    ) -> CHSHAnalysisResult | None:
        """Analyze CHSH experimental data using ZZ, ZX, XZ, XX correlations"""
        try:
            # Group results by measurement basis
            measurement_counts = self._extract_measurement_counts(all_results)

            if len(measurement_counts) != 4:
                return None

            # Calculate correlations for each measurement basis
            correlations = {}
            correlation_errors = {}
            total_shots = 0

            for basis, counts in measurement_counts.items():
                correlation, error, shots = self._calculate_correlation(counts)
                correlations[basis] = correlation
                correlation_errors[basis] = error
                total_shots += shots

            # Calculate CHSH quantities:
            # CHSH1 = <ZZ> - <ZX> + <XZ> + <XX>
            # CHSH2 = <ZZ> + <ZX> - <XZ> + <XX>
            chsh1 = (
                correlations["ZZ"]
                - correlations["ZX"]
                + correlations["XZ"]
                + correlations["XX"]
            )
            chsh2 = (
                correlations["ZZ"]
                + correlations["ZX"]
                - correlations["XZ"]
                + correlations["XX"]
            )

            # Take the maximum violation
            chsh_value = max(abs(chsh1), abs(chsh2))

            # Calculate uncertainty in CHSH value
            chsh_std_error = math.sqrt(
                correlation_errors["ZZ"] ** 2
                + correlation_errors["ZX"] ** 2
                + correlation_errors["XZ"] ** 2
                + correlation_errors["XX"] ** 2
            )

            # Check for Bell inequality violation (S > 2)
            bell_violation = chsh_value > 2.0

            # Calculate statistical significance
            significance = (
                (chsh_value - 2.0) / chsh_std_error if chsh_std_error > 0 else 0
            )

            return CHSHAnalysisResult(
                chsh_value=chsh_value,
                chsh_std_error=chsh_std_error,
                bell_violation=bell_violation,
                significance=significance,
                correlations=correlations,
                correlation_errors=correlation_errors,
                measurement_counts=measurement_counts,
                total_shots=total_shots,
            )

        except Exception as e:
            print(f"CHSH analysis failed: {e}")
            return None

    def _extract_measurement_counts(
        self, all_results: list[dict[str, Any]]
    ) -> dict[str, dict[str, int]]:
        """Extract measurement counts grouped by measurement basis"""
        measurement_counts = {}

        for i, result in enumerate(all_results):
            # Determine measurement basis from circuit index
            if i < len(self.experiment_params["measurement_bases"]):
                basis_name, _, _ = self.experiment_params["measurement_bases"][i]

                counts = result.get("counts", {})
                measurement_counts[basis_name] = counts

        return measurement_counts

    def _calculate_correlation(
        self, counts: dict[str, int]
    ) -> tuple[float, float, int]:
        """Calculate correlation E(A,B) = P(A=B) - P(A≠B)"""
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0, 0.0, 0

        # Count same outcomes (00, 11) and different outcomes (01, 10)
        # Backend should have normalized integer keys to string keys
        same_outcomes = counts.get("00", 0) + counts.get("11", 0)
        different_outcomes = counts.get("01", 0) + counts.get("10", 0)

        # Calculate correlation
        correlation = (same_outcomes - different_outcomes) / total_shots

        # Calculate standard error (assuming binomial statistics)
        p_same = same_outcomes / total_shots
        variance = p_same * (1 - p_same) / total_shots
        std_error = 2 * math.sqrt(variance)  # Factor of 2 from correlation formula

        return correlation, std_error, total_shots

    def _create_dataframe(self, analysis_result: CHSHAnalysisResult) -> pd.DataFrame:
        """Create DataFrame from CHSH analysis results"""
        # Create one row per measurement basis
        df_data = []

        for basis, correlation in analysis_result.correlations.items():
            counts = analysis_result.measurement_counts[basis]
            total = sum(counts.values())

            df_data.append(
                {
                    "measurement_basis": basis,
                    "correlation": correlation,
                    "correlation_error": analysis_result.correlation_errors[basis],
                    "counts_00": counts.get("00", 0),
                    "counts_01": counts.get("01", 0),
                    "counts_10": counts.get("10", 0),
                    "counts_11": counts.get("11", 0),
                    "total_shots": total,
                    "chsh_value": analysis_result.chsh_value,
                    "bell_violation": analysis_result.bell_violation,
                    "significance": analysis_result.significance,
                }
            )

        return pd.DataFrame(df_data)

    def _create_plot(
        self, experiment_result: CHSHExperimentResult, save_image: bool = False
    ):
        """Create CHSH visualization"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

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

            # Create subplots: correlations and CHSH value
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Correlation Values", "CHSH Test"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]],
            )

            analysis = experiment_result.analysis_result

            # Plot 1: Correlation values
            settings = list(analysis.correlations.keys())
            correlations = list(analysis.correlations.values())
            errors = [analysis.correlation_errors[s] for s in settings]

            fig.add_trace(
                go.Bar(
                    x=settings,
                    y=correlations,
                    error_y=dict(type="data", array=errors),
                    name="Correlations",
                    marker_color=colors[1],
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Plot 2: CHSH value comparison
            chsh_categories = ["Classical Limit", "CHSH Value", "Quantum Limit"]
            chsh_values = [2.0, analysis.chsh_value, 2.828]
            chsh_colors = [
                colors[2],
                colors[0] if analysis.bell_violation else colors[3],
                colors[2],
            ]

            fig.add_trace(
                go.Bar(
                    x=chsh_categories,
                    y=chsh_values,
                    error_y=dict(type="data", array=[0, analysis.chsh_std_error, 0]),
                    name="CHSH Comparison",
                    marker_color=chsh_colors,
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # Update layout
            fig.update_layout(
                title=f"CHSH Bell Test : Q{self.physical_qubit_0}-Q{self.physical_qubit_1}",
                height=400,
                showlegend=False,
            )

            # Update axes
            fig.update_xaxes(title_text="Measurement Settings", row=1, col=1)
            fig.update_yaxes(
                title_text="Correlation E(A,B)", row=1, col=1, range=[-1.1, 1.1]
            )

            fig.update_xaxes(title_text="CHSH Bounds", row=1, col=2)
            fig.update_yaxes(title_text="CHSH Value S", row=1, col=2, range=[0, 3])

            # Add violation annotation
            violation_text = (
                "Bell Violation!" if analysis.bell_violation else "No Violation"
            )
            violation_color = "green" if analysis.bell_violation else "red"

            fig.add_annotation(
                x=0.98,
                y=0.98,
                text=f"{violation_text}<br>S = {analysis.chsh_value:.3f} ± {analysis.chsh_std_error:.3f}<br>σ = {analysis.significance:.1f}",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color=violation_color),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=violation_color,
                borderwidth=2,
                align="right",
            )

            # Save and show
            if save_image:
                images_dir = (
                    getattr(self.data_manager, "session_dir", "./images") + "/plots"
                )
                save_plotly_figure(
                    fig,
                    name=f"chsh_Q{self.physical_qubit_0}_Q{self.physical_qubit_1}",
                    images_dir=images_dir,
                    width=800,
                    height=400,
                )

            config = get_plotly_config(
                f"chsh_Q{self.physical_qubit_0}_Q{self.physical_qubit_1}",
                width=800,
                height=400,
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("plotly not available, skipping plot")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def _save_results(self, experiment_result: CHSHExperimentResult):
        """Save CHSH analysis results"""
        try:
            analysis = experiment_result.analysis_result
            saved_path = self.save_experiment_data(
                experiment_result.dataframe.to_dict(orient="records"),
                metadata={
                    "chsh_summary": {
                        "chsh_value": analysis.chsh_value,
                        "bell_violation": analysis.bell_violation,
                        "significance": analysis.significance,
                        "correlations": analysis.correlations,
                    },
                    **experiment_result.metadata,
                },
                experiment_type="chsh",
            )
            print(f"CHSH analysis data saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: Could not save CHSH analysis data: {e}")

    def _get_circuit_params(self) -> list[dict[str, Any]] | None:
        """Get circuit parameters for OQTOPUS"""
        if not hasattr(self, "experiment_params"):
            return None

        measurement_bases = self.experiment_params["measurement_bases"]
        theta = self.experiment_params.get("theta", 0.0)
        logical_qubit_0 = self.experiment_params.get("logical_qubit_0", 0)
        logical_qubit_1 = self.experiment_params.get("logical_qubit_1", 1)
        physical_qubit_0 = self.experiment_params.get("physical_qubit_0", 0)
        physical_qubit_1 = self.experiment_params.get("physical_qubit_1", 1)

        circuit_params = []
        for i, (basis_name, alice_x, bob_x) in enumerate(measurement_bases):
            param_model = CHSHCircuitParams(
                measurement_setting=basis_name,
                alice_angle=theta,  # Single angle parameter
                bob_angle=0.0,  # Bob doesn't rotate
                logical_qubit_0=logical_qubit_0,
                logical_qubit_1=logical_qubit_1,
                physical_qubit_0=physical_qubit_0,
                physical_qubit_1=physical_qubit_1,
            )
            # Add circuit index for parameter tracking
            params_dict = param_model.model_dump()
            params_dict["circuit_index"] = i
            circuit_params.append(params_dict)

        return circuit_params
