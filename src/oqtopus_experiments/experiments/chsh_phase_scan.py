#!/usr/bin/env python3
"""
CHSH Phase Scan Experiment Class
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from ..core.base_experiment import BaseExperiment
from ..models.chsh_phase_scan_models import (
    CHSHPhaseScanAnalysisResult,
    CHSHPhaseScanCircuitParams,
    CHSHPhaseScanExperimentResult,
    CHSHPhaseScanParameters,
    CHSHPhaseScanPoint,
)


class CHSHPhaseScan(BaseExperiment):
    """CHSH phase scan experiment for studying Bell violation vs measurement phase"""

    def __init__(
        self,
        experiment_name: str | None = None,
        physical_qubit_0: int | None = None,
        physical_qubit_1: int | None = None,
        shots_per_circuit: int = 1000,
        phase_points: int = 21,
        phase_start: float = 0.0,
        phase_end: float = 2*math.pi,
    ):
        """Initialize CHSH phase scan experiment with explicit parameters"""
        # Track if physical qubits were explicitly specified
        self._physical_qubits_specified = (
            physical_qubit_0 is not None and physical_qubit_1 is not None
        )
        actual_physical_qubit_0 = physical_qubit_0 if physical_qubit_0 is not None else 0
        actual_physical_qubit_1 = physical_qubit_1 if physical_qubit_1 is not None else 1

        self.params = CHSHPhaseScanParameters(
            experiment_name=experiment_name,
            physical_qubit_0=actual_physical_qubit_0,
            physical_qubit_1=actual_physical_qubit_1,
            shots_per_circuit=shots_per_circuit,
            phase_points=phase_points,
            phase_start=phase_start,
            phase_end=phase_end,
        )
        super().__init__(self.params.experiment_name or "chsh_phase_experiment")

        self.physical_qubit_0 = self.params.physical_qubit_0
        self.physical_qubit_1 = self.params.physical_qubit_1
        self.shots_per_circuit = self.params.shots_per_circuit
        self.phase_points = self.params.phase_points
        self.phase_start = self.params.phase_start
        self.phase_end = self.params.phase_end

        # Generate phase array
        self.phases = np.linspace(self.phase_start, self.phase_end, self.phase_points)

    def analyze(
        self, results: dict[str, list[dict[str, Any]]], **kwargs: Any
    ) -> pd.DataFrame:
        """Analyze CHSH phase scan results"""
        plot = kwargs.get("plot", False)
        save_data = kwargs.get("save_data", False)
        save_image = kwargs.get("save_image", False)

        if not results:
            return pd.DataFrame()

        # Flatten all results into single list (no device separation)
        all_results = []
        for device_data in results.values():
            all_results.extend(device_data)

        if not all_results:
            return pd.DataFrame()

        # Analyze CHSH phase data
        analysis_result = self._analyze_chsh_phase_data(all_results)
        if not analysis_result:
            return pd.DataFrame()

        # Create DataFrame
        df = self._create_dataframe(analysis_result)

        # Create experiment result
        experiment_result = CHSHPhaseScanExperimentResult(
            analysis_result=analysis_result,
            dataframe=df,
            metadata={
                "experiment_type": "chsh_phase_scan",
                "physical_qubit_0": self.physical_qubit_0,
                "physical_qubit_1": self.physical_qubit_1,
                "phase_points": self.phase_points,
                "phase_range": f"{self.phase_start:.3f} to {self.phase_end:.3f} rad",
            },
        )

        # Optional actions
        if plot:
            self._create_plot(experiment_result, save_image)
        if save_data:
            self._save_results(experiment_result)

        return df

    def circuits(self, **kwargs: Any) -> list[Any]:
        """Generate CHSH circuits for all phases and measurement bases"""
        circuits = []

        # The four measurement bases for CHSH: ZZ, ZX, XZ, XX
        measurement_bases = [
            ("ZZ", False, False),  # No additional rotations
            ("ZX", False, True),   # H gate on qubit 1 (Bob)
            ("XZ", True, False),   # H gate on qubit 0 (Alice)
            ("XX", True, True),    # H gates on both qubits
        ]

        # Generate circuits for each phase and each measurement basis
        for phase in self.phases:
            for basis_name, alice_x, bob_x in measurement_bases:
                qc = QuantumCircuit(2, 2)

                # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
                qc.h(0)  # Put first qubit in superposition
                qc.cx(0, 1)  # Entangle with second qubit

                # Apply parameterized rotation to Alice (qubit 0)
                qc.ry(phase, 0)

                # Apply measurement basis rotations
                if alice_x:  # Measure Alice in X basis
                    qc.h(0)
                if bob_x:    # Measure Bob in X basis
                    qc.h(1)

                # Measurements
                qc.measure(0, 0)  # Alice's measurement
                qc.measure(1, 1)  # Bob's measurement

                circuits.append(qc)

        # Store parameters for analysis and OQTOPUS
        self.experiment_params = {
            "phases": self.phases,
            "measurement_bases": measurement_bases,
            "logical_qubit_0": 0,
            "logical_qubit_1": 1,
            "physical_qubit_0": self.physical_qubit_0,
            "physical_qubit_1": self.physical_qubit_1,
        }

        return circuits  # type: ignore

    def _analyze_chsh_phase_data(
        self, all_results: list[dict[str, Any]]
    ) -> CHSHPhaseScanAnalysisResult | None:
        """Analyze CHSH phase scan experimental data"""
        try:
            phase_points = []
            num_bases = 4  # ZZ, ZX, XZ, XX

            # Process results for each phase
            for i, phase in enumerate(self.phases):
                # Extract results for this phase (4 circuits per phase)
                phase_results = all_results[i*num_bases:(i+1)*num_bases]

                if len(phase_results) != num_bases:
                    continue

                # Calculate correlations for each measurement basis
                correlations = {}
                correlation_errors = {}
                total_shots = 0

                measurement_bases = self.experiment_params["measurement_bases"]
                for j, (basis_name, _, _) in enumerate(measurement_bases):
                    counts = phase_results[j].get("counts", {})
                    correlation, error, shots = self._calculate_correlation(counts)
                    correlations[basis_name] = correlation
                    correlation_errors[basis_name] = error
                    total_shots += shots

                # Calculate CHSH quantities
                chsh1 = correlations["ZZ"] - correlations["ZX"] + correlations["XZ"] + correlations["XX"]
                chsh2 = correlations["ZZ"] + correlations["ZX"] - correlations["XZ"] + correlations["XX"]
                chsh_max = max(abs(chsh1), abs(chsh2))

                # Create phase point
                phase_point = CHSHPhaseScanPoint(
                    phase_radians=phase,
                    phase_degrees=math.degrees(phase),
                    chsh1_value=chsh1,
                    chsh2_value=chsh2,
                    chsh_max=chsh_max,
                    bell_violation=chsh_max > 2.0,
                    correlations=correlations,
                    correlation_errors=correlation_errors,
                    total_shots=total_shots,
                )
                phase_points.append(phase_point)

            if not phase_points:
                return None

            # Find maximum CHSH value and corresponding phase
            chsh_max_values = [point.chsh_max for point in phase_points]
            max_idx = np.argmax(chsh_max_values)
            max_chsh_value = chsh_max_values[max_idx]
            max_chsh_phase = phase_points[max_idx].phase_radians

            # Count violations
            violation_count = sum(1 for point in phase_points if point.bell_violation)

            return CHSHPhaseScanAnalysisResult(
                phase_points=phase_points,
                max_chsh_value=max_chsh_value,
                max_chsh_phase=max_chsh_phase,
                violation_count=violation_count,
            )

        except Exception as e:
            print(f"CHSH phase analysis failed: {e}")
            return None

    def _calculate_correlation(
        self, counts: dict[str, int]
    ) -> tuple[float, float, int]:
        """Calculate correlation E(A,B) = P(A=B) - P(A≠B)"""
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0, 0.0, 0

        # Count same outcomes (00, 11) and different outcomes (01, 10)
        same_outcomes = counts.get("00", 0) + counts.get("11", 0)
        different_outcomes = counts.get("01", 0) + counts.get("10", 0)

        # Calculate correlation
        correlation = (same_outcomes - different_outcomes) / total_shots

        # Calculate standard error (assuming binomial statistics)
        p_same = same_outcomes / total_shots
        variance = p_same * (1 - p_same) / total_shots
        std_error = 2 * math.sqrt(variance)  # Factor of 2 from correlation formula

        return correlation, std_error, total_shots

    def _create_dataframe(self, analysis_result: CHSHPhaseScanAnalysisResult) -> pd.DataFrame:
        """Create DataFrame from CHSH phase analysis results"""
        df_data = []

        for point in analysis_result.phase_points:
            df_data.append({
                "phase_radians": point.phase_radians,
                "phase_degrees": point.phase_degrees,
                "chsh1_value": point.chsh1_value,
                "chsh2_value": point.chsh2_value,
                "chsh_max": point.chsh_max,
                "bell_violation": point.bell_violation,
                "correlation_ZZ": point.correlations["ZZ"],
                "correlation_ZX": point.correlations["ZX"],
                "correlation_XZ": point.correlations["XZ"],
                "correlation_XX": point.correlations["XX"],
                "total_shots": point.total_shots,
            })

        return pd.DataFrame(df_data)

    def _create_plot(
        self, experiment_result: CHSHPhaseScanExperimentResult, save_image: bool = False
    ):
        """Create CHSH phase scan visualization with filled limit regions"""
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

            analysis = experiment_result.analysis_result

            # Get data arrays
            phases_deg = analysis.get_phases_degrees()
            chsh1_values = analysis.get_chsh1_values()
            chsh2_values = analysis.get_chsh2_values()

            # Create single plot figure
            fig = go.Figure()

            # Add filled regions only for quantum violation areas
            x_full = list(phases_deg) + list(phases_deg[::-1])
            quantum_max = 2*np.sqrt(2)

            # Upper quantum violation region: 2 < CHSH ≤ 2√2
            fig.add_trace(
                go.Scatter(
                    x=x_full,
                    y=[quantum_max]*len(phases_deg) + [2]*len(phases_deg),
                    fill='toself',
                    fillcolor='rgba(100, 150, 200, 0.2)',  # Light blue for quantum violation
                    line=dict(color='rgba(100, 150, 200, 0)'),
                    name='Bell Violation Region',
                    hoverinfo='skip',
                    showlegend=True
                )
            )

            # Lower quantum violation region: -2√2 ≤ CHSH < -2
            fig.add_trace(
                go.Scatter(
                    x=x_full,
                    y=[-2]*len(phases_deg) + [-quantum_max]*len(phases_deg),
                    fill='toself',
                    fillcolor='rgba(100, 150, 200, 0.2)',  # Same light blue
                    line=dict(color='rgba(100, 150, 200, 0)'),
                    name='Bell Violation Region',
                    hoverinfo='skip',
                    showlegend=False  # Don't duplicate legend
                )
            )

            # Add CHSH1 and CHSH2 data
            fig.add_trace(
                go.Scatter(
                    x=phases_deg,
                    y=chsh1_values,
                    mode='lines+markers',
                    name="CHSH1",
                    line=dict(color=colors[0], width=3),
                    marker=dict(size=6)
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=phases_deg,
                    y=chsh2_values,
                    mode='lines+markers',
                    name="CHSH2",
                    line=dict(color=colors[1], width=3),
                    marker=dict(size=6)
                )
            )

            # Add reference lines
            # Classical limits
            fig.add_hline(y=2.0, line_dash="dash", line_color="gray", line_width=2,
                         annotation_text="Classical Limit (+2)",
                         annotation_position="top right")
            fig.add_hline(y=-2.0, line_dash="dash", line_color="gray", line_width=2,
                         annotation_text="Classical Limit (-2)",
                         annotation_position="bottom right")

            # Quantum limits
            fig.add_hline(y=quantum_max, line_dash="dot", line_color="red", line_width=2,
                         annotation_text=f"Quantum Maximum (+{quantum_max:.3f})",
                         annotation_position="top right")
            fig.add_hline(y=-quantum_max, line_dash="dot", line_color="red", line_width=2,
                         annotation_text=f"Quantum Maximum (-{quantum_max:.3f})",
                         annotation_position="bottom right")

            # Update layout with white background to match other experiments
            fig.update_layout(
                title=f"CHSH Phase Scan: Bell Inequality vs Measurement Phase<br>Q{self.physical_qubit_0}-Q{self.physical_qubit_1}",
                xaxis_title="Phase θ (degrees)",
                yaxis_title="CHSH Value",
                height=600,
                width=900,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                plot_bgcolor='white',  # White background
                paper_bgcolor='white'  # White paper background
            )

            # Update axes with grid for consistency with other experiments
            fig.update_xaxes(
                range=[phases_deg.min()-5, phases_deg.max()+5],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
            fig.update_yaxes(
                range=[-3.2, 3.2],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )

            # Add summary annotation
            max_violation = analysis.max_chsh_value
            max_phase_deg = math.degrees(analysis.max_chsh_phase)
            violation_percent = 100 * analysis.violation_count / len(analysis.phase_points)

            fig.add_annotation(
                x=0.98, y=0.02,
                text="<b>Summary</b><br>" +
                     f"Max |CHSH|: {max_violation:.3f}<br>" +
                     f"Optimal θ: {max_phase_deg:.1f}°<br>" +
                     f"Bell Violations: {violation_percent:.1f}%<br>" +
                     f"Theoretical Max: {quantum_max:.3f}",
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="black",
                borderwidth=1,
                align="left",
            )

            # Save and show
            if save_image:
                images_dir = (
                    getattr(self.data_manager, "session_dir", "./images") + "/plots"
                )
                save_plotly_figure(
                    fig,
                    name=f"chsh_phase_Q{self.physical_qubit_0}_Q{self.physical_qubit_1}",
                    images_dir=images_dir,
                    width=900,
                    height=600,
                )

            config = get_plotly_config(
                f"chsh_phase_Q{self.physical_qubit_0}_Q{self.physical_qubit_1}",
                width=900, height=600
            )
            show_plotly_figure(fig, config)

        except ImportError:
            print("plotly not available, skipping plot")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def _save_results(self, experiment_result: CHSHPhaseScanExperimentResult):
        """Save CHSH phase analysis results"""
        try:
            analysis = experiment_result.analysis_result
            saved_path = self.save_experiment_data(
                experiment_result.dataframe.to_dict(orient="records"),
                metadata={
                    "chsh_phase_summary": {
                        "max_chsh_value": analysis.max_chsh_value,
                        "max_chsh_phase_rad": analysis.max_chsh_phase,
                        "max_chsh_phase_deg": math.degrees(analysis.max_chsh_phase),
                        "violation_count": analysis.violation_count,
                        "total_points": len(analysis.phase_points),
                    },
                    **experiment_result.metadata,
                },
                experiment_type="chsh_phase",
            )
            print(f"CHSH phase analysis data saved to: {saved_path}")
        except Exception as e:
            print(f"Warning: Could not save CHSH phase analysis data: {e}")

    def _get_circuit_params(self) -> list[dict[str, Any]] | None:
        """Get circuit parameters for OQTOPUS"""
        if not hasattr(self, "experiment_params"):
            return None

        phases = self.experiment_params["phases"]
        measurement_bases = self.experiment_params["measurement_bases"]
        logical_qubit_0 = self.experiment_params.get("logical_qubit_0", 0)
        logical_qubit_1 = self.experiment_params.get("logical_qubit_1", 1)
        physical_qubit_0 = self.experiment_params.get("physical_qubit_0", 0)
        physical_qubit_1 = self.experiment_params.get("physical_qubit_1", 1)

        circuit_params = []
        for phase in phases:
            for basis_name, _, _ in measurement_bases:
                param_model = CHSHPhaseScanCircuitParams(
                    phase_radians=phase,
                    measurement_basis=basis_name,
                    logical_qubit_0=logical_qubit_0,
                    logical_qubit_1=logical_qubit_1,
                    physical_qubit_0=physical_qubit_0,
                    physical_qubit_1=physical_qubit_1,
                )
                circuit_params.append(param_model.model_dump())

        return circuit_params
