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
    CHSHCircuitParams,
    CHSHData,
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

        # Extract measurement counts for CHSH analysis
        measurement_counts = self._extract_measurement_counts(all_results)
        if len(measurement_counts) != 4:
            return pd.DataFrame()

        # Calculate total shots
        total_shots = sum(
            sum(counts.values()) for counts in measurement_counts.values()
        )

        # Create CHSH data structure
        chsh_data = CHSHData(
            measurement_counts=measurement_counts,
            correlations={},  # Will be filled by analysis
            correlation_errors={},  # Will be filled by analysis
            total_shots=total_shots,
        )

        # Create experiment result with new error handling pattern
        experiment_result = CHSHExperimentResult(
            data=chsh_data,
            raw_results=results,
            experiment_instance=self,
        )

        # Analysis handled by CHSHExperimentResult class
        df = experiment_result.analyze(
            plot=plot, save_data=save_data, save_image=save_image
        )

        return df

    def circuits(self, **kwargs: Any) -> list["QuantumCircuit"]:
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

        for _basis_name, alice_x, bob_x in measurement_bases:
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

        return circuits

    def run(
        self, backend: Any, shots: int = 1024, theta: float = math.pi / 4, **kwargs: Any
    ) -> Any:
        """
        Run CHSH experiment with specified theta angle

        Args:
            backend: Backend instance
            shots: Number of shots per circuit
            theta: Measurement angle for Alice (radians), default π/4 for optimal violation
            **kwargs: Additional arguments
        """
        # Store theta temporarily for use in circuits method
        original_theta = self.theta
        self.theta = theta

        try:
            # Use BaseExperiment's run method
            result = super().run(backend=backend, shots=shots, **kwargs)
            return result
        finally:
            # Restore original theta
            self.theta = original_theta

    def _extract_measurement_counts(
        self, all_results: list[dict[str, Any]]
    ) -> dict[str, dict[str, int]]:
        """Extract measurement counts grouped by measurement basis"""
        measurement_counts = {}

        # Default measurement bases if experiment_params not set (for testing)
        default_bases = [
            ("ZZ", False, False),  # No additional rotations
            ("ZX", False, True),  # H gate on qubit 1 (Bob)
            ("XZ", True, False),  # H gate on qubit 0 (Alice)
            ("XX", True, True),  # H gates on both qubits
        ]

        measurement_bases = (
            self.experiment_params.get("measurement_bases", default_bases)
            if hasattr(self, "experiment_params")
            else default_bases
        )

        for i, result in enumerate(all_results):
            # Determine measurement basis from circuit index
            if i < len(measurement_bases):
                basis_name, _, _ = measurement_bases[i]

                counts = result.get("counts", {})
                measurement_counts[basis_name] = counts

        return measurement_counts

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
        for i, (basis_name, _alice_x, _bob_x) in enumerate(measurement_bases):
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
