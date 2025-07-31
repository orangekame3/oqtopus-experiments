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

        # Analysis handled by experiment class for BernsteinVazirani
        # Return the DataFrame directly since analysis is complete
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
