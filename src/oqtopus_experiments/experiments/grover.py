#!/usr/bin/env python3
"""
Grover's Quantum Search Algorithm Experiment Class
"""

import math
import random
from typing import Any

import pandas as pd
from qiskit import QuantumCircuit

from ..core.base_experiment import BaseExperiment
from ..models.grover_models import GroverData, GroverParameters, GroverResult


class Grover(BaseExperiment):
    """
    Grover's quantum search algorithm experiment

    Implements Grover's algorithm to find marked items in an unsorted database
    with quadratic speedup over classical algorithms. The algorithm amplifies
    the probability amplitudes of marked states through repeated application
    of the Grover operator (oracle + diffuser).
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        n_qubits: int = 2,
        marked_states: list[int] | str = "random",
        num_iterations: int | None = None,
    ):
        """
        Initialize Grover's algorithm experiment

        Args:
            experiment_name: Experiment name
            n_qubits: Number of qubits in the search space (2-8 recommended)
            marked_states: List of marked state indices or "random" for random selection
            num_iterations: Number of Grover iterations (None for optimal)
        """
        self.params = GroverParameters(
            experiment_name=experiment_name,
            n_qubits=n_qubits,
            marked_states=marked_states,
            num_iterations=num_iterations,
        )

        super().__init__(self.params.experiment_name or "grover_experiment")

        self.n_qubits = self.params.n_qubits
        self.search_space_size = 2**self.n_qubits

        # Resolve marked states
        if (
            isinstance(self.params.marked_states, str)
            and self.params.marked_states == "random"
        ):
            # Select random marked states (typically 1-2 for demonstration)
            num_marked = min(2, max(1, self.search_space_size // 4))
            self.marked_states = random.sample(
                range(self.search_space_size), num_marked
            )
        else:
            # Cast to list[int] for type safety
            if isinstance(self.params.marked_states, list):
                self.marked_states = self.params.marked_states
            else:
                # This should never happen due to Pydantic validation, but for type safety
                self.marked_states = []

        # Validate marked states
        if not all(0 <= state < self.search_space_size for state in self.marked_states):
            raise ValueError(
                f"Marked states must be in range [0, {self.search_space_size - 1}]"
            )

        # Calculate optimal number of iterations
        if len(self.marked_states) > 0:
            self.optimal_iterations = int(
                (math.pi / 4)
                * math.sqrt(self.search_space_size / len(self.marked_states))
            )
        else:
            self.optimal_iterations = 0

        # Set actual iterations
        self.num_iterations = (
            self.params.num_iterations
            if self.params.num_iterations is not None
            else self.optimal_iterations
        )

        print(
            f"Grover Search: {self.n_qubits} qubits, {len(self.marked_states)} marked states"
        )
        print(f"Marked states: {self.marked_states}")
        print(f"Iterations: {self.num_iterations} (optimal: {self.optimal_iterations})")

    def _create_oracle(self, qc: QuantumCircuit) -> None:
        """
        Create oracle that flips the phase of marked states

        The oracle implements a reflection about the marked states by applying
        a negative phase (-1) to the marked computational basis states.
        """
        for marked_state in self.marked_states:
            # Convert state to binary representation
            binary_state = format(marked_state, f"0{self.n_qubits}b")

            # Apply X gates to qubits that should be 0 in the marked state
            for i, bit in enumerate(binary_state):
                if bit == "0":
                    qc.x(i)

            # Apply multi-controlled Z gate (phase flip)
            if self.n_qubits == 1:
                qc.z(0)
            elif self.n_qubits == 2:
                qc.cz(0, 1)
            else:
                # Multi-controlled Z using ancilla qubit approach
                # For simplicity, we'll use a decomposition with CX and single-qubit gates
                self._multi_controlled_z(qc)

            # Undo X gates
            for i, bit in enumerate(binary_state):
                if bit == "0":
                    qc.x(i)

    def _multi_controlled_z(self, qc: QuantumCircuit) -> None:
        """
        Apply multi-controlled Z gate for oracle implementation

        Uses a decomposition approach suitable for NISQ devices.
        For larger systems, this should be optimized based on the target hardware.
        """
        if self.n_qubits <= 2:
            if self.n_qubits == 2:
                qc.cz(0, 1)
            else:
                qc.z(0)
        elif self.n_qubits == 3:
            # 3-qubit controlled Z using Toffoli decomposition
            qc.ccx(0, 1, 2)
            qc.z(2)
            qc.ccx(0, 1, 2)
        else:
            # For larger systems, use a cascaded approach
            # This is not the most gate-efficient but works on NISQ devices
            control_qubits = list(range(self.n_qubits - 1))
            target_qubit = self.n_qubits - 1

            # Apply controlled-controlled-...-Z
            # Simplified implementation using MCX decomposition
            qc.h(target_qubit)
            for i in range(len(control_qubits)):
                if i == 0:
                    qc.cx(control_qubits[i], target_qubit)
                else:
                    qc.ccx(control_qubits[i - 1], control_qubits[i], target_qubit)
            qc.h(target_qubit)

    def _create_diffuser(self, qc: QuantumCircuit) -> None:
        """
        Create diffuser (inversion about average) operator

        The diffuser reflects the amplitudes about their average value,
        implemented as 2|s⟩⟨s| - I where |s⟩ is the equal superposition state.
        """
        # Apply Hadamard to all qubits
        for i in range(self.n_qubits):
            qc.h(i)

        # Apply X to all qubits
        for i in range(self.n_qubits):
            qc.x(i)

        # Apply multi-controlled Z (phase flip for |00...0⟩ state)
        if self.n_qubits == 1:
            qc.z(0)
        elif self.n_qubits == 2:
            qc.cz(0, 1)
        else:
            self._multi_controlled_z(qc)

        # Apply X to all qubits
        for i in range(self.n_qubits):
            qc.x(i)

        # Apply Hadamard to all qubits
        for i in range(self.n_qubits):
            qc.h(i)

    def _create_grover_circuit(self) -> QuantumCircuit:
        """Create the complete Grover's algorithm circuit"""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)

        # Initialize superposition (apply Hadamard to all qubits)
        for i in range(self.n_qubits):
            qc.h(i)

        # Add barrier for clarity
        qc.barrier()

        # Apply Grover operator (oracle + diffuser) for specified iterations
        for iteration in range(self.num_iterations):
            # Oracle: flip phase of marked states
            self._create_oracle(qc)
            qc.barrier(label=f"Oracle {iteration + 1}")

            # Diffuser: inversion about average
            self._create_diffuser(qc)
            qc.barrier(label=f"Diffuser {iteration + 1}")

        # Measure all qubits
        for i in range(self.n_qubits):
            qc.measure(i, i)

        return qc

    def circuits(self, **kwargs) -> list[QuantumCircuit]:
        """Generate Grover's algorithm circuit"""
        qc = self._create_grover_circuit()
        qc.name = f"grover_{self.n_qubits}q_{len(self.marked_states)}marked_{self.num_iterations}iter"
        return [qc]

    def analyze(
        self,
        results: dict[str, list[dict[str, Any]]],
        plot: bool = True,
        save_data: bool = True,
        save_image: bool = True,
    ) -> pd.DataFrame:
        """Analyze Grover's algorithm results"""
        try:
            # Flatten all results into single list
            all_results = []
            for device_data in results.values():
                all_results.extend(device_data)

            if not all_results:
                return pd.DataFrame(
                    {
                        "state": list(range(self.search_space_size)),
                        "error": ["No results available"] * self.search_space_size,
                    }
                )

            # Extract counts from results (assuming single circuit)
            counts = all_results[0].get("counts", {}) if all_results else {}
            total_shots = sum(counts.values()) if counts else 0

            # Create GroverData
            grover_data = GroverData(
                n_qubits=self.n_qubits,
                marked_states=self.marked_states,
                num_iterations=self.num_iterations,
                measurement_counts=counts,
                total_shots=total_shots,
                search_space_size=self.search_space_size,
            )

            # Create result object and analyze
            grover_result = GroverResult(
                raw_results=results,
                experiment_instance=self,
                data=grover_data,
                backend="unknown",
                device="unknown",
                shots=total_shots,
                metadata={"experiment_type": "grover"},
            )

            # Perform analysis
            df = grover_result.analyze(
                plot=plot,
                save_data=False,  # Disable model's direct save
                save_image=save_image,
            )

            # Use standard experiment data saving if requested
            if save_data:
                self._save_grover_analysis(df, grover_result.data.analysis_result)

            return df

        except Exception as e:
            print(f"Analysis failed: {e}")
            return pd.DataFrame(
                {
                    "state": list(range(self.search_space_size)),
                    "error": [str(e)] * self.search_space_size,
                }
            )

    def _save_grover_analysis(self, df, analysis_result):
        """Save Grover analysis results using standard experiment data saving."""
        try:
            # Convert DataFrame to records for saving
            analysis_data = df.to_dict(orient="records")

            # Save using standard experiment data saving
            saved_path = self.save_experiment_data(
                analysis_data,
                metadata={
                    "grover_summary": {
                        "n_qubits": self.n_qubits,
                        "marked_states": self.marked_states,
                        "num_iterations": self.num_iterations,
                        "optimal_iterations": self.optimal_iterations,
                        "success_probability": (
                            analysis_result.success_probability
                            if analysis_result
                            else 0
                        ),
                        "theoretical_success_probability": (
                            analysis_result.theoretical_success_probability
                            if analysis_result
                            else 0
                        ),
                        "success_rate_error": (
                            analysis_result.success_rate_error if analysis_result else 0
                        ),
                    },
                    "experiment_type": "grover",
                    "search_space_size": self.search_space_size,
                },
                experiment_type="grover",
            )
            print(f"📊 Saved Grover analysis: {saved_path}")

        except Exception as e:
            print(f"Warning: Could not save Grover analysis data: {e}")
