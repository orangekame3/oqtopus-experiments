#!/usr/bin/env python3
"""
Hadamard Test Example (Local Backend)

This example demonstrates how to use the HadamardTest experiment
to measure expectation values of unitary operators using the
LocalBackend.

The Hadamard Test is a fundamental quantum algorithm for measuring
expectation values ⟨ψ|U|ψ⟩ using ancilla qubits and quantum interference.
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import HadamardTest


def main():
    """Run Hadamard Test experiment with LocalBackend"""

    # Local backend for Qiskit Aer simulator
    backend = LocalBackend(device="noisy")

    # Create HadamardTest experiment
    experiment = HadamardTest(
        experiment_name="hadamard_test_local",
        physical_qubit=0,
        angle_points=16,
    )

    # Run experiment with backend
    result = experiment.run(backend=backend, shots=1000)

    # Analyze results
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
