#!/usr/bin/env python3
"""
Hadamard Test Example (Qulacs Backend)

This example demonstrates how to use the HadamardTest experiment
with the LocalBackend for Qulacs-style simulation.
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import HadamardTest


def main():
    """Run Hadamard Test experiment with Qulacs-style backend"""

    # Local backend for Qulacs-style simulation
    backend = LocalBackend(device="ideal")  # Use ideal for high-performance simulation

    # Create HadamardTest experiment
    experiment = HadamardTest(
        experiment_name="hadamard_test_qulacs",
        physical_qubit=0,
        angle_points=20,  # More points for detailed analysis
    )

    # Run experiment with backend
    result = experiment.run(
        backend=backend, shots=10000
    )  # More shots for better statistics

    # Analyze results
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
