#!/usr/bin/env python3
"""
Hadamard Test Example (QPU Backend)

This example demonstrates how to use the HadamardTest experiment
with the LocalBackend configured for QPU simulation.
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import HadamardTest


def main():
    """Run Hadamard Test experiment with QPU backend simulation"""

    # Local backend configured for QPU-like behavior
    backend = LocalBackend(device="noisy")

    # Create HadamardTest experiment optimized for hardware
    experiment = HadamardTest(
        experiment_name="hadamard_test_qpu",
        physical_qubit=0,
        angle_points=8,  # Fewer points for hardware efficiency
    )

    # Run experiment with backend
    result = experiment.run(backend=backend, shots=4096)

    # Analyze results
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
