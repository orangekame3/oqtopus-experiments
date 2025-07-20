#!/usr/bin/env python3
"""
Parallel Rabi experiment with Qulacs
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi


def main():
    print("=== Parallel Rabi with Qulacs ===")

    # OQTOPUS backend for Qulacs
    backend = OqtopusBackend(device="qulacs")

    # Create Rabi experiment
    rabi = Rabi(
        experiment_name="parallel_rabi_experiment",
        physical_qubit=3,  # Test: should still disable transpilation when specified
        amplitude_points=12,
        max_amplitude=2.0,
    )
    circuits = rabi.circuits()
    print(f"Created {len(circuits)} circuits")
    print("Original circuit:")
    print(circuits[1].draw())

    # # Parallel execution with backend
    result = rabi.run_parallel(backend=backend, shots=1000, workers=4)
    print(result.raw_results)
    result.analyze()


if __name__ == "__main__":
    main()
