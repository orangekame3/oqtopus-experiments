#!/usr/bin/env python3
"""
Rabi experiment with Noisy Simulator
"""


from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Rabi


def main():
    print("=== Rabi with Noisy Simulator ===")

    # Local backend for Qiskit Aer simulator
    backend = LocalBackend(device="noisy")

    # Create Rabi experiment
    exp = Rabi(
        experiment_name="parallel_rabi_experiment",
        physical_qubit=3,
        amplitude_points=12,
        max_amplitude=4.0,
    )

    # Parallel execution with backend
    result = exp.run(backend=backend, shots=1000)

    # Analyze results (defaults to DataFrame)
    df = result.analyze(plot=True, save_data=True, save_image=True)
    print(df.head())


if __name__ == "__main__":
    main()
