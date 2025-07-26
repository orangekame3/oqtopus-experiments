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
        amplitude_points=12,
        max_amplitude=4.0,
    )

    # Sequential execution with backend
    result = rabi.run_parallel(backend=backend, shots=1000)

    # Analyze results (defaults to DataFrame)
    df = result.analyze(plot=True, save_data=True, save_image=True)
    print(df.head())


if __name__ == "__main__":
    main()
