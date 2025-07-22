#!/usr/bin/env python3
"""
Parallel Rabi experiment on QPU device
"""


from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi


def main():
    print("=== Parallel Rabi on QPU ===")

    # OQTOPUS backend for QPU
    backend = OqtopusBackend(device="anemone")

    # Create Rabi experiment
    rabi = Rabi(
        experiment_name="parallel_rabi_experiment",
        physical_qubit=3,
        amplitude_points=12,
        max_amplitude=4.0,
    )

    # Parallel execution with backend
    result = rabi.run_parallel(backend=backend, shots=1000, workers=4)

    # Analyze results (defaults to DataFrame)
    df = result.analyze(plot=True, save_data=True, save_image=True)
    print(df.head())


if __name__ == "__main__":
    main()