#!/usr/bin/env python3
"""
Parallel Rabi experiment on QPU device
"""


from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi


def main():

    # OQTOPUS backend for QPU
    backend = OqtopusBackend(device="anemone")

    # Create Rabi experiment
    rabi = Rabi(
        experiment_name="parallel_rabi_experiment",
        physical_qubit=34,
        amplitude_points=12,
        max_amplitude=4.0,
    )

    # Sequential execution with backend
    result = rabi.run_parallel(backend=backend, shots=1000)

    # Analyze results (defaults to DataFrame)
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
