#!/usr/bin/env python3
"""
Parallel Rabi experiment with QPU
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi


def main():
    print("=== Parallel Rabi with QPU ===")

    # OQTOPUS backend for QPU
    backend = OqtopusBackend(device="anemone")

    # Create Rabi experiment
    rabi = Rabi(
        experiment_name="rabi_experiment",
        amplitude_points=12,
        max_amplitude=4.0,
    )

    # Sequential execution with backend
    result = rabi.run(backend=backend, shots=1000)

    # Analyze results (defaults to DataFrame)
    df = result.analyze(plot=True, save_data=True, save_image=True)
    print(df.head())


if __name__ == "__main__":
    main()
