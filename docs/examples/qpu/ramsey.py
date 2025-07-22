#!/usr/bin/env python3
"""
Ramsey experiment on QPU device
"""


from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Ramsey


def main():
    print("=== Ramsey on QPU ===")

    # OQTOPUS backend for QPU
    backend = OqtopusBackend(device="anemone")

    # Create Ramsey experiment
    exp = Ramsey(
        experiment_name="ramsey_qpu",
        delay_points=30,
        max_delay=20000.0,  # 20μs max delay to see fringe oscillations
        detuning_frequency=5e6,  # 5 MHz detuning for visible fringes
    )

    # Parallel execution with backend
    result = exp.run(backend=backend, shots=1000)

    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
