#!/usr/bin/env python3
"""
T1 measurement with QPU
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import T1


def main():
    # OQTOPUS backend for QPU
    backend = OqtopusBackend(device="anemone")

    # Create and run experiment
    exp = T1(
        experiment_name="t1_experiment",
        physical_qubit=13,
        delay_points=25,
        max_delay=100000.0,
    )

    result = exp.run_parallel(backend=backend, shots=2000)

    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
