#!/usr/bin/env python3
"""
Parallel CHSH Bell inequality experiment on QPU device
"""

import math

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import CHSH


def main():

    # OQTOPUS backend for QPU
    backend = OqtopusBackend(device="anemone")

    # Create CHSH experiment with optimal theta angle for Bell violation
    exp = CHSH(
        experiment_name="chsh_bell_qpu_parallel",
        shots_per_circuit=2000,
        theta=math.pi / 4,
    )

    # Run CHSH experiment
    result = exp.run_parallel(backend=backend, shots=2000)

    # Analyze results
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
