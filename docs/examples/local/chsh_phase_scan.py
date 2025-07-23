#!/usr/bin/env python3
"""
CHSH Phase Scan: S value vs measurement phase
Now using the dedicated CHSHPhase experiment class
"""

import math

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import CHSHPhaseScan


def main():

    # Local backend for Qiskit Aer simulator with noise
    backend = LocalBackend(device="noisy")
    # backend = OqtopusBackend("anemone"))

    # Create CHSH phase scan experiment
    exp = CHSHPhaseScan(
        experiment_name="chsh_phase_scan_updated",
        shots_per_circuit=1000,
        phase_points=20,
        phase_start=0.0,
        phase_end=math.pi * 3,
    )

    # Run the experiment
    result = exp.run(backend=backend, shots=1000)

    # Analyze results
    df = result.analyze()

    print(df.head())


if __name__ == "__main__":
    main()
