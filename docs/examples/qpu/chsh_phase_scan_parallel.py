#!/usr/bin/env python3
"""
CHSH Phase Scan: S value vs measurement phase on QPU device
Using the dedicated CHSHPhaseScan experiment class
"""

import math

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import CHSHPhaseScan


def main():

    # OQTOPUS backend for QPU
    backend = OqtopusBackend(device="urchin")

    # Create CHSH phase scan experiment
    exp = CHSHPhaseScan(
        experiment_name="chsh_phase_scan_qpu",
        physical_qubit_0=3,
        physical_qubit_1=2,
        shots_per_circuit=1000,
        phase_points=20,
        phase_start=0.0,
        phase_end=math.pi * 3,
    )

    # Run experiment
    result = exp.run_parallel(backend=backend, shots=1000, workers=2)

    # Analyze results
    df = result.analyze()

    print(df.head())


if __name__ == "__main__":
    main()
