#!/usr/bin/env python3
"""
CHSH Phase Scan: S value vs measurement phase
Now using the dedicated CHSHPhase experiment class
"""

import math

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import CHSHPhaseScan


def main():
    print("=== CHSH Phase Scan: S vs Phase ===")
    print("Using the new CHSHPhase experiment class for improved analysis")

    # Local backend for ideal simulation
    backend = LocalBackend(device="noisy")

    # Create CHSH phase scan experiment
    exp = CHSHPhaseScan(
        experiment_name="chsh_phase_scan_updated",
        shots_per_circuit=1000,
        phase_points=20,  # Same resolution as before
        phase_start=0.0,
        phase_end=math.pi * 3,  # 0° to 540°
    )

    print(f"Scanning {exp.phase_points} phase points from 0° to 540°")

    # Run experiment with unified interface
    result = exp.run(backend=backend, shots=1000)

    # Analyze results with the new visualization
    df = result.analyze(plot=True, save_data=True, save_image=True)

    print(df.head())


if __name__ == "__main__":
    main()
