#!/usr/bin/env python3
"""
CHSH Bell inequality experiment with Qulacs Simulator
"""

import math

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import CHSH


def main():
    print("=== CHSH Bell Inequality Test with Qulacs ===")

    # OQTOPUS backend for Qulacs
    backend = OqtopusBackend(device="qulacs")

    # Create CHSH experiment with optimal theta angle for Bell violation
    exp = CHSH(
        experiment_name="chsh_bell_qulacs",
        shots_per_circuit=2000,  # More shots for better statistics
        theta=math.pi / 4,
    )

    # Run CHSH experiment with optimal theta = π/4 (45°) for maximum CHSH violation

    result = exp.run_parallel(backend=backend, shots=2000)

    # Analyze results
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
