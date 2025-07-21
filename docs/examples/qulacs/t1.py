#!/usr/bin/env python3
"""
T1 measurement with Qulacs simulator
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import T1


def main():
    # Qulacs-like backend (noiseless simulation)
    backend = OqtopusBackend(device="qulacs")

    print("Running T1 experiment with Qulacs simulator")
    print("Note: Noiseless simulation shows ideal exponential decay")

    # Create and run experiment
    exp = T1(
        experiment_name="t1_experiment",
        delay_points=25,
        max_delay=100000.0,  # 150μs max delay to see full decay
    )

    result = exp.run(backend=backend, shots=2000)
    print("Simulation completed")
    result.analyze()


if __name__ == "__main__":
    main()
