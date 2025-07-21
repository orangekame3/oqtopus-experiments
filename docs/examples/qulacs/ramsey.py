#!/usr/bin/env python3
"""
Ramsey experiment with Qulacs Simulator
"""


from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Ramsey


def main():
    print("=== Ramsey with Qulacs Simulator ===")

    # OQTOPUS backend for Qulacs
    backend = OqtopusBackend(device="qulacs")

    # Create Ramsey experiment
    exp = Ramsey(
        experiment_name="ramsey_qulacs",
        delay_points=30,
        max_delay=20000.0,  # 20μs max delay to see fringe oscillations
        detuning_frequency=5e6,  # 5 MHz detuning for visible fringes
    )

    # Parallel execution with backend
    result = exp.run(backend=backend, shots=1000)

    # Analyze results (defaults to DataFrame)
    df = result.analyze(plot=True, save_data=True, save_image=True)
    print(df.head())


if __name__ == "__main__":
    main()