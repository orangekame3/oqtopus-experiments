#!/usr/bin/env python3
"""
Ramsey experiment with Noisy Simulator
"""


from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Ramsey


def main():
    print("=== Ramsey with Noisy Simulator ===")

    # Local backend for Qiskit Aer simulator with T2 parameters
    backend = LocalBackend(device="noisy", t1=25.0, t2=15.0)  # T1=25μs, T2=15μs

    # Create Ramsey experiment
    exp = Ramsey(
        experiment_name="ramsey_experiment",
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