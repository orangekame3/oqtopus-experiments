#!/usr/bin/env python3
"""
Ramsey experiment with Noisy Simulator
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Ramsey


def main():
    # Local backend for Qiskit Aer simulator with T2 parameters
    backend = LocalBackend(device="noisy", t1=25.0, t2=15.0)

    # Create Ramsey experiment
    exp = Ramsey(
        experiment_name="ramsey_experiment",
        delay_points=30,
        max_delay=20000.0,
        detuning_frequency=5e6,
    )

    # Run the experiment
    result = exp.run(backend=backend, shots=1000)

    # Analyze results
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
