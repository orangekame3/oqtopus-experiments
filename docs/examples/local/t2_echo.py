#!/usr/bin/env python3
"""
T2 Echo experiment with Noisy Simulator
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import T2Echo


def main():
    # Local backend for Qiskit Aer simulator with shorter T2 time
    backend = LocalBackend(device="noisy", t1=25.0, t2=8.0)

    # Create T2 Echo experiment with longer measurement range
    exp = T2Echo(
        experiment_name="t2_echo_experiment",
        delay_points=30,
        max_delay=100000.0,
    )

    # Run the experiment
    result = exp.run(backend=backend, shots=1000)

    # Analyze results
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
