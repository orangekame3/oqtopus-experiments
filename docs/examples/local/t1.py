#!/usr/bin/env python3
"""
T1 experiment with Noisy Simulator
"""


from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import T1


def main():

    # Local backend for Qiskit Aer simulator with shorter T1 time
    backend = LocalBackend(device="noisy")

    # Create T1 experiment with appropriate measurement range
    exp = T1(
        experiment_name="t1_experiment",
        delay_points=25,
        max_delay=100000.0,
    )
    circuits = exp.circuits()
    print(circuits[2].draw())

    # Run the experiment
    result = exp.run(backend=backend, shots=1000)

    # Analyze results
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
