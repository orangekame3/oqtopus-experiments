#!/usr/bin/env python3
"""
CHSH Bell inequality experiment with Local Simulator
"""


from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import CHSH


def main():

    # Local backend for Qiskit Aer simulator
    backend = LocalBackend(device="ideal")

    # Create CHSH experiment
    exp = CHSH(
        experiment_name="chsh_bell_test",
        shots_per_circuit=2000,
    )

    # Run the experiment
    result = exp.run(backend=backend, shots=2000)

    # Analyze results
    df = result.analyze(plot=True, save_data=True, save_image=True)
    print(df.head())


if __name__ == "__main__":
    main()
