#!/usr/bin/env python3
"""
Compare noisy vs noiseless local simulation
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Rabi


def main():
    print("Comparing noisy vs noiseless simulation")

    # Noisy backend
    noisy_backend = LocalBackend(noise=True, t1=50.0, t2=100.0)

    # Noiseless backend
    clean_backend = LocalBackend(noise=False)

    # Run noisy experiment
    print("\n1. Running noisy simulation...")
    rabi_noisy = Rabi()
    circuits_noisy = rabi_noisy.circuits(
        qubits=[0], amplitude_points=15, max_amplitude=2.0
    )
    result_noisy = rabi_noisy.run(backend=noisy_backend, shots=3000)

    print("Noisy results:")
    result_noisy.analyze()

    # Run clean experiment
    print("\n2. Running noiseless simulation...")
    rabi_clean = Rabi()
    circuits_clean = rabi_clean.circuits(
        qubits=[0], amplitude_points=15, max_amplitude=2.0
    )
    result_clean = rabi_clean.run(backend=clean_backend, shots=3000)

    print("Noiseless results:")
    result_clean.analyze()

    print("\nComparison complete")


if __name__ == "__main__":
    main()
