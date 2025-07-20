#!/usr/bin/env python3
"""
Rabi experiment with Qulacs simulator
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Rabi


def main():
    # Qulacs-like backend (noiseless simulation)
    backend = LocalBackend(noise=False)

    print("Running Rabi experiment with Qulacs simulator")

    # Create and run experiment
    rabi = Rabi(physical_qubit=0, amplitude_points=20, max_amplitude=2.0)
    circuits = rabi.circuits()

    print(f"Created {len(circuits)} circuits")
    print("Running noiseless simulation...")

    result = rabi.run(backend=backend, shots=1000)
    print("Simulation completed")
    result.analyze()


if __name__ == "__main__":
    main()
