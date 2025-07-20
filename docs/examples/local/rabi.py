#!/usr/bin/env python3
"""
Rabi experiment with local noisy simulator
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Rabi


def main():
    # Local backend with realistic noise
    backend = LocalBackend(noise=True, t1=50.0, t2=100.0)

    print("Running Rabi experiment with local noisy simulator")
    print(f"Noise model: T1={backend.t1_us}μs, T2={backend.t2_us}μs")

    # Create and run experiment
    rabi = Rabi()
    circuits = rabi.circuits(qubits=[0], amplitude_points=20, max_amplitude=2.5)

    print(f"Created {len(circuits)} circuits")
    print("Running noisy simulation...")

    result = rabi.run(backend=backend, shots=3000)
    print("Simulation completed")
    result.analyze()


if __name__ == "__main__":
    main()
