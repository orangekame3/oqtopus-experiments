#!/usr/bin/env python3
"""
Rabi experiment on Anemone device
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi


def main():
    backend = OqtopusBackend(device="anemone", timeout_seconds=120)

    print("Running Rabi experiment on Anemone")

    # Select best qubit
    best_qubits = backend.get_best_qubits(1)
    if best_qubits is not None:
        target_qubit = int(best_qubits.index[0])
        print(f"Using qubit {target_qubit}")
    else:
        target_qubit = 0
        print(f"Using default qubit {target_qubit}")

    # Create and run experiment
    rabi = Rabi()
    circuits = rabi.circuits(
        qubits=[target_qubit], amplitude_points=15, max_amplitude=2.0
    )

    print(f"Created {len(circuits)} circuits")
    print("Running experiment...")

    try:
        result = rabi.run(backend=backend, shots=1000)
        print("Experiment completed")
        result.analyze()
    except Exception as e:
        print(f"Experiment failed: {e}")


if __name__ == "__main__":
    main()
