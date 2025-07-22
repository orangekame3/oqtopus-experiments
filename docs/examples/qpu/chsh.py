#!/usr/bin/env python3
"""
CHSH Bell test on Anemone device
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import CHSH


def main():
    backend = OqtopusBackend(device="anemone", timeout_seconds=180)

    print("Running CHSH experiment on Anemone")

    # Select best qubit pair
    best_qubits = backend.get_best_qubits(2)
    if best_qubits is not None and len(best_qubits) >= 2:
        qubit_a = int(best_qubits.index[0])
        qubit_b = int(best_qubits.index[1])
        print(f"Using qubit pair: ({qubit_a}, {qubit_b})")
    else:
        qubit_a, qubit_b = 0, 1
        print(f"Using default qubit pair: ({qubit_a}, {qubit_b})")

    # Create and run experiment
    chsh = CHSH()
    circuits = chsh.circuits(
        qubits=[qubit_a, qubit_b],
        angles_a=[0, 0.25, 0.5, 0.75],
        angles_b=[0.125, 0.375, 0.625, 0.875],
    )

    print(f"Created {len(circuits)} circuits")
    print("Classical limit: S ≤ 2.0")
    print("Quantum limit: S ≤ 2.828")
    print("Running experiment...")

    try:
        result = chsh.run(backend=backend, shots=3000)
        print("Experiment completed")
        result.analyze()
    except Exception as e:
        print(f"Experiment failed: {e}")


if __name__ == "__main__":
    main()
