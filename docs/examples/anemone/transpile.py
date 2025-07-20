#!/usr/bin/env python3
"""
Anemone transpilation example
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi


def main():
    backend = OqtopusBackend(device="anemone", timeout_seconds=60)

    # Get best qubits
    best_qubits = backend.get_best_qubits(3)
    print("Top qubits:", best_qubits["physical_id"].tolist())

    # Create circuits
    rabi = Rabi()
    circuits = rabi.circuits(amplitude_points=3, max_amplitude=1.0)
    print(f"Created {len(circuits)} circuits")

    # Test transpilation
    transpiled = backend.transpile(
        circuits, physical_qubits=best_qubits["physical_id"].tolist()
    )
    print(transpiled[2].draw())
    print(f"Transpiled {len(transpiled)} circuits")
    print(f"Result type: {type(transpiled)}")


if __name__ == "__main__":
    main()
