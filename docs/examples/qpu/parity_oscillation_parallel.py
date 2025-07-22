#!/usr/bin/env python3
"""
Parity Oscillation Experiment on QPU device
Demonstrates GHZ state decoherence and parity oscillations
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import ParityOscillation


def main():
    print("=== Parity Oscillation: GHZ State Decoherence Study on QPU ===")

    # OQTOPUS backend for QPU
    backend = OqtopusBackend(device="anemone")

    # Create Parity Oscillation experiment with simplified parameters for testing
    for num_qubits in [
        3,
        4,
    ]:
        exp = ParityOscillation(
            experiment_name="parity_oscillation_qpu",
            num_qubits=num_qubits,
            delay_us=0.0,
        )

        print(f"Testing with N={exp.num_qubits} qubits")
        print(f"Delay: {exp.delay_us} μs")
        print(f"Phase points: {exp.phase_points}")

        # Run experiment
        result = exp.run_parallel(backend=backend, shots=3000)

        # Analyze results
        df = result.analyze()
        print(df.head())


if __name__ == "__main__":
    main()
