#!/usr/bin/env python3
"""
T1 measurement on Anemone device
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import T1

def main():
    backend = OqtopusBackend(device="anemone", timeout_seconds=150)
    
    print("Running T1 experiment on Anemone")
    
    # Select best qubit
    best_qubits = backend.get_best_qubits(1)
    if best_qubits is not None:
        target_qubit = int(best_qubits.index[0])
        expected_t1 = best_qubits.iloc[0].get('t1_us', 'Unknown')
        print(f"Using qubit {target_qubit}, expected T1: {expected_t1} μs")
    else:
        target_qubit = 0
        expected_t1 = 100
        print(f"Using default qubit {target_qubit}")
    
    # Create and run experiment
    t1 = T1()
    max_delay = 150 if expected_t1 == "Unknown" else min(expected_t1 * 3, 200)
    
    circuits = t1.circuits(
        qubits=[target_qubit],
        delay_points=15,
        max_delay=max_delay
    )
    
    print(f"Created {len(circuits)} circuits, max delay: {max_delay} μs")
    print("Running experiment...")
    
    try:
        result = t1.run(backend=backend, shots=2000)
        print("Experiment completed")
        result.analyze()
    except Exception as e:
        print(f"Experiment failed: {e}")

if __name__ == "__main__":
    main()