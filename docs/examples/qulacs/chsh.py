#!/usr/bin/env python3
"""
CHSH Bell test with Qulacs simulator
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import CHSH

def main():
    # Qulacs-like backend (noiseless simulation)
    backend = LocalBackend(noise=False)
    
    print("Running CHSH experiment with Qulacs simulator")
    
    # Create and run experiment
    chsh = CHSH()
    circuits = chsh.circuits(
        qubits=[0, 1],
        angles_a=[0, 0.25, 0.5, 0.75],
        angles_b=[0.125, 0.375, 0.625, 0.875]
    )
    
    print(f"Created {len(circuits)} circuits")
    print("Classical limit: S ≤ 2.0")
    print("Quantum limit: S ≤ 2.828")
    print("Running noiseless simulation...")
    
    result = chsh.run(backend=backend, shots=3000)
    print("Simulation completed")
    result.analyze()

if __name__ == "__main__":
    main()