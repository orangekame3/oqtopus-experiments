#!/usr/bin/env python3
"""
T1 measurement with Qulacs simulator
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import T1

def main():
    # Qulacs-like backend (noiseless simulation)
    backend = LocalBackend(noise=False)
    
    print("Running T1 experiment with Qulacs simulator")
    print("Note: Noiseless simulation shows ideal exponential decay")
    
    # Create and run experiment
    t1 = T1()
    circuits = t1.circuits(
        qubits=[0],
        delay_points=15,
        max_delay=100
    )
    
    print(f"Created {len(circuits)} circuits")
    print("Running noiseless simulation...")
    
    result = t1.run(backend=backend, shots=2000)
    print("Simulation completed")
    result.analyze()

if __name__ == "__main__":
    main()