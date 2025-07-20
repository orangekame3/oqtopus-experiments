#!/usr/bin/env python3
"""
CHSH Bell test with local noisy simulator
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import CHSH

def main():
    # Local backend with realistic noise
    backend = LocalBackend(noise=True, t1=60.0, t2=120.0)
    
    print("Running CHSH experiment with local noisy simulator")
    print(f"Noise model: T1={backend.t1_us}μs, T2={backend.t2_us}μs")
    
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
    print("Running noisy simulation...")
    
    result = chsh.run(backend=backend, shots=4000)
    print("Simulation completed")
    result.analyze()

if __name__ == "__main__":
    main()