#!/usr/bin/env python3
"""
Parallel execution demo - effect of circuit count
"""

import time

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Rabi


def test_parallel_scaling(backend, circuit_counts):
    """Test parallel scaling with different circuit counts"""

    for count in circuit_counts:
        print(f"\n--- {count} circuits ---")

        # Create circuits
        rabi = Rabi()
        circuits = rabi.circuits(amplitude_points=count, max_amplitude=2.0)

        # Sequential execution
        start_time = time.time()
        result_seq = rabi.run(backend=backend, shots=300)
        seq_time = time.time() - start_time

        # Parallel execution
        start_time = time.time()
        result_par = rabi.run_parallel(backend=backend, shots=300, workers=4)
        par_time = time.time() - start_time

        speedup = seq_time / par_time
        print(f"Sequential: {seq_time:.2f}s")
        print(f"Parallel:   {par_time:.2f}s")
        print(f"Speedup:    {speedup:.2f}x")


def main():
    backend = LocalBackend(noise=True, t1=30.0, t2=15.0)

    print("=== Parallel Scaling Demo ===")
    print("Testing speedup with different circuit counts")

    # Test with increasing circuit counts
    circuit_counts = [4, 16, 24, 94, 128, 256]
    test_parallel_scaling(backend, circuit_counts)


if __name__ == "__main__":
    main()
