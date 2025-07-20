#!/usr/bin/env python3
"""
T1 experiment with local simulator
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import T1


def main():
    # Local backend with noise parameters
    target_t1 = 50.0
    backend = LocalBackend(
        noise=True,
        t1=target_t1,  # T1 time in microseconds
        t2=25.0,  # T2 time in microseconds
    )

    print(f"Target T1: {target_t1} μs")

    # Create T1 experiment
    t1_exp = T1()
    circuits = t1_exp.circuits(delay_points=15, max_delay=150.0)

    print(f"Created {len(circuits)} circuits")
    print(f"Delay range: 0-150 μs")

    # Run experiment
    result = t1_exp.run(backend=backend, shots=2000)
    result.analyze()


if __name__ == "__main__":
    main()
