#!/usr/bin/env python3
"""
T2 Echo experiment on QPU device
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import T2Echo


def main():
    # OQTOPUS backend for QPU
    backend = OqtopusBackend(device="anemone")

    # Create T2 Echo experiment with longer measurement range
    exp = T2Echo(
        experiment_name="t2_echo_qpu",
        delay_points=30,
        max_delay=100000.0,  # 100μs max delay to see full T2 decay
    )

    # Parallel execution with backend
    result = exp.run_parallel(backend=backend, shots=1000)

    # Analyze results
    df = result.analyze()
    print(df.head())


if __name__ == "__main__":
    main()
