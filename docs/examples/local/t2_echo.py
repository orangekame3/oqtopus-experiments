#!/usr/bin/env python3
"""
T2 Echo experiment with Noisy Simulator
"""


from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import T2Echo


def main():
    print("=== T2 Echo with Noisy Simulator ===")

    # Local backend for Qiskit Aer simulator with shorter T2 time
    backend = LocalBackend(device="noisy", t1=25.0, t2=8.0)  # T1=25μs, T2=8μs

    # Create T2 Echo experiment with longer measurement range
    exp = T2Echo(
        experiment_name="t2_echo_experiment",
        delay_points=30,
        max_delay=100000.0,  # 100μs max delay to see full T2 decay
    )

    # Parallel execution with backend
    result = exp.run(backend=backend, shots=1000)

    # Analyze results (defaults to DataFrame)
    df = result.analyze(plot=True, save_data=True, save_image=True)
    print(df.head())


if __name__ == "__main__":
    main()
