#!/usr/bin/env python3
"""
Deutsch-Jozsa Algorithm Example (Local Simulator)

This example demonstrates the Deutsch-Jozsa algorithm on a local quantum simulator.
The algorithm determines whether a black-box Boolean function is constant or balanced
with a single oracle query.
"""

from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import DeutschJozsa


def main():
    """Run Deutsch-Jozsa algorithm examples with different oracle types"""

    # Initialize backend
    backend = LocalBackend(device="noisy")

    print("=" * 60)
    print("Deutsch-Jozsa Algorithm - Local Simulator Examples")
    print("=" * 60)

    # Example 1: Constant function (always returns 0)
    print("\n1. Testing Constant Function (f(x) = 0):")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=3,
        oracle_type="constant_0",
        experiment_name="dj_constant_0",
    )
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: constant_0")
    print("Expected: Constant function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 2: Constant function (always returns 1)
    print("\n2. Testing Constant Function (f(x) = 1):")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=3,
        oracle_type="constant_1",
        experiment_name="dj_constant_1",
    )
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: constant_1")
    print("Expected: Constant function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 3: Balanced function (alternating - XOR of all bits)
    print("\n3. Testing Balanced Function (f(x) = XOR of all bits):")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=3,
        oracle_type="balanced_alternating",
        experiment_name="dj_balanced_alternating",
    )
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: balanced_alternating")
    print("Expected: Balanced function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 4: Balanced function (random)
    print("\n4. Testing Balanced Function (random balanced):")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=3,
        oracle_type="balanced_random",
        experiment_name="dj_balanced_random",
    )
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: balanced_random")
    print("Expected: Balanced function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 5: Scaling test with different qubit numbers
    print("\n5. Scaling Test (different qubit numbers):")
    print("-" * 40)
    for n in [2, 4, 5]:
        exp = DeutschJozsa(
            n_qubits=n,
            oracle_type="balanced_alternating",
            experiment_name=f"dj_scaling_{n}qubits",
        )
        result = exp.run(backend=backend, shots=1000)
        df = result.analyze(plot=False, save_data=False)
        print(
            f"n_qubits={n}: Result={'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}, "
            f"Correct={df['is_correct'].iloc[0]}"
        )

    # Example 6: Noisy simulator test
    print("\n6. Testing with Noisy Simulator:")
    print("-" * 40)
    noisy_backend = LocalBackend(device="noisy")
    exp = DeutschJozsa(
        n_qubits=3,
        oracle_type="constant_0",
        experiment_name="dj_noisy_test",
    )
    result = exp.run(backend=noisy_backend, shots=1000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: constant_0 (with noise)")
    print("Expected: Constant function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")
    print("Note: Noise may affect the accuracy of the result")

    print("\n" + "=" * 60)
    print("Deutsch-Jozsa Algorithm Examples Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
