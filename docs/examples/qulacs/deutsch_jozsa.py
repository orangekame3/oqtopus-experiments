#!/usr/bin/env python3
"""
Deutsch-Jozsa Algorithm Example (Qulacs Simulator via OQTOPUS)

This example demonstrates the Deutsch-Jozsa algorithm using the Qulacs simulator
through the OQTOPUS platform. The algorithm determines whether a black-box Boolean
function is constant or balanced with a single oracle query.
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import DeutschJozsa


def main():
    """Run Deutsch-Jozsa algorithm examples using Qulacs simulator"""

    # Initialize backend with Qulacs simulator
    backend = OqtopusBackend(device="qulacs")

    print("=" * 60)
    print("Deutsch-Jozsa Algorithm - Qulacs Simulator Examples")
    print("=" * 60)

    # Example 1: Constant function (always returns 0)
    print("\n1. Testing Constant Function (f(x) = 0):")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=4,
        oracle_type="constant_0",
        experiment_name="dj_qulacs_constant_0",
    )
    result = exp.run(backend=backend, shots=2000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: constant_0")
    print("Expected: Constant function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|0000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 2: Constant function (always returns 1)
    print("\n2. Testing Constant Function (f(x) = 1):")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=4,
        oracle_type="constant_1",
        experiment_name="dj_qulacs_constant_1",
    )
    result = exp.run(backend=backend, shots=2000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: constant_1")
    print("Expected: Constant function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|0000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 3: Balanced function (alternating - XOR of all bits)
    print("\n3. Testing Balanced Function (f(x) = XOR of all bits):")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=4,
        oracle_type="balanced_alternating",
        experiment_name="dj_qulacs_balanced_alternating",
    )
    result = exp.run(backend=backend, shots=2000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: balanced_alternating")
    print("Expected: Balanced function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|0000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 4: Balanced function (random)
    print("\n4. Testing Balanced Function (random balanced):")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=4,
        oracle_type="balanced_random",
        experiment_name="dj_qulacs_balanced_random",
    )
    result = exp.run(backend=backend, shots=2000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: balanced_random")
    print("Expected: Balanced function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|0000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 5: Larger problem size test
    print("\n5. Testing with Larger Problem Size (6 qubits):")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=6,
        oracle_type="balanced_alternating",
        experiment_name="dj_qulacs_6qubits",
    )
    result = exp.run(backend=backend, shots=2000)
    df = result.analyze(plot=True, save_data=True)
    print("n_qubits=6, oracle=balanced_alternating")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|000000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 6: Parallel execution test
    print("\n6. Testing Parallel Execution:")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=5,
        oracle_type="balanced_random",
        experiment_name="dj_qulacs_parallel",
    )
    # Note: For DJ algorithm, we only have one circuit, but we can still use parallel execution
    result = exp.run_parallel(backend=backend, shots=5000, workers=4)
    df = result.analyze(plot=True, save_data=True)
    print("Parallel execution with 5 qubits")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")

    print("\n" + "=" * 60)
    print("Deutsch-Jozsa Algorithm Qulacs Examples Completed!")
    print("=" * 60)
    print("\nNote: Qulacs provides noiseless simulation, so results should be")
    print("nearly perfect for the Deutsch-Jozsa algorithm.")


if __name__ == "__main__":
    main()
