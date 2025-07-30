#!/usr/bin/env python3
"""
Deutsch-Jozsa Algorithm Example (Real QPU Hardware)

This example demonstrates the Deutsch-Jozsa algorithm on real quantum hardware
through the OQTOPUS platform. The algorithm determines whether a black-box Boolean
function is constant or balanced with a single oracle query.

Note: Real hardware may have noise and limited connectivity, which can affect results.
"""

from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import DeutschJozsa


def main():
    """Run Deutsch-Jozsa algorithm examples on real quantum hardware"""

    # Initialize backend with real QPU
    backend = OqtopusBackend(device="anemone")

    print("=" * 60)
    print("Deutsch-Jozsa Algorithm - Real QPU Examples")
    print("=" * 60)
    print("Device: anemone (Fujitsu quantum processor)")
    print("Note: Real hardware results may vary due to noise")
    print("=" * 60)

    # Example 1: Small constant function test
    print("\n1. Testing Constant Function (f(x) = 0) - 2 qubits:")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=2,
        oracle_type="constant_0",
        experiment_name="dj_qpu_constant_0_2q",
    )
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: constant_0")
    print("Expected: Constant function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|00⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 2: Small balanced function test
    print("\n2. Testing Balanced Function (XOR) - 2 qubits:")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=2,
        oracle_type="balanced_alternating",
        experiment_name="dj_qpu_balanced_2q",
    )
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: balanced_alternating")
    print("Expected: Balanced function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|00⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 3: 3-qubit constant function with error mitigation
    print("\n3. Testing with Error Mitigation - 3 qubits:")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=3,
        oracle_type="constant_1",
        experiment_name="dj_qpu_constant_1_3q_mitigated",
    )
    # Apply error mitigation
    mitigation_info = {"method": "ReadoutMitigation", "params": {"num_shots": 100}}
    result = exp.run(backend=backend, shots=2000, mitigation_info=mitigation_info)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: constant_1 (with error mitigation)")
    print("Expected: Constant function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 4: 3-qubit balanced function
    print("\n4. Testing 3-qubit Balanced Function:")
    print("-" * 40)
    exp = DeutschJozsa(
        n_qubits=3,
        oracle_type="balanced_random",
        experiment_name="dj_qpu_balanced_3q",
    )
    result = exp.run(backend=backend, shots=2000)
    df = result.analyze(plot=True, save_data=True)
    print("Oracle type: balanced_random")
    print("Expected: Balanced function")
    print(f"Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}")
    print(f"Correct: {df['is_correct'].iloc[0]}")
    print(f"P(|000⟩): {df['all_zeros_probability'].iloc[0]:.3f}")

    # Example 5: Parallel execution for multiple oracle types
    print("\n5. Testing Different Oracle Types in Parallel:")
    print("-" * 40)
    oracle_types = ["constant_0", "constant_1", "balanced_alternating"]

    for oracle_type in oracle_types:
        exp = DeutschJozsa(
            n_qubits=2,
            oracle_type=oracle_type,
            experiment_name=f"dj_qpu_parallel_{oracle_type}",
        )
        result = exp.run_parallel(backend=backend, shots=1000, workers=2)
        df = result.analyze(plot=False, save_data=True)
        print(
            f"Oracle: {oracle_type}, Result: {'Constant' if df['is_constant_measured'].iloc[0] else 'Balanced'}, "
            f"Correct: {df['is_correct'].iloc[0]}"
        )

    print("\n" + "=" * 60)
    print("Deutsch-Jozsa Algorithm QPU Examples Completed!")
    print("=" * 60)
    print("\nNotes on QPU Results:")
    print("- Real quantum hardware has noise and decoherence")
    print("- Error rates increase with circuit depth and qubit count")
    print("- Error mitigation can improve results but not eliminate all errors")
    print("- The Deutsch-Jozsa algorithm is relatively robust to small error rates")


if __name__ == "__main__":
    main()
