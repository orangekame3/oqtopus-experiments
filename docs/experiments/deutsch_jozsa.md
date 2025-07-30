# Deutsch-Jozsa Algorithm

## Overview

The Deutsch-Jozsa algorithm is one of the first quantum algorithms to demonstrate exponential speedup over classical deterministic algorithms. It determines whether a black-box Boolean function is constant (returns the same value for all inputs) or balanced (returns 0 for half the inputs and 1 for the other half) with a single oracle query, compared to up to 2^(n-1) + 1 queries required classically.

## Algorithm Description

Given a black-box oracle implementing a Boolean function f: {0,1}^n → {0,1} that is promised to be either constant or balanced, the algorithm determines which type it is with certainty using only one oracle evaluation.

### Quantum Circuit

The algorithm uses the following steps:

1. Initialize n input qubits in |0⟩ and one ancilla qubit in |1⟩
2. Apply Hadamard gates to all qubits to create superposition
3. Apply the oracle function
4. Apply Hadamard gates to input qubits only
5. Measure input qubits - all zeros indicates constant, any other result indicates balanced

## Usage

### Basic Example

```python
from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import DeutschJozsa

# Test a constant function
experiment = DeutschJozsa(
    n_qubits=3,
    oracle_type="constant_0"
)

# Run on local simulator
backend = LocalBackend(device="noiseless")
result = experiment.run(backend=backend, shots=1000)

# Analyze results
df = result.analyze()
```

### Different Oracle Types

```python
# Constant function that always returns 0
exp1 = DeutschJozsa(oracle_type="constant_0")

# Constant function that always returns 1
exp2 = DeutschJozsa(oracle_type="constant_1")

# Balanced function with XOR pattern
exp3 = DeutschJozsa(oracle_type="balanced_alternating")

# Random balanced function
exp4 = DeutschJozsa(oracle_type="balanced_random")
```

## Parameters

- `n_qubits` (int, default=3): Number of input qubits (1-10)
- `oracle_type` (str): Type of oracle function
  - `"constant_0"`: Always returns 0
  - `"constant_1"`: Always returns 1
  - `"balanced_random"`: Random balanced function
  - `"balanced_alternating"`: XOR of all input bits
- `experiment_name` (str, optional): Name for the experiment

## Analysis

The analysis method returns a DataFrame containing:
- `outcome`: Measured bit strings
- `probability`: Measurement probability for each outcome
- `is_all_zeros`: Whether the outcome is the all-zeros string
- `is_constant_measured`: Whether the algorithm determined the function is constant
- `all_zeros_probability`: Probability of measuring |00...0⟩

## Visualization

The experiment creates a histogram showing:
- Measurement outcomes and their counts
- All-zeros outcome highlighted differently
- Results summary with actual vs. measured function type
- 50% threshold line for constant detection

## Algorithm Details

### Oracle Implementation

- **Constant 0**: Identity operation (no gates)
- **Constant 1**: X gate on ancilla
- **Balanced Alternating**: CNOT from each input qubit to ancilla (implements XOR)
- **Balanced Random**: Multi-controlled X gates based on random truth table

### Expected Results

For ideal (noiseless) execution:
- **Constant functions**: Always measure |00...0⟩ (P = 1.0)
- **Balanced functions**: Never measure |00...0⟩ (P = 0.0)

For noisy hardware:
- Use P(|00...0⟩) > 0.5 to classify as constant
- Error rates affect classification accuracy

## Example Results

### Noiseless Simulation
```
Constant function: P(|000⟩) ≈ 1.0
Balanced function: P(|000⟩) ≈ 0.0
```

### Noisy Hardware
```
Constant function: P(|000⟩) ≈ 0.85-0.95
Balanced function: P(|000⟩) ≈ 0.05-0.15
```

## Scaling Behavior

The algorithm scales linearly in circuit depth with the number of qubits:
- Circuit depth: O(1) - constant regardless of n
- Number of gates: O(n) - linear in number of qubits
- Classical queries needed: O(2^n) worst case

This demonstrates the exponential quantum advantage of the Deutsch-Jozsa algorithm.