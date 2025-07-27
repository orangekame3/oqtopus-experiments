# Bernstein-Vazirani Algorithm

## Overview

The Bernstein-Vazirani algorithm is a quantum algorithm that demonstrates quantum advantage by finding an n-bit secret string with a single quantum query to an oracle, compared to n queries required classically.

## Algorithm Description

Given a black-box oracle that computes the Boolean function f_s(x) = s·x (dot product mod 2), where s is an unknown n-bit secret string, the algorithm finds s with probability 1 using only one oracle query.

### Quantum Circuit

The algorithm uses the following steps:

1. Initialize n input qubits in |0⟩ and one ancilla qubit in |1⟩
2. Apply Hadamard gates to all qubits
3. Apply the oracle (implemented as CNOTs controlled by bits where s_i = 1)
4. Apply Hadamard gates to input qubits
5. Measure input qubits to obtain the secret string

## Usage

### Basic Example

```python
from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import BernsteinVazirani

# Create experiment with a specific secret string
experiment = BernsteinVazirani(secret_string="1101")

# Run on local simulator
backend = LocalBackend(device="noiseless")
result = experiment.run(backend=backend, shots=1000)

# Analyze results
df = result.analyze()
```

### Random Secret String

```python
# Generate random n-bit secret
experiment = BernsteinVazirani(n_bits=6)
```

## Parameters

- `experiment_name` (str, optional): Name for the experiment
- `secret_string` (str, optional): Binary string to find (e.g., "1011")
- `n_bits` (int): Number of bits in the secret string (used if secret_string is None)

## Analysis

The analysis method returns a DataFrame containing:
- `outcome`: Measured bit strings (displayed in human-readable big-endian format)
- `probability`: Measurement probability for each outcome
- `is_secret`: Whether the outcome matches the secret
- `success_probability`: Probability of measuring the correct secret

**Note on Bit Ordering**: Qiskit returns measurement results in little-endian format (qubit 0 is right-most). The implementation automatically handles this conversion to display results in the standard big-endian format for easy comparison with the input secret string.

## Visualization

The experiment creates a bar chart showing:
- Measurement outcomes and their probabilities
- The secret string highlighted in a different color
- Success probability and correctness status

## Example Results

For a noiseless simulation, you should see:
- Success probability ≈ 1.0
- Single peak at the secret string
- No other measurement outcomes

For noisy hardware:
- Success probability < 1.0
- Multiple measurement outcomes
- The secret string should still have the highest probability