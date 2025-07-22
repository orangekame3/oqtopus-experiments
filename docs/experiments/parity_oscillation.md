# Parity Oscillation

Parity oscillation experiment for studying GHZ state decoherence

Measures the coherence C(N, τ) of N-qubit GHZ states as a function of:
- Number of qubits N
- Delay time τ
- Rotation phase φ

The coherence is extracted from parity oscillations amplitude.

## Overview

The `ParityOscillation` class implements parity oscillation experiments with automatic circuit generation, data analysis, and visualization.

## Quick Start

```python
from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import ParityOscillation

backend = OqtopusBackend(device="qulacs")

exp = ParityOscillation(
    num_qubits=2,
    delay_us=0.0,
)

result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True, save_data=True)
print(df.head())
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | Optional | Optional name for the experiment (used in data files) |
| `num_qubits` | int | 2 | Parameter for parity_oscillation experiment |
| `delay_us` | float | 0.0 | Parameter for parity_oscillation experiment |

## Circuit Structure

Multi-qubit GHZ state preparation and measurement:

```python
# GHZ state preparation
qc.h(0)
for i in range(1, n_qubits):
    qc.cx(0, i)

qc.delay(delay_time, range(n_qubits))  # Collective evolution
qc.rz(φ, range(n_qubits))              # Parity rotation
qc.measure_all()
```

Parity measurements reveal multi-qubit decoherence.
## Analysis and Results

Analysis details for parity_oscillation experiment.
## Examples

### Basic Usage

```python
exp = ParityOscillation()
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### Multiple Qubits

```python
results = {}
for qubit in [0, 1, 2]:
    exp = ParityOscillation(physical_qubit=qubit)
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True)
    results[qubit] = df
```

### High Precision

```python
exp = ParityOscillation(
    # High precision parameters
)
result = exp.run(backend=backend, shots=2000)
df = result.analyze(plot=True, save_data=True)
```
## Backend Considerations

### OQTOPUS Platform
```python
# Real quantum hardware
backend = OqtopusBackend()

# Fast noiseless simulation
backend = OqtopusBackend(device="qulacs")
```

### Local Simulation
```python
from oqtopus_experiments.backends import LocalBackend

# Realistic noisy simulation
backend = LocalBackend(device="noisy")
result = exp.run(backend=backend, shots=1000)
```

## API Reference

For complete API documentation, see [`ParityOscillation`](../reference/oqtopus_experiments/experiments/parity_oscillation.md).

