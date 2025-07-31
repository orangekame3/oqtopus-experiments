# Rabi

Rabi experiment.

## Overview

The `Rabi` class implements rabi experiments with automatic circuit generation, data analysis, and visualization.

## Quick Start

```python
from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi

backend = OqtopusBackend(device="qulacs")

exp = Rabi(
    physical_qubit=0,
    amplitude_points=10,
    max_amplitude=2.0,
)

result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True, save_data=True)
print(df.head())
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | Optional | Optional name for the experiment (used in data files) |
| `physical_qubit` | int | Optional | Target qubit index for hardware execution |
| `amplitude_points` | int | 10 | Number of amplitude values to test |
| `max_amplitude` | float | 2.0 | Maximum drive amplitude in arbitrary units |

## Circuit Structure

Each circuit applies an RX rotation with varying amplitude:

```python
qc = QuantumCircuit(1, 1)
qc.rx(amplitude * π, 0)  # Parameterized rotation
qc.measure(0, 0)
```

The amplitude sweep reveals the π-pulse calibration point.
## Analysis and Results

### Automatic Fitting

The experiment fits data to the Rabi oscillation model:
```
P(|1⟩) = A × sin²(π × amplitude / 2) + offset
```

This formula corresponds to RX(amplitude × π) gates, where the π-pulse occurs at amplitude = 1.

**Key Outputs:**
- `pi_amplitude`: Drive amplitude for π-pulse
- `frequency`: Rabi frequency 
- `r_squared`: Fit quality (0-1)
## Examples

### Basic Usage

```python
exp = Rabi()
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### Multiple Qubits

```python
results = {}
for qubit in [0, 1, 2]:
    exp = Rabi(physical_qubit=qubit)
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True)
    results[qubit] = df
```

### High Precision

```python
exp = Rabi(
    amplitude_points=50,
    max_amplitude=4.0
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

For complete API documentation, see [`Rabi`](../reference/oqtopus_experiments/experiments/rabi.md).

