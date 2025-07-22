# Chsh Phase Scan

CHSH phase scan experiment for studying Bell violation vs measurement phase.

## Overview

The `CHSHPhaseScan` class implements chsh phase scan experiments with automatic circuit generation, data analysis, and visualization.

## Quick Start

```python
from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import CHSHPhaseScan

backend = OqtopusBackend(device="qulacs")

exp = CHSHPhaseScan(
    physical_qubit_0=0,
    physical_qubit_1=0,
    shots_per_circuit=1000,
)

result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True, save_data=True)
print(df.head())
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | Optional | Optional name for the experiment (used in data files) |
| `physical_qubit_0` | int | Optional | First qubit index for two-qubit experiments |
| `physical_qubit_1` | int | Optional | Second qubit index for two-qubit experiments |
| `shots_per_circuit` | int | 1000 | Number of measurement shots per circuit |
| `phase_points` | int | 21 | Parameter for chsh_phase_scan experiment |
| `phase_start` | float | 0.0 | Parameter for chsh_phase_scan experiment |
| `phase_end` | float | 2 * math.pi | Parameter for chsh_phase_scan experiment |

## Circuit Structure

Circuit structure for chsh_phase_scan experiment.
## Analysis and Results

Analysis details for chsh_phase_scan experiment.
## Examples

### Basic Usage

```python
exp = CHSHPhaseScan()
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### Multiple Qubits

```python
results = {}
for qubit in [0, 1, 2]:
    exp = CHSHPhaseScan(physical_qubit=qubit)
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True)
    results[qubit] = df
```

### High Precision

```python
exp = CHSHPhaseScan(
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

For complete API documentation, see [`CHSHPhaseScan`](../reference/oqtopus_experiments/experiments/chsh_phase_scan.md).

