# T2 Echo

T2 Echo (Hahn Echo) experiment for T2 measurement.

## Overview

The `T2Echo` class implements t2 echo experiments with automatic circuit generation, data analysis, and visualization.

## Quick Start

```python
from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import T2Echo

backend = OqtopusBackend(device="qulacs")

exp = T2Echo(
    physical_qubit=0,
    delay_points=20,
    max_delay=30000.0,
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
| `delay_points` | int | 20 | Number of delay time points to measure |
| `max_delay` | float | 30000.0 | Maximum delay time in nanoseconds |

## Circuit Structure

Echo sequences refocus dephasing errors:

```python
qc.ry(π/2, 0)              # Initial π/2 pulse
qc.delay(delay/2, 0)       # First half delay
qc.x(0)                    # π pulse (echo)
qc.delay(delay/2, 0)       # Second half delay
qc.ry(π/2, 0)              # Final π/2 pulse
qc.measure(0, 0)
```

Multiple echoes can extend coherence measurement.
## Analysis and Results

### Echo Decay Fitting

The experiment fits data to echo decay:
```
P(|1⟩) = A × exp(-t/T2) + offset
```

**Key Outputs:**
- `t2_time`: True coherence time in nanoseconds
- `amplitude`: Echo efficiency
- `r_squared`: Fit quality (0-1)
## Examples

### Basic Usage

```python
exp = T2Echo()
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### Multiple Qubits

```python
results = {}
for qubit in [0, 1, 2]:
    exp = T2Echo(physical_qubit=qubit)
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True)
    results[qubit] = df
```

### High Precision

```python
exp = T2Echo(
    delay_points=25,
    max_delay=100000.0
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

For complete API documentation, see [`T2Echo`](../reference/oqtopus_experiments/experiments/t2_echo.md).

