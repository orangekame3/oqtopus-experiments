# Ramsey

Ramsey fringe experiment for T2* measurement.

## Overview

The `Ramsey` class implements ramsey experiments with automatic circuit generation, data analysis, and visualization.

## Quick Start

```python
from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Ramsey

backend = OqtopusBackend(device="qulacs")

exp = Ramsey(
    physical_qubit=0,
    delay_points=20,
    max_delay=10000.0,
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
| `max_delay` | float | 10000.0 | Maximum delay time in nanoseconds |
| `detuning_frequency` | float | 0.0 | Frequency detuning in Hz for Ramsey experiments |

## Circuit Structure

Each circuit implements Ramsey interferometry:

```python
qc = QuantumCircuit(1, 1)
qc.ry(π/2, 0)              # First π/2 pulse
qc.delay(delay_time, 0)    # Free evolution
qc.rz(detuning_phase, 0)   # Optional detuning
qc.ry(π/2, 0)              # Second π/2 pulse
qc.measure(0, 0)
```

The interference fringes reveal T2* dephasing time.
## Analysis and Results

### Oscillatory Decay Fitting

The experiment fits data to damped oscillation:
```
P(|1⟩) = A × exp(-t/T2*) × cos(2πft + φ) + offset
```

**Key Outputs:**
- `t2_star_time`: Dephasing time in nanoseconds
- `frequency`: Oscillation frequency in Hz
- `phase`: Phase offset in radians
## Examples

### Basic Usage

```python
exp = Ramsey()
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### Multiple Qubits

```python
results = {}
for qubit in [0, 1, 2]:
    exp = Ramsey(physical_qubit=qubit)
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True)
    results[qubit] = df
```

### High Precision

```python
exp = Ramsey(
    delay_points=40,
    max_delay=20000.0,
    detuning_frequency=1e6
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

For complete API documentation, see [`Ramsey`](../reference/oqtopus_experiments/experiments/ramsey.md).

