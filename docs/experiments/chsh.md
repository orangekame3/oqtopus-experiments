# CHSH Bell Test

CHSH Bell inequality experiment for testing quantum nonlocality.

## Overview

The `CHSH` class implements chsh bell test experiments with automatic circuit generation, data analysis, and visualization.

## Quick Start

```python
from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import CHSH

backend = OqtopusBackend(device="qulacs")

exp = CHSH(
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
| `measurement_angles` | dict[str, float] | Optional | Dictionary of measurement angles for CHSH test |
| `theta` | float | 0.0 | Phase parameter for Bell state preparation |

## Circuit Structure

Four circuits measure Bell state correlations:

```python
# Bell state preparation
qc.h(0)        # Superposition
qc.cx(0, 1)    # Entanglement
qc.ry(θ, 0)    # Alice rotation

# Measurement basis (ZZ, ZX, XZ, XX)
if alice_x: qc.h(0)  # Alice X measurement
if bob_x: qc.h(1)    # Bob X measurement
qc.measure_all()
```

The correlations test Bell inequality violation.
## Analysis and Results

### Bell Inequality Analysis

The experiment calculates CHSH correlation:
```
S = |⟨ZZ⟩ - ⟨ZX⟩ + ⟨XZ⟩ + ⟨XX⟩|
```

**Key Outputs:**
- `chsh_value`: CHSH parameter S
- `bell_violation`: True if S > 2 (quantum behavior)
- `significance`: Statistical significance of violation
## Examples

### Basic Usage

```python
exp = CHSH()
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### Multiple Qubits

```python
results = {}
for qubit in [0, 1, 2]:
    exp = CHSH(physical_qubit=qubit)
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True)
    results[qubit] = df
```

### High Precision

```python
exp = CHSH(
    shots_per_circuit=2000
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

For complete API documentation, see [`CHSH`](../reference/oqtopus_experiments/experiments/chsh.md).

