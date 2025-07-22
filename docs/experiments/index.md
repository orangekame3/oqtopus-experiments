# Experiments Overview

OQTOPUS Experiments provides a comprehensive suite of quantum experiments for characterization, benchmarking, and research. Each experiment is designed with a consistent API and supports multiple backends.

## Available Experiments

### Qubit Characterization

| Experiment | Purpose | Key Parameters |
|-----------|---------|----------------|
| [**Rabi**](rabi.md) | π-pulse calibration | `amplitude_points`, `max_amplitude` |
| [**T1**](t1.md) | Energy relaxation time | `delay_points`, `max_delay` |
| [**T2 Echo**](t2_echo.md) | Dephasing time measurement | `delay_points`, `echo_type` |
| [**Ramsey**](ramsey.md) | Frequency characterization | `delay_points`, `detuning` |

### Quantum Information

| Experiment | Purpose | Key Parameters |
|-----------|---------|----------------|
| [**CHSH**](chsh.md) | Bell inequality test | `phase_points`, `theta_a`, `theta_b` |
| [**Parity Oscillation**](parity_oscillation.md) | Multi-qubit decoherence | `num_qubits_list`, `delays_us` |

## Common Experiment Workflow

All experiments follow a consistent pattern:

### 1. Import and Setup

```python
from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Rabi  # or T1, CHSH, etc.

backend = LocalBackend(device="qulacs")
```

### 2. Configure Experiment

```python
exp = Rabi(
    experiment_name="my_experiment",
    physical_qubit=0,
    amplitude_points=20,
    max_amplitude=2.0,
)
```

### 3. Execute

```python
result = exp.run(backend=backend, shots=1000)
```

### 4. Analyze

```python
df = result.analyze(
    plot=True,         # Interactive visualization
    save_data=True,    # Save to JSON/CSV
    save_image=True,   # Save plot images
)
```

## Advanced Features

### Parallel Execution

All experiments support parallel circuit submission for faster execution:

```python
# Automatic parallel execution for OQTOPUS backend
backend = OqtopusBackend()
result = exp.run(backend=backend, shots=1000)  # Circuits run in parallel
```

### Custom Analysis

You can access raw results for custom analysis:

```python
result = exp.run(backend=backend, shots=1000)

# Access raw data
raw_data = result.raw_results

# Custom analysis
import numpy as np
import matplotlib.pyplot as plt

# Your custom analysis code here
```

### Experiment Metadata

Each experiment tracks comprehensive metadata:

```python
# After running experiment
metadata = exp.get_metadata()
print(f"Experiment type: {metadata['experiment_type']}")
print(f"Parameters: {metadata['parameters']}")
print(f"Execution time: {metadata['execution_time']}")
```

## Experiment Design Principles

### Type Safety

All experiments use Pydantic models for type-safe parameter validation:

```python
# This will raise validation error for invalid parameters
exp = Rabi(amplitude_points=-5)  # Error: amplitude_points must be positive
```

### Consistent Results

All experiments return structured results with:

- **Raw measurement data**: Original counts/probabilities
- **Fitted parameters**: Curve fitting results (where applicable)
- **Quality metrics**: R², fit errors, confidence intervals
- **Metadata**: Experiment configuration and execution details

### Backend Agnostic

Write once, run anywhere:

```python
# Same experiment code works with any backend
for backend_name in ["qulacs", "noisy", "oqtopus"]:
    if backend_name == "oqtopus":
        backend = OqtopusBackend()
    else:
        backend = LocalBackend(device=backend_name)
    
    result = exp.run(backend=backend, shots=1000)
    print(f"{backend_name}: {result.summary}")
```

## Next Steps

- Choose an experiment to learn more about:
  - [Rabi Oscillations](rabi.md) - Start here for qubit calibration
  - [T1 Measurements](t1.md) - Characterize energy relaxation
  - [CHSH Tests](chsh.md) - Demonstrate quantum entanglement
- Explore [Examples](../examples/index.md) for complete workflows
- Check [API Reference](../reference/SUMMARY.md) for detailed documentation