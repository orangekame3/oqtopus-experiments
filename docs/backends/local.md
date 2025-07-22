# Local Backend

The Local backend provides realistic quantum simulation using Qiskit Aer with configurable noise models.

## Overview

`LocalBackend` enables:
- **Noisy simulation**: Realistic quantum errors and decoherence
- **Fast execution**: Local computation without queue delays
- **Customizable noise**: Adjustable error rates and models
- **Development testing**: Ideal for algorithm development

## Basic Usage

### Noisy Simulation

```python
from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import Rabi

# Realistic noisy simulation
backend = LocalBackend(device="noisy")

exp = Rabi(physical_qubit=0, amplitude_points=20, max_amplitude=2.0)
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### Custom Noise Configuration

```python
# Default noise parameters
backend = LocalBackend(
    device="noisy",
    noise_level=0.01,  # 1% error rate
    decoherence_time=50000.0  # 50 μs
)

exp = T1(delay_points=15, max_delay=100000.0)
result = exp.run(backend=backend, shots=1000)
```

## Noise Models

### Realistic Hardware Simulation

The local backend simulates common quantum errors:

- **Gate errors**: Imperfect quantum operations
- **Measurement errors**: Readout fidelity limitations
- **Decoherence**: T1 and T2 relaxation processes
- **Crosstalk**: Inter-qubit coupling effects

### Noise Parameters

```python
backend = LocalBackend(
    device="noisy",
    gate_error_rate=0.001,      # 0.1% gate errors
    measurement_error_rate=0.05, # 5% readout errors
    t1_time=50000.0,            # T1 = 50 μs
    t2_time=25000.0,            # T2 = 25 μs
)
```

## Configuration Options

### Device Types

```python
# Noisy simulation (default)
backend = LocalBackend(device="noisy")

# Custom noise model
backend = LocalBackend(
    device="custom",
    noise_model=custom_noise_model
)
```

### Advanced Settings

```python
backend = LocalBackend(
    device="noisy",
    max_qubits=20,              # System size limit
    memory_optimization=True,    # Reduce memory usage
    parallel_threads=4,         # CPU parallelization
)
```

## Use Cases

### Algorithm Development

```python
# Fast iteration for algorithm testing
backend = LocalBackend(device="noisy")

for amplitude in [1.0, 2.0, 3.0]:
    exp = Rabi(max_amplitude=amplitude, amplitude_points=20)
    result = exp.run(backend=backend, shots=1000)
    print(f"Pi-amplitude at {amplitude}: {result.pi_amplitude}")
```

### Noise Sensitivity Analysis

```python
noise_levels = [0.001, 0.01, 0.1]
results = {}

for noise in noise_levels:
    backend = LocalBackend(device="noisy", noise_level=noise)

    exp = CHSH()
    result = exp.run(backend=backend, shots=2000)
    df = result.analyze()

    results[noise] = df['chsh_value'].iloc[0]

# Compare CHSH values vs noise
for noise, chsh_val in results.items():
    print(f"Noise {noise}: CHSH = {chsh_val:.3f}")
```

### Hardware Prediction

```python
# Simulate expected hardware performance
backend = LocalBackend(
    device="noisy",
    t1_time=45000.0,    # Match target hardware T1
    t2_time=20000.0,    # Match target hardware T2
    gate_error_rate=0.002,  # Match gate fidelities
)

exp = Ramsey(delay_points=25, max_delay=50000.0)
result = exp.run(backend=backend, shots=1000)
predicted_t2_star = result.t2_star_time
```

## Performance

### Execution Speed

| Experiment Type  | Simulation Time | Memory Usage |
| ---------------- | --------------- | ------------ |
| **Single Qubit** | ~1 second       | ~10 MB       |
| **Two Qubits**   | ~5 seconds      | ~50 MB       |
| **Multi-Qubit**  | ~30 seconds     | ~500 MB      |

### Optimization Tips

```python
# Reduce shots for faster execution
backend = LocalBackend(device="noisy")
result = exp.run(backend=backend, shots=500)  # vs 1000

# Use memory optimization for large systems
backend = LocalBackend(
    device="noisy",
    memory_optimization=True,
    max_qubits=10  # Limit system size
)
```

## Comparison with Real Hardware

### Advantages

- **No queue delays**: Immediate execution
- **Reproducible results**: Consistent noise behavior
- **Cost effective**: No hardware usage charges
- **Debugging friendly**: Controlled error injection

### Limitations

- **Simplified noise**: Real hardware has more complex errors
- **No device-specific effects**: Generic noise models
- **Limited fidelity**: Cannot capture all hardware phenomena
- **Scalability**: Memory constraints for large systems

## Best Practices

### Development Workflow

1. **Start Local**: Develop and test with Local backend
2. **Validate Logic**: Ensure algorithm correctness
3. **Test Noise Sensitivity**: Evaluate robustness
4. **Move to Hardware**: Deploy on real quantum devices

### Debugging

```python
# Compare noisy vs noiseless
local_backend = LocalBackend(device="noisy")
qulacs_backend = OqtopusBackend(device="qulacs")

# Same experiment, different backends
for name, backend in [("noisy", local_backend), ("ideal", qulacs_backend)]:
    result = exp.run(backend=backend, shots=1000)
    print(f"{name}: {result.summary}")
```

## API Reference

For complete API documentation, see [`LocalBackend`](../reference/oqtopus_experiments/backends/local_backend.md).
