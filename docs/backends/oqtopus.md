# OQTOPUS Backend

The OQTOPUS backend enables execution on real quantum hardware and simulators through the OQTOPUS platform.

## Overview

`OqtopusBackend` provides access to:
- **Real quantum hardware**: IBM Quantum, IonQ, and other providers
- **High-performance simulators**: Qulacs for fast noiseless simulation
- **Queue management**: Automatic job submission and result collection
- **Parallel execution**: Efficient batch processing

## Basic Usage

### Real Hardware

```python
from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi

# Connect to real quantum hardware
backend = OqtopusBackend(device="anemone")  # Example device

exp = Rabi(physical_qubit=0, amplitude_points=20, max_amplitude=2.0)
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### Qulacs Simulation

```python
# Fast noiseless simulation
backend = OqtopusBackend(device="qulacs")

exp = Rabi(amplitude_points=30, max_amplitude=4.0)
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

## Configuration

### Authentication

OQTOPUS backend requires proper authentication:

1. **API Keys**: Configure your OQTOPUS API credentials
2. **Device Access**: Ensure access to target quantum devices
3. **Queue Limits**: Check your allocation and queue limits

### Device Selection

```python

# Specific device selection
backend = OqtopusBackend(device="anemone")  # Example device
backend = OqtopusBackend(device="qulacs")
```

## Advanced Features

### Parallel Execution

The OQTOPUS backend automatically parallelizes circuit submission:

```python
backend = OqtopusBackend("anemone")  # Real hardware

# All circuits submitted in parallel
exp = Rabi(amplitude_points=50, max_amplitude=3.0)
result = exp.run(backend=backend, shots=1000)
```

### Job Management

```python
# Monitor job progress
backend = OqtopusBackend("anemone")
result = exp.run(backend=backend, shots=1000)

# Access job metadata
job_info = result.job_metadata
print(f"Job ID: {job_info['job_id']}")
print(f"Status: {job_info['status']}")
```

### Error Handling

```python
try:
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze()
except Exception as e:
    print(f"Experiment failed: {e}")
    # Handle queue timeouts, device errors, etc.
```

## Best Practices

### For Real Hardware

1. **Start Small**: Begin with fewer data points
2. **Check Connectivity**: Verify qubit coupling maps
3. **Monitor Queues**: Check queue status before submission
4. **Validate Results**: Compare with simulation when possible

### For Simulation

1. **Use Qulacs**: Fastest option for algorithm development
2. **Scale Gradually**: Test with small systems first
3. **Batch Jobs**: Submit multiple experiments together

## Performance Considerations

| Feature             | Real Hardware   | Qulacs Simulation |
| ------------------- | --------------- | ----------------- |
| **Latency**         | Minutes-Hours   | Seconds           |
| **Parallelization** | Automatic       | Single-threaded   |
| **Noise**           | Device-specific | None              |
| **Scalability**     | Limited qubits  | Memory-limited    |

## Troubleshooting

### Common Issues

**Queue Timeouts**:
```python
# Reduce job size
exp = Rabi(amplitude_points=10, max_amplitude=2.0)  # Fewer points
```

## API Reference

For complete API documentation, see [`OqtopusBackend`](../reference/oqtopus_experiments/backends/oqtopus_backend.md).
