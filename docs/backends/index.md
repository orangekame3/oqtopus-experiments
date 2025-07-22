# Backends Overview

OQTOPUS Experiments supports multiple execution backends, allowing you to run the same experiment code on different platforms - from fast simulators to real quantum hardware.

## Available Backends

### OqtopusBackend
**Real quantum hardware via OQTOPUS platform**

```python
from oqtopus_experiments.backends import OqtopusBackend

backend = OqtopusBackend(device="anemone")
```

- **Use case**: Production experiments, real hardware characterization
- **Requirements**: OQTOPUS account and credentials
- **Features**: Automatic parallel job submission, queue management
- **Latency**: Minutes to hours depending on queue

### LocalBackend (Qulacs)
**Fast, noiseless quantum simulation**

```python
backend = OqtopusBackend(device="qulacs")
```

- **Use case**: Algorithm development, ideal case studies
- **Requirements**: Qulacs library
- **Features**: Near-instantaneous execution, perfect quantum operations
- **Limitations**: No noise modeling

### LocalBackend (Noisy)
**Realistic quantum simulation with noise**

```python
backend = LocalBackend(device="noisy")
```

- **Use case**: Testing noise resilience, realistic predictions
- **Requirements**: Qiskit Aer
- **Features**: Configurable noise models, realistic gate errors
- **Accuracy**: Good approximation of real hardware behavior

## Backend Selection Guide

### Choose OqtopusBackend when:
- Running final experiments for publication
- Characterizing real hardware properties
- Validating theoretical predictions
- Studying device-specific phenomena

### Choose LocalBackend(device="qulacs") when:
- Developing new experiments
- Testing algorithm correctness
- Studying ideal quantum behavior
- Need fast iteration cycles

### Choose LocalBackend(device="noisy") when:
- Optimizing for noise resilience
- Predicting real hardware performance
- Studying error correction protocols
- Validating noise models

## Backend-Agnostic Code

Write your experiments once and run them anywhere:

```python
def run_characterization(backend, qubit=0):
    """Run complete qubit characterization on any backend."""

    # Rabi calibration
    rabi = Rabi(physical_qubit=qubit, amplitude_points=20, max_amplitude=2.0)
    rabi_result = rabi.run(backend=backend, shots=1000)
    rabi_df = rabi_result.analyze(plot=True)

    # T1 measurement
    t1 = T1(physical_qubit=qubit, delay_points=15, max_delay=50000.0)
    t1_result = t1.run(backend=backend, shots=1000)
    t1_df = t1_result.analyze(plot=True)

    return {
        'pi_amplitude': rabi_df['pi_amplitude'].iloc[0],
        't1_time': t1_df['t1_time'].iloc[0] if 't1_time' in t1_df.columns else None,
    }

# Run on any backend
backends = {
    'ideal': LocalBackend(device="qulacs"),
    'noisy': LocalBackend(device="noisy"),
    'hardware': OqtopusBackend(),
}

results = {}
for name, backend in backends.items():
    results[name] = run_characterization(backend, qubit=0)
    print(f"{name}: π-amp={results[name]['pi_amplitude']:.3f}")
```

## Performance Considerations

### Execution Speed
- **Qulacs**: Microseconds per circuit
- **Noisy**: Seconds per circuit
- **OQTOPUS**: Minutes per circuit (including queue time)

### Parallel Execution
```python
# Automatic parallelization for OQTOPUS
backend = OqtopusBackend(device="anemone")
result = exp.run(backend=backend, shots=1000)  # Circuits submitted in parallel

# Sequential execution for local backends
backend = LocalBackend(device="noisy")
result = exp.run(backend=backend, shots=1000)  # Fast enough to be sequential
```

### Resource Usage
- **Qulacs**: Minimal memory, CPU-bound
- **Noisy**: Higher memory for noise simulation
- **OQTOPUS**: Network-bound, minimal local resources

## Next Steps

- Learn about specific backends:
  - [OQTOPUS Backend](oqtopus.md) - Real hardware setup
  - [Local Backend](local.md) - Simulation configuration
- See [Examples](../examples/index.md) for backend-specific usage patterns
- Check [API Reference](../reference/SUMMARY.md) for detailed configuration options
