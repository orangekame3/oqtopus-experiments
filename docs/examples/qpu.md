# Real Hardware Examples

Examples for running experiments on real quantum hardware via the OQTOPUS platform.

## Overview

These examples demonstrate production-ready quantum experiments on real hardware, including:
- Device characterization protocols
- Parallel execution for efficiency
- Error handling and validation
- Result analysis and interpretation

## Prerequisites

### OQTOPUS Access

1. **Account Setup**: Create OQTOPUS account
2. **API Credentials**: Configure authentication keys
3. **Device Access**: Verify quantum device availability
4. **Queue Limits**: Check allocation and priority

### Installation

```bash
pip install git+https://github.com/orangekame3/oqtopus-experiments.git
```

## Basic Examples

### Single-Qubit Characterization

```bash
# Rabi oscillation calibration
python docs/examples/qpu/rabi.py

# T1 relaxation time measurement
python docs/examples/qpu/t1.py

# T2 echo coherence measurement
python docs/examples/qpu/t2_echo_parallel.py

# Ramsey interference
python docs/examples/qpu/ramsey.py
```

### Two-Qubit Experiments

```bash
# CHSH Bell inequality test
python docs/examples/qpu/chsh.py

# Parity oscillation (GHZ states)
python docs/examples/qpu/parity_oscillation.py
```

### Device Information

```bash
# Query device properties
python docs/examples/qpu/device.py
```

## Parallel Execution Examples

For faster data collection on real hardware:

```bash
# Parallel single-qubit experiments
python docs/examples/qpu/rabi_parallel.py
python docs/examples/qpu/t1_parallel.py
python docs/examples/qpu/ramsey_parallel.py

# Parallel two-qubit experiments
python docs/examples/qpu/chsh_parallel.py
python docs/examples/qpu/parity_oscillation_parallel.py
```

## Advanced Examples

### Physical Qubit Targeting

```bash
# Target specific physical qubits
python docs/examples/qpu/rabi_parallel_physical_qubit.py
python docs/examples/qpu/chsh_parallel_physical_qubit.py
python docs/examples/qpu/t1_parallel_physical.py
```

### Parameter Sweeps

```bash
# CHSH phase scan experiment
python docs/examples/qpu/chsh_phase_scan.py
python docs/examples/qpu/chsh_phase_scan_parallel.py
```

### Circuit Transpilation

```bash
# Custom transpilation examples
python docs/examples/qpu/transpile.py
```

## Example Code Structure

Each example follows this pattern:

```python
from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi

def main():
    # Configure backend for real hardware
    backend = OqtopusBackend()
    
    # Create experiment with hardware-appropriate parameters
    exp = Rabi(
        experiment_name="hardware_rabi",
        physical_qubit=0,
        amplitude_points=15,  # Conservative for hardware
        max_amplitude=2.0,
    )
    
    # Execute with error handling
    try:
        result = exp.run(backend=backend, shots=1000)
        df = result.analyze(plot=True, save_data=True)
        
        # Extract key results
        pi_amplitude = df['pi_amplitude'].iloc[0]
        r_squared = df['r_squared'].iloc[0]
        
        print(f"π-pulse amplitude: {pi_amplitude:.3f}")
        print(f"Fit quality (R²): {r_squared:.3f}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        # Handle queue timeouts, device errors, etc.

if __name__ == "__main__":
    main()
```

## Hardware Considerations

### Qubit Selection

```python
# Check device connectivity
backend = OqtopusBackend()
device_info = backend.get_device_info()

# Select well-characterized qubits
good_qubits = device_info.get_high_fidelity_qubits()
exp = Rabi(physical_qubit=good_qubits[0])
```

### Parameter Optimization

```python
# Hardware-appropriate parameters
exp = Rabi(
    amplitude_points=12,     # Fewer points for faster execution
    max_amplitude=1.5,       # Conservative amplitude range
)

# Two-qubit experiments
exp = CHSH(
    physical_qubit_0=0,
    physical_qubit_1=1,      # Check coupling map
    shots_per_circuit=2000,  # More shots for noise resilience
)
```

### Error Mitigation

```python
# Use parallel execution for statistics
results = []
for run in range(5):
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze()
    results.append(df)

# Aggregate results
import pandas as pd
combined_df = pd.concat(results)
mean_pi_amplitude = combined_df['pi_amplitude'].mean()
std_pi_amplitude = combined_df['pi_amplitude'].std()
```

## Performance Tips

### Queue Management

1. **Monitor Status**: Check queue before submission
2. **Batch Jobs**: Submit multiple experiments together
3. **Off-Peak Hours**: Schedule during low-usage periods
4. **Fallback Strategy**: Have simulation backup ready

### Data Collection

```python
# Efficient parameter sweeps
amplitudes = [0.5, 1.0, 1.5, 2.0]
results = {}

for amp in amplitudes:
    exp = Rabi(max_amplitude=amp, amplitude_points=10)
    result = exp.run(backend=backend, shots=1000)
    results[amp] = result.analyze()
```

### Resource Optimization

```python
# Conservative shot counts for initial testing
result = exp.run(backend=backend, shots=500)

# Increase shots for final measurements
if result.quality_check():
    final_result = exp.run(backend=backend, shots=2000)
```

## Troubleshooting

### Common Issues

**Authentication Errors**:
```bash
# Check API credentials
export OQTOPUS_API_KEY="your_key_here"
export OQTOPUS_API_SECRET="your_secret_here"
```

**Device Unavailable**:
```python
# Query available devices
backend = OqtopusBackend()
available_devices = backend.list_devices()
print("Available devices:", available_devices)
```

**Queue Timeouts**:
```python
# Reduce job size
exp = Rabi(amplitude_points=8, max_amplitude=1.5)  # Smaller experiment
```

**Poor Results**:
```python
# Validate with simulation first
sim_backend = OqtopusBackend(device="qulacs")
sim_result = exp.run(backend=sim_backend, shots=1000)

# Compare with hardware
hw_result = exp.run(backend=hardware_backend, shots=1000)
```

## Next Steps

1. **Start Simple**: Begin with single-qubit experiments
2. **Validate Results**: Compare with simulation when possible
3. **Scale Gradually**: Increase complexity once basics work
4. **Document Findings**: Save results for reproducibility

For more examples and tutorials, see:
- [Simulation Examples](simulation.md)
- [Backend Documentation](../backends/index.md)
- [API Reference](../reference/SUMMARY.md)