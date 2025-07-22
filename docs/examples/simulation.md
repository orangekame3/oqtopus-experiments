# Simulation Examples

Examples for running experiments with quantum simulators for development and testing.

## Overview

These examples demonstrate quantum experiments using simulators, including:
- Fast noiseless simulation with Qulacs (via OQTOPUS)
- Realistic noisy simulation with Qiskit Aer
- Algorithm development and validation
- Performance comparison studies

## Prerequisites

### Installation

```bash
pip install git+https://github.com/orangekame3/oqtopus-experiments.git
```

### Simulator Access

- **Qulacs**: Available via OQTOPUS platform
- **Local Simulator**: Built-in Qiskit Aer support

## Noiseless Simulation (Qulacs)

Fast, ideal quantum simulation via OQTOPUS:

### Basic Examples

```bash
# Single-qubit experiments
python docs/examples/qulacs/rabi.py       # Ideal Rabi oscillation
python docs/examples/qulacs/t1.py         # Ideal T1 measurement
python docs/examples/qulacs/t2_echo.py    # Ideal T2 echo
python docs/examples/qulacs/ramsey.py     # Ideal Ramsey interference

# Multi-qubit experiments  
python docs/examples/qulacs/chsh.py       # Perfect Bell test
python docs/examples/qulacs/parity_oscillation.py  # Ideal GHZ states
```

### Parallel Examples

```bash
# Parallel execution for parameter sweeps
python docs/examples/qulacs/rabi_parallel.py
python docs/examples/qulacs/chsh_parallel.py
python docs/examples/qulacs/parity_oscillation_parallel.py

# Phase scan experiments
python docs/examples/qulacs/chsh_phase_scan.py
python docs/examples/qulacs/chsh_phase_scan_parallel.py
```

### Example Code Structure

```python
from oqtopus_experiments.backends import OqtopusBackend
from oqtopus_experiments.experiments import Rabi

def main():
    # Qulacs simulation via OQTOPUS
    backend = OqtopusBackend(device="qulacs")
    
    # High-resolution experiment (fast with ideal simulation)
    exp = Rabi(
        experiment_name="ideal_rabi",
        amplitude_points=50,     # High resolution
        max_amplitude=4.0,       # Wide range
    )
    
    # Fast execution
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True, save_data=True)
    
    # Perfect theoretical results
    pi_amplitude = df['pi_amplitude'].iloc[0]
    print(f"Theoretical π-pulse: {pi_amplitude:.6f}")

if __name__ == "__main__":
    main()
```

## Noisy Simulation (Local)

Realistic quantum simulation with errors:

### Basic Examples

```bash
# Single-qubit with noise
python docs/examples/local/rabi.py        # Noisy Rabi oscillation
python docs/examples/local/t1.py          # Noisy T1 measurement
python docs/examples/local/t2_echo.py     # Noisy T2 echo
python docs/examples/local/ramsey.py      # Noisy Ramsey interference

# Multi-qubit with noise
python docs/examples/local/chsh.py        # Noisy Bell test
python docs/examples/local/parity_oscillation.py  # Noisy GHZ states
```

### Parameter Studies

```bash
# Phase scan with noise effects
python docs/examples/local/chsh_phase_scan.py

# Parallel execution
python docs/examples/local/rabi_parallel.py
```

### Example Code Structure

```python
from oqtopus_experiments.backends import LocalBackend
from oqtopus_experiments.experiments import CHSH

def main():
    # Realistic noisy simulation
    backend = LocalBackend(device="noisy")
    
    # CHSH with noise
    exp = CHSH(
        experiment_name="noisy_chsh",
        shots_per_circuit=2000,  # More shots for noise resilience
    )
    
    # Execute with realistic errors
    result = exp.run(backend=backend, shots=2000)
    df = result.analyze(plot=True, save_data=True)
    
    # Analyze noise effects
    chsh_value = df['chsh_value'].iloc[0]
    bell_violation = df['bell_violation'].iloc[0]
    
    print(f"CHSH with noise: {chsh_value:.3f}")
    print(f"Bell violation: {bell_violation}")

if __name__ == "__main__":
    main()
```

## Development Workflow

### Algorithm Testing

```python
def test_algorithm_development():
    # 1. Start with ideal simulation
    ideal_backend = OqtopusBackend(device="qulacs")
    
    # 2. Validate algorithm correctness
    exp = Rabi(amplitude_points=20, max_amplitude=2.0)
    ideal_result = exp.run(backend=ideal_backend, shots=1000)
    ideal_df = ideal_result.analyze()
    
    # 3. Test with realistic noise
    noisy_backend = LocalBackend(device="noisy")
    noisy_result = exp.run(backend=noisy_backend, shots=1000)
    noisy_df = noisy_result.analyze()
    
    # 4. Compare results
    print("Ideal π-amplitude:", ideal_df['pi_amplitude'].iloc[0])
    print("Noisy π-amplitude:", noisy_df['pi_amplitude'].iloc[0])
    
    # 5. Ready for hardware deployment
    return ideal_df, noisy_df
```

### Performance Comparison

```python
def compare_backends():
    experiments = [
        ("Ideal", OqtopusBackend(device="qulacs")),
        ("Noisy", LocalBackend(device="noisy")),
    ]
    
    results = {}
    
    for name, backend in experiments:
        print(f"Testing {name} backend...")
        
        # Same experiment, different backend
        exp = Rabi(amplitude_points=15, max_amplitude=2.0)
        result = exp.run(backend=backend, shots=1000)
        df = result.analyze(plot=True)
        
        results[name] = {
            'pi_amplitude': df['pi_amplitude'].iloc[0],
            'r_squared': df['r_squared'].iloc[0],
            'frequency': df['frequency'].iloc[0],
        }
    
    # Compare results
    for name, data in results.items():
        print(f"{name}: π-amp={data['pi_amplitude']:.3f}, R²={data['r_squared']:.3f}")
    
    return results
```

## Use Cases

### 1. Algorithm Development

**Goal**: Validate quantum algorithm correctness

```python
# Fast iteration with ideal simulation
backend = OqtopusBackend(device="qulacs")

# Test multiple parameter sets quickly
for max_amp in [1.0, 2.0, 3.0, 4.0]:
    exp = Rabi(max_amplitude=max_amp, amplitude_points=20)
    result = exp.run(backend=backend, shots=1000)
    print(f"Max amplitude {max_amp}: π-pulse = {result.pi_amplitude:.3f}")
```

### 2. Noise Sensitivity Analysis

**Goal**: Understand algorithm robustness

```python
# Test different noise levels
noise_levels = [0.001, 0.01, 0.1]
results = {}

for noise in noise_levels:
    backend = LocalBackend(device="noisy", noise_level=noise)
    
    exp = CHSH()
    result = exp.run(backend=backend, shots=2000)
    df = result.analyze()
    
    results[noise] = df['chsh_value'].iloc[0]

# Plot noise sensitivity
import matplotlib.pyplot as plt
plt.plot(list(results.keys()), list(results.values()))
plt.xlabel('Noise Level')
plt.ylabel('CHSH Value')
plt.title('CHSH vs Noise Level')
```

### 3. Hardware Prediction

**Goal**: Estimate real hardware performance

```python
# Simulate target hardware characteristics
backend = LocalBackend(
    device="noisy",
    t1_time=50000.0,        # Target T1 = 50 μs
    t2_time=25000.0,        # Target T2 = 25 μs
    gate_error_rate=0.001,  # Target gate fidelity
)

exp = Ramsey(delay_points=20, max_delay=40000.0)
result = exp.run(backend=backend, shots=1000)
predicted_t2_star = result.t2_star_time

print(f"Predicted T2* on hardware: {predicted_t2_star:.1f} ns")
```

### 4. Educational Examples

**Goal**: Learn quantum computing concepts

```python
def educational_rabi():
    """Demonstrate Rabi oscillations with clear explanations."""
    
    backend = OqtopusBackend(device="qulacs")
    
    # Simple Rabi experiment
    exp = Rabi(
        amplitude_points=25,
        max_amplitude=4.0,  # Two full oscillations
    )
    
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True)
    
    # Educational output
    pi_amp = df['pi_amplitude'].iloc[0]
    freq = df['frequency'].iloc[0]
    
    print("=== Rabi Oscillation Results ===")
    print(f"π-pulse amplitude: {pi_amp:.3f}")
    print(f"Rabi frequency: {freq:.3f}")
    print(f"2π-pulse amplitude: {2*pi_amp:.3f}")
    print(f"π/2-pulse amplitude: {pi_amp/2:.3f}")
    
    return df
```

## Performance Comparison

| Backend | Speed | Noise | Use Case |
|---------|-------|-------|----------|
| **Qulacs** | Very Fast | None | Algorithm development |
| **Local Noisy** | Fast | Realistic | Noise analysis |
| **Real Hardware** | Slow | Device-specific | Production |

### Execution Times

```python
import time

backends = [
    ("Qulacs", OqtopusBackend(device="qulacs")),
    ("Local", LocalBackend(device="noisy")),
]

for name, backend in backends:
    start_time = time.time()
    
    exp = Rabi(amplitude_points=20, max_amplitude=2.0)
    result = exp.run(backend=backend, shots=1000)
    
    elapsed = time.time() - start_time
    print(f"{name} backend: {elapsed:.2f} seconds")
```

## Best Practices

### 1. Development Cycle

1. **Prototype**: Start with Qulacs for fast iteration
2. **Validate**: Test with noisy simulation
3. **Optimize**: Adjust for noise resilience
4. **Deploy**: Move to real hardware

### 2. Parameter Selection

```python
# Simulation-appropriate parameters
simulation_params = {
    'amplitude_points': 50,     # High resolution
    'delay_points': 30,         # Fine time steps
    'shots': 1000,              # Standard statistics
}

# Hardware-appropriate parameters  
hardware_params = {
    'amplitude_points': 15,     # Faster execution
    'delay_points': 12,         # Reduce queue time
    'shots': 2000,              # Better noise averaging
}
```

### 3. Result Validation

```python
def validate_results(ideal_result, noisy_result, tolerance=0.1):
    """Compare ideal and noisy results for validation."""
    
    ideal_pi = ideal_result['pi_amplitude'].iloc[0]
    noisy_pi = noisy_result['pi_amplitude'].iloc[0]
    
    difference = abs(ideal_pi - noisy_pi)
    relative_error = difference / ideal_pi
    
    if relative_error < tolerance:
        print(f"✓ Results consistent: {relative_error:.1%} difference")
        return True
    else:
        print(f"✗ Large difference: {relative_error:.1%}")
        return False
```

## Next Steps

1. **Explore Examples**: Run different simulation examples
2. **Modify Parameters**: Experiment with different settings
3. **Compare Results**: Validate between simulators
4. **Move to Hardware**: Deploy on real quantum devices

For more information, see:
- [Real Hardware Examples](qpu.md)
- [Backend Documentation](../backends/index.md)
- [Experiment Guides](../experiments/index.md)