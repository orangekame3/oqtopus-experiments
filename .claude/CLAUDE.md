# <system_context>
OQTOPUS Experiments: A modular quantum computing experiment library for OQTOPUS platform with multi-backend support (Qiskit Aer, Qulacs, OQTOPUS hardware), rich visualization, and parallel execution.

# <critical_notes>
- IMPORTANT: Never expose OQTOPUS credentials or device tokens
- IMPORTANT: All code comments must be in English
- IMPORTANT: Use existing file edits over creating new files
- Python 3.12+ required
- Use `uv` package manager, NOT pip directly
- Project uses Pydantic v2 for data models
- When encountering technical challenges or needing web search, consult o3 via mcp__o3__o3-search

# <paved_path>
Standard workflow: `uv sync` → `pip install -e .` → `task check` → run experiments

# Commands
## Essential Commands (via Taskfile)
- install: `uv sync && pip install -e .`
- format: `task fmt`
- lint: `task lint`
- typecheck: `task mypy` or `task mypy-strict`
- all checks: `task check`
- test: `uv run pytest`
- docs: `task docs-serve`
- run examples: `task run-local`

## Example Usage
```python
from oqtopus_experiments.backends import LocalBackend, OqtopusBackend
from oqtopus_experiments.experiments import Rabi

# Choose backend
backend = OqtopusBackend(device="anemone")  # Real hardware
# backend = OqtopusBackend(device="qulacs")  # Noiseless simulation
# backend = LocalBackend(device="noisy")     # Noisy simulation

# Run experiment
exp = Rabi(physical_qubit=0, amplitude_points=20, max_amplitude=2.0)
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True, save_data=True)
```

# Code Style
- Black formatter with line-length: 88
- Ruff linter: E, W, F, I, B, C4, UP rules
- Type hints required for all public functions
- Docstrings in Google style
- No mutable default arguments (B006 allowed for CLI)

# File Map
| Path | Purpose |
|------|---------|
| src/oqtopus_experiments/backends/ | Backend implementations (Local, OQTOPUS) |
| src/oqtopus_experiments/experiments/ | Experiment classes (Rabi, T1, CHSH, etc.) |
| src/oqtopus_experiments/models/ | Pydantic models for type safety |
| src/oqtopus_experiments/core/ | Base classes and data manager |
| src/oqtopus_experiments/utils/ | Utilities (visualization, statistics) |
| src/oqtopus_experiments/devices/ | Device information and configurations |
| docs/examples/ | Example scripts for all experiments |
| tests/ | Pytest test files |

# Architecture Patterns
## Experiment Implementation
1. Inherit from `BaseExperiment`
2. Implement `circuits()` method to create quantum circuits
3. Implement `analyze()` method in result models
4. Results are returned as `ExperimentResult` objects

## Backend Pattern
- `LocalBackend`: Qiskit Aer simulator (noisy/noiseless)
- `OqtopusBackend`: Real hardware or Qulacs simulation
- All backends implement `run()` method

# Data Models
```python
# Core result structure (Pydantic models)
class ExperimentResult:
    data: ExperimentData  # Type varies by experiment
    backend: str
    device: str
    shots: int
    metadata: dict[str, Any]
    
    def analyze(self, plot=True, save_data=False, save_image=False) -> pd.DataFrame
```

# Available Experiments
- **Rabi**: Rabi oscillation for qubit calibration
- **T1**: Relaxation time measurement
- **T2Echo**: Coherence time with echo sequences
- **CHSH**: Bell inequality test
- **CHSHPhaseScan**: CHSH with phase scanning
- **Ramsey**: Ramsey interference
- **ParityOscillation**: GHZ state decoherence

# Common Tasks
## Add New Experiment
1. Create class in `src/oqtopus_experiments/experiments/{name}.py`
2. Implement `circuits()` method
3. Create models in `src/oqtopus_experiments/models/{name}_models.py`
4. Add to `__init__.py` exports
5. Create example in `docs/examples/`
6. Add tests in `tests/experiments/`

## Run Quality Checks
```bash
task check  # Runs format, lint, and type checks
```

## Debug Issues
- Check device availability: `OqtopusBackend.available_devices()`
- Verify transpilation for hardware constraints
- Use `plot=True` in analyze() for visualization

## Technical Support
When facing technical challenges:
- Use `mcp__o3__o3-search` for web search and technical consultation
- Example: Error troubleshooting, latest quantum computing techniques, library updates
- o3 can help with algorithm optimization, debugging strategies, and best practices

# Next Steps
- [ ] Add more advanced calibration experiments
- [ ] Implement error mitigation strategies
- [ ] Add batch execution optimization
- [ ] Create experiment comparison tools