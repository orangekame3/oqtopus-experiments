# OQTOPUS Experiments - Development Guide

## Project Overview
OQTOPUS Experiments: A modular quantum computing experiment library for OQTOPUS platform with multi-backend support (Qiskit Aer, Qulacs, OQTOPUS hardware), rich visualization, and parallel execution.

## Critical Notes
- **IMPORTANT**: Never expose OQTOPUS credentials or device tokens
- **IMPORTANT**: All code comments must be in English
- **IMPORTANT**: Use existing file edits over creating new files
- **IMPORTANT**: After implementing any code changes, ALWAYS run these commands in order:
  1. `task fmt` - Format code with Black and Ruff
  2. `task lint` - Check code quality with Ruff
  3. `task test` - Run pytest test suite
  4. `task check` - Run all checks (format, lint, type check)
- Python 3.12+ required
- Use `uv` package manager, NOT pip directly
- Project uses Pydantic v2 for data models
- When encountering technical challenges or needing web search, consult o3 via mcp__o3__o3-search

## Quick Start
Standard workflow: `uv sync` → `pip install -e .` → `task check` → run experiments

## Essential Commands

### Development Commands (via Taskfile)
- **Install**: `uv sync && pip install -e .`
- **Format**: `task fmt`
- **Lint**: `task lint`
- **Type check**: `task mypy` or `task mypy-strict`
- **All checks**: `task check`
- **Test**: `uv run pytest`
- **Documentation**: `task docs-serve`
- **Run examples**: `task run-local`

### Example Usage
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

## Code Style Guidelines
- **Black formatter** with line-length: 88
- **Ruff linter**: E, W, F, I, B, C4, UP rules
- **Type hints** required for all public functions
- **Docstrings** in Google style
- No mutable default arguments (B006 allowed for CLI)

## Project Structure

| Path | Purpose |
|------|---------|
| `src/oqtopus_experiments/backends/` | Backend implementations (Local, OQTOPUS) |
| `src/oqtopus_experiments/experiments/` | Experiment classes (Rabi, T1, CHSH, etc.) |
| `src/oqtopus_experiments/models/` | Pydantic models for type safety |
| `src/oqtopus_experiments/core/` | Base classes and data manager |
| `src/oqtopus_experiments/utils/` | Utilities (visualization, statistics) |
| `src/oqtopus_experiments/devices/` | Device information and configurations |
| `docs/examples/` | Example scripts for all experiments |
| `tests/` | Pytest test files |

## Architecture Patterns

### Experiment Implementation
1. Inherit from `BaseExperiment`
2. Implement `circuits()` method to create quantum circuits
3. Implement `analyze()` method in result models
4. Results are returned as `ExperimentResult` objects

### Backend Pattern
- `LocalBackend`: Qiskit Aer simulator (noisy/noiseless)
- `OqtopusBackend`: Real hardware or Qulacs simulation
- All backends implement `run()` method

## Data Models
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

## Available Experiments
- **Rabi**: Rabi oscillation for qubit calibration
- **T1**: Relaxation time measurement
- **T2Echo**: Coherence time with echo sequences
- **CHSH**: Bell inequality test
- **CHSHPhaseScan**: CHSH with phase scanning
- **Ramsey**: Ramsey interference
- **ParityOscillation**: GHZ state decoherence
- **RandomizedBenchmarking**: Gate error characterization (Standard/Interleaved RB)
- **HadamardTest**: Quantum algorithm for eigenvalue estimation

## Add New Experiment - Detailed Workflow

When adding a new experiment (e.g., "bell_state"), follow this exact sequence:

### 1. Create Experiment Class
```python
# src/oqtopus_experiments/experiments/bell_state.py
from ..core.base_experiment import BaseExperiment
from ..models.bell_state_models import BellStateData

class BellState(BaseExperiment):
    def __init__(self, experiment_name: str | None = None):
        super().__init__(experiment_name)
        
    def circuits(self) -> list[QuantumCircuit]:
        # Implementation here
        pass
```

### 2. Create Pydantic Models
```python
# src/oqtopus_experiments/models/bell_state_models.py
from pydantic import BaseModel
from ..models.experiment_result import ExperimentResult

class BellStateData(BaseModel):
    # Define data structure
    pass

class BellStateResult(ExperimentResult):
    data: BellStateData
    
    def analyze(self, plot=True, save_data=False, save_image=False):
        # Analysis implementation
        pass
```

### 3. Update Exports
```python
# src/oqtopus_experiments/experiments/__init__.py
from .bell_state import BellState

# src/oqtopus_experiments/models/__init__.py  
from .bell_state_models import BellStateData, BellStateResult
```

### 4. Create Examples
```python
# docs/examples/local/bell_state.py - Local simulator example
# docs/examples/qulacs/bell_state.py - Qulacs example
# docs/examples/qpu/bell_state.py - Real hardware example
```

### 5. Create Tests
```python
# tests/experiments/test_bell_state.py
# tests/models/test_bell_state_models.py
```

### 6. Update Documentation
```markdown
# docs/experiments/bell_state.md
```

### 7. Run Validation
```bash
task check  # Format, lint, and type check
uv run pytest tests/experiments/test_bell_state.py
```

## Automated Scaffolding

Instead of using scripts, use the prompt template:
1. Copy the template from the "New Experiment Prompt Template" section below
2. Fill in the template with your experiment details
3. Ask Claude: "Please create a new experiment using this specification: [paste filled template]"

This approach ensures:
- Always uses latest project patterns
- Adapts to current codebase structure
- No maintenance of scaffold scripts needed

## Plot Settings and Visualization Guidelines

### Standard Plot Configuration

#### Background and Layout
- `plot_bgcolor="white"` - White plot background
- `paper_bgcolor="white"` - White figure background
- `showgrid=True` - Display grid lines
- `gridwidth=1` - Standard grid line width
- `gridcolor="LightGray"` - Light gray grid color for readability

#### Color Scheme
Use consistent colors from `get_experiment_colors()`:
- Primary data: `colors[1]` (typically green)
- Fit/theoretical: `colors[0]` (typically blue)
- Error bars: Match data point colors
- Annotation background: `rgba(255,255,255,0.95)` with `bordercolor="#CCCCCC"`

#### Standard Dimensions
- Width: 1000px for analysis plots, 700px for simple plots
- Height: 500px for analysis plots, 400px for simple plots

#### Theoretical Fitting Formulas

**Rabi Oscillations**
For RX(amplitude * π) gates:
- Formula: `P(|1⟩) = A * sin²(π * amp / 2) + offset`
- π-pulse occurs at amplitude = 1.0
- Use scipy.optimize.curve_fit with bounds: A ∈ [0, 1], offset ∈ [0, 0.2]
- **IMPORTANT**: Not `sin²(f * amp)` - must use the correct formula for RX(amp * π)

**T1 Relaxation**
For excited state decay:
- Formula: `P(|1⟩) = A * exp(-t/T1) + offset`
- Use logarithmic time spacing: `np.logspace(log10(1), log10(max_delay), points)`
- Plot with logarithmic x-axis for better visualization

**T2 Echo Decay**
For coherence time measurement:
- Formula: `P(|0⟩) = A * exp(-t/T2) + offset`
- Use logarithmic time spacing for data collection
- Plot with logarithmic x-axis

**Ramsey Fringes**
For T2* measurement with detuning:
- Formula: `P(|0⟩) = A * exp(-t/T2*) * cos(2πft + φ) + offset`
- Use LINEAR time spacing to capture oscillations
- Plot with LINEAR x-axis to show fringes clearly

**Phase Kickback**
For H-RY(θ)-CP(φ)-H circuit:
- Formula: `P(|0⟩) = 1 - sin²(θ/2) sin²(φ/2)`
- Use scipy.optimize.curve_fit for robust parameter estimation
- Display both applied and measured phase with accuracy metrics

**Hadamard Test**
For H-controlled_U-H circuit:
- Formula: `P(|0⟩) = (1 + Re⟨ψ|U|ψ⟩)/2`
- Complex plane visualization when measuring both real and imaginary parts
- Unit circle reference for complex expectation values

**Randomized Benchmarking**
For Clifford gate sequences:
- Formula: `P(m) = A × r^m + B`
- Error per Clifford: `(1 - r) / 2`
- Statistics box with fitting parameters (error rate, R-squared, etc.)

## New Experiment Prompt Template

Use this prompt to request Claude to create a new quantum experiment:

```
Please create a new quantum experiment called `{EXPERIMENT_NAME}` with the following specifications:

**Experiment Details:**
- Name: {EXPERIMENT_NAME} (e.g., bell_state, ghz_state, quantum_teleportation)
- Purpose: {DESCRIBE_PURPOSE}
- Required qubits: {NUMBER_OF_QUBITS}
- Key parameters: {LIST_PARAMETERS}

**Expected behavior:**
{DESCRIBE_EXPECTED_BEHAVIOR}

Please create all necessary files following the project patterns:
1. Experiment class inheriting from BaseExperiment
2. Pydantic models for data and results
3. Test files with proper test cases
4. Example scripts for local, qulacs, and QPU backends
5. Documentation page
6. Update the __init__.py exports

Make sure to:
- Follow existing code patterns from Rabi, T1, CHSH, RandomizedBenchmarking experiments
- Use proper type hints and docstrings
- Include meaningful analysis in the analyze() method
- Create appropriate visualizations following plot_settings guidelines
- Add comprehensive tests

**IMPORTANT: Follow these standardized patterns:**

1. **Analyze Method Signature**:
   ```python
   def analyze(
       self,
       plot: bool = True,
       save_data: bool = True, 
       save_image: bool = True,
   ) -> pd.DataFrame:
   ```

2. **Use Standard Visualization Utilities**:
   ```python
   from ..utils.visualization import (
       apply_experiment_layout,
       get_experiment_colors,
       save_plotly_figure,
       show_plotly_figure,
   )
   ```

3. **Standard Analysis Flow**:
   ```python
   # In analyze() method:
   if plot:
       self._create_plot(analysis_result, save_image)
   if save_data:
       self._save_results(analysis_result)
   return df
   ```
```

### Example Usage

```
Please create a new quantum experiment called `bell_state` with the following specifications:

**Experiment Details:**
- Name: bell_state
- Purpose: Create and measure Bell states to verify quantum entanglement
- Required qubits: 2
- Key parameters: 
  - state_type: str (options: "phi_plus", "phi_minus", "psi_plus", "psi_minus")
  - measure_basis: str (options: "computational", "bell")

**Expected behavior:**
The experiment should create the specified Bell state and measure correlations between the two qubits. It should calculate the fidelity of the prepared state and visualize the measurement results as a correlation matrix.
```

## Quality Assurance

### Run Quality Checks
```bash
task check  # Runs format, lint, and type checks
```

### Debug Issues
- Check device availability: `OqtopusBackend.available_devices()`
- Verify transpilation for hardware constraints
- Use `plot=True` in analyze() for visualization

## Technical Support

When facing technical challenges:
- Use `mcp__o3__o3-search` for web search and technical consultation
- Example: Error troubleshooting, latest quantum computing techniques, library updates
- o3 can help with algorithm optimization, debugging strategies, and best practices

## Completed Implementations

### Recently Added
- **Comprehensive Error Handling System (2024)**: Complete error handling overhaul
  - Custom exception hierarchy with recovery suggestions
  - AnalysisResult class for structured error reporting  
  - Physics-aware validation for all experiment types
  - Graceful error recovery maintaining backward compatibility
  - Modern Python Union syntax (X | Y) throughout codebase
  - All 447 tests passing with no regressions

- **Critical Physics Bug Fixes (2024)**: Corrected fundamental formulas
  - **Rabi oscillations**: Fixed formula from sin²(f*amp) to sin²(π*amp/2) for RX(amp*π) gates
  - **Validation logic**: Updated to allow zero amplitudes in Rabi experiments
  - **Import corrections**: Fixed non-existent function imports across multiple models
  - **sklearn removal**: Replaced sklearn dependency with manual R² calculations

- **Randomized Benchmarking**: Complete implementation with both standard and interleaved variants
  - Exponential decay fitting for gate error characterization
  - SPAM-insensitive measurements
  - Comprehensive visualization with plot_settings guidelines
  - Full test coverage and documentation

## Standard Implementation Patterns

### Analyze Method Standardization
**Current Standard** (as of Randomized Benchmarking implementation):
```python
def analyze(
    self,
    plot: bool = True,
    save_data: bool = True,
    save_image: bool = True,
) -> pd.DataFrame:
```

### Visualization Pattern
**Use standardized utils.visualization helpers**:
```python
from ..utils.visualization import (
    apply_experiment_layout,
    get_experiment_colors,
    get_plotly_config,
    save_plotly_figure,
    setup_plotly_environment,
    show_plotly_figure,
)

# Standard flow:
setup_plotly_environment()
colors = get_experiment_colors()
fig = go.Figure()
# ... add traces ...
apply_experiment_layout(fig, title, xaxis_title, yaxis_title, width, height)
# ... add annotations ...
if save_image:
    save_plotly_figure(fig, name="experiment_name", images_dir="./plots")
config = get_plotly_config("experiment_name", width, height)
show_plotly_figure(fig, config)
```

### File Organization Pattern
**Standard file locations**:
- **Plot images**: `./plots/YYYYMMDD_experiment_name_N.png`
- **CSV data**: `experiment_results.csv` (in current directory)
- **JSON metadata**: `experiment_results_YYYYMMDD_HHMMSS.json`
- **Session data**: `.results/experiment_name_YYYYMMDD_HHMMSS/data/`

## Comprehensive Error Handling System

### Current Implementation (2024)

We have implemented a comprehensive error handling system across all experiment types with the following patterns:

#### Custom Exception Hierarchy
```python
# src/oqtopus_experiments/exceptions.py
class OQTOPUSExperimentError(Exception):
    """Base exception with recovery suggestions"""
    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or []

class FittingError(OQTOPUSExperimentError):
    """Raised when curve fitting fails"""

class InsufficientDataError(OQTOPUSExperimentError):
    """Raised when insufficient data for analysis"""

class InvalidParameterError(OQTOPUSExperimentError):
    """Raised when parameters are invalid"""
```

#### Structured Error Reporting
```python
# src/oqtopus_experiments/models/analysis_result.py
@dataclass
class AnalysisResult:
    """Structured result with comprehensive error handling"""
    success: bool
    data: pd.DataFrame | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    suggestions: list[str] | None = None
    metadata: dict[str, Any] | None = None
    
    def to_legacy_dataframe(self) -> pd.DataFrame:
        """Convert to legacy DataFrame format for backward compatibility"""
```

#### Error Handling in Model Classes
All experiment model classes (RabiAnalysisResult, T1AnalysisResult, etc.) now implement:
- **Comprehensive input validation** with detailed error messages
- **Graceful error recovery** - return meaningful results even when fitting fails
- **Quality assessment** - warn about poor fit quality or unusual parameters
- **Backward compatibility** - maintain legacy DataFrame return format

#### Validation Helpers
```python
# src/oqtopus_experiments/utils/validation_helpers.py
def validate_non_negative_values(values: list[float], parameter_name: str) -> None:
    """Validate values are non-negative (>= 0) - for Rabi amplitudes"""

def validate_probability_values(values: list[float], allow_zero: bool = False) -> None:
    """Validate probability values are in [0, 1] range"""

def validate_fitting_data(x_data: list[float], y_data: list[float], experiment_name: str) -> None:
    """Validate data is suitable for curve fitting"""
```

#### Key Implementation Details
- **Modern Python Union syntax**: Use `X | Y` instead of `Union[X, Y]`
- **No sklearn dependency**: All R² calculations done manually with numpy
- **Physics-aware validation**: Different validation rules for different experiments
- **Comprehensive suggestions**: Every error includes actionable recovery suggestions
- **All 447 tests passing**: Maintained full backward compatibility

### Error Handling Best Practices

**✅ DO: Comprehensive validation with helpful messages**
```python
def analyze(self, plot=True, save_data=True, save_image=True) -> pd.DataFrame:
    try:
        # Validate inputs with detailed error reporting
        result = self._validate_inputs(amplitudes, probabilities, errors)
        if not result.success:
            return result.to_legacy_dataframe()
            
        # Attempt analysis with graceful error handling
        fitting_result = self._fit_data(...)
        if fitting_result.error_info:
            result.add_warning(f"Fitting issues: {fitting_result.error_info}")
            result.add_suggestion("Check data quality and calibration")
            
    except InsufficientDataError as e:
        return AnalysisResult.error_result(
            errors=[str(e)], 
            suggestions=e.suggestions
        ).to_legacy_dataframe()
```

**❌ DON'T: Silent failures or generic error messages**
```python
def analyze(self):
    try:
        fit_data()
    except:
        return pd.DataFrame()  # Silent failure - no error info!
```

## Known Implementation Issues & Future Improvements

**⚠️ Based on comprehensive analysis with o3, the following issues have been identified for future improvements:**

### High Priority Issues

1. **❌ Inconsistent Analyze Method Signatures**
   - Base class returns `dict[str, Any]` but implementations return `pd.DataFrame`
   - Default parameter values vary between experiments
   - **Fix**: Standardize to return consistent types with unified signatures

2. **❌ File Scattering Problem**
   ```
   ./plots/YYYYMMDD_experiment_name_N.png          # Plot images
   ./experiment_results.csv                         # Some data files
   ./.results/exp_TIMESTAMP/data/                   # Session data
   ./experiment_results_YYYYMMDD_HHMMSS.json      # Metadata
   ```
   - **Fix**: Implement unified session-based file organization

3. **❌ Confusing Data Saving Architecture**
   - Multiple layers of saving logic: Model classes vs Experiment classes
   - Model classes (e.g., `RandomizedBenchmarkingResult`) directly save CSV files to current directory
   - Experiment classes use `BaseExperiment.save_experiment_data()` for structured saving
   - No clear ownership of saving responsibility
   - **Example Issue**: `RandomizedBenchmarkingResult._save_results()` creates `randomized_benchmarking_results.csv` in project root
   - **Root Cause**: Models shouldn't handle I/O - only experiments should save via data manager
   - **Architecture Confusion**: 
     ```python
     # Model class doing I/O (❌ Wrong)
     class ExperimentResult:
         def _save_results(self, df):
             df.to_csv("results.csv")  # Direct file creation
     
     # Experiment class using data manager (✅ Correct)
     class MyExperiment(BaseExperiment):
         def analyze(self):
             return self.save_experiment_data(data, metadata)
     ```
   - **Fix**: 
     - Remove all direct file I/O from model classes
     - Experiment classes should be sole responsibility for data saving
     - Models should return structured data, experiments decide when/where to save

4. **❌ Boolean Parameter Overload**
   - Too many boolean flags: `plot=True, save_data=True, save_image=True`
   - No control over output destinations or formats
   - **Fix**: Implement configuration object pattern

### Medium Priority Issues

5. **❌ Tight Coupling in Visualization**
   - Visualization logic embedded in each experiment class
   - Hard to add new plot types or modify existing ones
   - **Fix**: Create decoupled visualization system

6. **✅ RESOLVED: Comprehensive Error Handling Implemented (2024)**
   - Added custom exception hierarchy with recovery suggestions
   - Implemented structured AnalysisResult class for detailed error reporting
   - All experiment models now include comprehensive input validation
   - Graceful error recovery with meaningful fallback results
   - Modern Python Union syntax (X | Y) throughout codebase

## Architecture Guidelines for New Implementations

### Data Saving Best Practices

**❌ DON'T: Model classes handling I/O**
```python
class ExperimentResult:
    def analyze(self, save_data=True):
        # Process data...
        if save_data:
            self.df.to_csv("results.csv")  # Direct file creation - WRONG!
```

**✅ DO: Experiment classes managing all I/O**
```python
class MyExperiment(BaseExperiment):
    def analyze(self, results, save_data=True):
        # Get analysis from model (pure data processing)
        result_model = MyExperimentResult(...)
        df = result_model.analyze(save_data=False)  # No I/O in model
        
        # Experiment handles all saving via data manager
        if save_data:
            self.save_experiment_data(
                df.to_dict(orient="records"),
                metadata={"experiment_type": "my_experiment"},
                experiment_type="my_experiment"
            )
        return df
```

### Separation of Concerns
- **Models**: Pure data processing, fitting, calculations
- **Experiments**: I/O operations, user interface, orchestration
- **Utils**: Shared functionality, no state

### Proposed Future Architecture

```python
# Proposed improved patterns for future implementations:

@dataclass
class AnalysisConfig:
    plot: bool = True
    save_data: bool = True
    save_plots: bool = True
    output_dir: Optional[str] = None
    plot_formats: List[str] = ["png"]
    data_formats: List[str] = ["json"]

def analyze(self, results: dict, config: AnalysisConfig = None) -> AnalysisResult:
    """Standardized analysis with configuration object"""
    # Unified file organization:
    # .results/experiment_YYYYMMDD_HHMMSS/
    # ├── data/
    # │   ├── raw_results.json
    # │   ├── analysis_results.json
    # │   └── metadata.json
    # ├── plots/
    # │   ├── main_plot.png
    # │   ├── main_plot.html
    # │   └── supplementary/
    # └── summary.json
```

**Note**: Current implementations work correctly but follow these patterns. The issues above are for consideration in future major refactoring or new experiment development.

## Next Steps
- [ ] Add more advanced calibration experiments
- [ ] Implement error mitigation strategies  
- [ ] Add batch execution optimization
- [ ] Create experiment comparison tools
- [ ] **Major Refactoring** (Future): Implement unified configuration and file management system
- [ ] **Architecture Improvement** (Future): Decouple visualization and implement analysis pipeline pattern