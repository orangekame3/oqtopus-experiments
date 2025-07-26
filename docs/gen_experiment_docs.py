"""Generate experiment documentation from class docstrings and code."""

import ast
import re
from pathlib import Path
from typing import Any

import mkdocs_gen_files


def extract_class_info(source_path: Path) -> dict[str, Any] | None:
    """Extract experiment class information from source file."""
    try:
        with open(source_path, encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Find class that inherits from BaseExperiment
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseExperiment":
                        return extract_experiment_class_details(node, source)
                    elif (
                        isinstance(base, ast.Attribute)
                        and base.attr == "BaseExperiment"
                    ):
                        return extract_experiment_class_details(node, source)

        return None
    except Exception as e:
        print(f"Error parsing {source_path}: {e}")
        return None


def extract_experiment_class_details(
    class_node: ast.ClassDef, source: str
) -> dict[str, Any]:
    """Extract detailed information from experiment class node."""

    # Extract class docstring
    class_docstring = ast.get_docstring(class_node)

    # Extract __init__ method details
    init_method = None
    analyze_method = None
    circuits_method = None

    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "__init__":
                init_method = node
            elif node.name == "analyze":
                analyze_method = node
            elif node.name == "circuits":
                circuits_method = node

    # Extract parameters from __init__
    parameters = []
    if init_method:
        parameters = extract_init_parameters(init_method)

    # Extract example usage from docstrings or code
    examples = extract_examples_from_source(source)

    return {
        "class_name": class_node.name,
        "class_docstring": class_docstring or "",
        "parameters": parameters,
        "examples": examples,
        "has_analyze": analyze_method is not None,
        "has_circuits": circuits_method is not None,
    }


def extract_init_parameters(init_node: ast.FunctionDef) -> list[dict[str, Any]]:
    """Extract parameter information from __init__ method."""
    parameters = []

    # Skip 'self' parameter
    for arg in init_node.args.args[1:]:
        param_info = {
            "name": arg.arg,
            "type": None,
            "default": None,
            "description": "",
        }

        # Extract type annotation
        if arg.annotation:
            param_info["type"] = ast.unparse(arg.annotation)

        # Extract default values
        num_defaults = len(init_node.args.defaults)
        num_args = len(init_node.args.args) - 1  # Exclude self
        if num_defaults > 0:
            default_index = len(parameters) - (num_args - num_defaults)
            if default_index >= 0 and default_index < len(init_node.args.defaults):
                default_val = init_node.args.defaults[default_index]
                try:
                    param_info["default"] = ast.unparse(default_val)
                except:
                    param_info["default"] = str(default_val)

        parameters.append(param_info)

    return parameters


def extract_examples_from_source(source: str) -> list[str]:
    """Extract example code blocks from source file."""
    examples = []

    # Look for example patterns in comments or docstrings
    example_patterns = [
        r'"""[\s\S]*?```python([\s\S]*?)```[\s\S]*?"""',
        r"'''[\s\S]*?```python([\s\S]*?)```[\s\S]*?'''",
    ]

    for pattern in example_patterns:
        matches = re.findall(pattern, source, re.MULTILINE)
        examples.extend(matches)

    return [example.strip() for example in examples if example.strip()]


def generate_experiment_page(class_info: dict[str, Any], experiment_name: str) -> str:
    """Generate markdown documentation for an experiment class."""

    class_name = class_info["class_name"]
    docstring = class_info["class_docstring"]
    parameters = class_info["parameters"]
    examples = class_info["examples"]

    # Create a clean title from class name
    title = experiment_name.replace("_", " ").title()
    if title.lower() == "chsh":
        title = "CHSH Bell Test"
    elif title.lower() == "t1":
        title = "T1 Relaxation"
    elif title.lower() == "t2_echo":
        title = "T2 Echo"

    # Start building markdown content
    content = f"# {title}\n\n"

    # Add class description
    if docstring:
        # Clean up docstring
        clean_docstring = docstring.strip()
        if not clean_docstring.endswith("."):
            clean_docstring += "."
        content += f"{clean_docstring}\n\n"

    # Add overview section
    content += "## Overview\n\n"
    content += f"The `{class_name}` class implements {title.lower()} experiments "
    content += (
        "with automatic circuit generation, data analysis, and visualization.\n\n"
    )

    # Add quick start example
    content += "## Quick Start\n\n"
    content += "```python\n"
    content += "from oqtopus_experiments.backends import OqtopusBackend\n"
    content += f"from oqtopus_experiments.experiments import {class_name}\n\n"
    content += 'backend = OqtopusBackend(device="qulacs")\n\n'

    # Generate example instantiation
    content += f"exp = {class_name}(\n"
    if parameters:
        for param in parameters[:4]:  # Show first 4 parameters
            if param["name"] not in ["experiment_name"]:
                if param["default"] and param["default"] != "None":
                    content += f"    {param['name']}={param['default']},\n"
                else:
                    # Provide reasonable defaults
                    if "points" in param["name"]:
                        content += f"    {param['name']}=20,\n"
                    elif "delay" in param["name"] and "max" in param["name"]:
                        content += f"    {param['name']}=50000.0,\n"
                    elif "amplitude" in param["name"] and "max" in param["name"]:
                        content += f"    {param['name']}=2.0,\n"
                    elif "qubit" in param["name"]:
                        content += f"    {param['name']}=0,\n"
    content += ")\n\n"

    content += "result = exp.run(backend=backend, shots=1000)\n"
    content += "df = result.analyze(plot=True, save_data=True)\n"
    content += "print(df.head())\n"
    content += "```\n\n"

    # Add parameters section
    if parameters:
        content += "## Parameters\n\n"
        content += "| Parameter | Type | Default | Description |\n"
        content += "|-----------|------|---------|-------------|\n"

        for param in parameters:
            param_type = (
                param.get("type", "")
                .replace(" | None", "")
                .replace("str | None", "str")
            )
            param_type = param_type.replace("int | None", "int").replace(
                "float | None", "float"
            )
            default_val = param.get("default", "")
            if default_val == "None":
                default_val = "Optional"

            # Generate description based on parameter name
            description = generate_param_description(param["name"], experiment_name)

            content += f"| `{param['name']}` | {param_type} | {default_val} | {description} |\n"
        content += "\n"

    # Add circuit information
    content += "## Circuit Structure\n\n"
    content += generate_circuit_description(experiment_name)
    content += "\n"

    # Add analysis section
    content += "## Analysis and Results\n\n"
    content += generate_analysis_description(experiment_name)
    content += "\n"

    # Add examples section
    content += "## Examples\n\n"
    content += generate_usage_examples(class_name, experiment_name)
    content += "\n"

    # Add backend considerations
    content += "## Backend Considerations\n\n"
    content += "### OQTOPUS Platform\n"
    content += "```python\n"
    content += "# Real quantum hardware\n"
    content += "backend = OqtopusBackend()\n\n"
    content += "# Fast noiseless simulation\n"
    content += 'backend = OqtopusBackend(device="qulacs")\n'
    content += "```\n\n"

    content += "### Local Simulation\n"
    content += "```python\n"
    content += "from oqtopus_experiments.backends import LocalBackend\n\n"
    content += "# Realistic noisy simulation\n"
    content += 'backend = LocalBackend(device="noisy")\n'
    content += "result = exp.run(backend=backend, shots=1000)\n"
    content += "```\n\n"

    # Add API reference link
    content += "## API Reference\n\n"
    content += f"For complete API documentation, see [`{class_name}`](../reference/oqtopus_experiments/experiments/{experiment_name}.md).\n\n"

    return content


def generate_param_description(param_name: str, experiment_name: str) -> str:
    """Generate description for parameter based on name and experiment type."""
    descriptions = {
        "experiment_name": "Optional name for the experiment (used in data files)",
        "physical_qubit": "Target qubit index for hardware execution",
        "physical_qubit_0": "First qubit index for two-qubit experiments",
        "physical_qubit_1": "Second qubit index for two-qubit experiments",
        "amplitude_points": "Number of amplitude values to test",
        "max_amplitude": "Maximum drive amplitude in arbitrary units",
        "delay_points": "Number of delay time points to measure",
        "max_delay": "Maximum delay time in nanoseconds",
        "detuning_frequency": "Frequency detuning in Hz for Ramsey experiments",
        "shots_per_circuit": "Number of measurement shots per circuit",
        "measurement_angles": "Dictionary of measurement angles for CHSH test",
        "theta": "Phase parameter for Bell state preparation",
        "echo_type": "Type of echo sequence (hahn or cpmg)",
        "num_echoes": "Number of echo pulses in sequence",
    }

    return descriptions.get(param_name, f"Parameter for {experiment_name} experiment")


def generate_circuit_description(experiment_name: str) -> str:
    """Generate circuit description based on experiment type."""
    descriptions = {
        "rabi": """Each circuit applies an RX rotation with varying amplitude:

```python
qc = QuantumCircuit(1, 1)
qc.rx(amplitude * π, 0)  # Parameterized rotation
qc.measure(0, 0)
```

The amplitude sweep reveals the π-pulse calibration point.""",
        "t1": """Each circuit prepares |1⟩ state and waits for decay:

```python
qc = QuantumCircuit(1, 1)
qc.x(0)                    # Prepare |1⟩
qc.delay(delay_time, 0)    # Wait for relaxation
qc.measure(0, 0)           # Measure survival probability
```

The delay sweep reveals the T1 relaxation time.""",
        "ramsey": """Each circuit implements Ramsey interferometry:

```python
qc = QuantumCircuit(1, 1)
qc.ry(π/2, 0)              # First π/2 pulse
qc.delay(delay_time, 0)    # Free evolution
qc.rz(detuning_phase, 0)   # Optional detuning
qc.ry(π/2, 0)              # Second π/2 pulse
qc.measure(0, 0)
```

The interference fringes reveal T2* dephasing time.""",
        "chsh": """Four circuits measure Bell state correlations:

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

The correlations test Bell inequality violation.""",
        "t2_echo": """Echo sequences refocus dephasing errors:

```python
qc.ry(π/2, 0)              # Initial π/2 pulse
qc.delay(delay/2, 0)       # First half delay
qc.x(0)                    # π pulse (echo)
qc.delay(delay/2, 0)       # Second half delay
qc.ry(π/2, 0)              # Final π/2 pulse
qc.measure(0, 0)
```

Multiple echoes can extend coherence measurement.""",
        "parity_oscillation": """Multi-qubit GHZ state preparation and measurement:

```python
# GHZ state preparation
qc.h(0)
for i in range(1, n_qubits):
    qc.cx(0, i)

qc.delay(delay_time, range(n_qubits))  # Collective evolution
qc.rz(φ, range(n_qubits))              # Parity rotation
qc.measure_all()
```

Parity measurements reveal multi-qubit decoherence.""",
    }

    return descriptions.get(
        experiment_name, f"Circuit structure for {experiment_name} experiment."
    )


def generate_analysis_description(experiment_name: str) -> str:
    """Generate analysis description based on experiment type."""
    descriptions = {
        "rabi": """### Automatic Fitting

The experiment fits data to the Rabi oscillation model:
```
P(|1⟩) = A × sin²(π × amplitude × frequency) + offset
```

**Key Outputs:**
- `pi_amplitude`: Drive amplitude for π-pulse
- `frequency`: Rabi frequency 
- `r_squared`: Fit quality (0-1)""",
        "t1": """### Exponential Decay Fitting

The experiment fits data to exponential decay:
```
P(|1⟩) = A × exp(-t/T1) + offset
```

**Key Outputs:**
- `t1_time`: Relaxation time in nanoseconds
- `amplitude`: Initial state fidelity
- `r_squared`: Fit quality (0-1)""",
        "ramsey": """### Oscillatory Decay Fitting

The experiment fits data to damped oscillation:
```
P(|1⟩) = A × exp(-t/T2*) × cos(2πft + φ) + offset
```

**Key Outputs:**
- `t2_star_time`: Dephasing time in nanoseconds
- `frequency`: Oscillation frequency in Hz
- `phase`: Phase offset in radians""",
        "chsh": """### Bell Inequality Analysis

The experiment calculates CHSH correlation:
```
S = |⟨ZZ⟩ - ⟨ZX⟩ + ⟨XZ⟩ + ⟨XX⟩|
```

**Key Outputs:**
- `chsh_value`: CHSH parameter S
- `bell_violation`: True if S > 2 (quantum behavior)
- `significance`: Statistical significance of violation""",
        "t2_echo": """### Echo Decay Fitting

The experiment fits data to echo decay:
```
P(|1⟩) = A × exp(-t/T2) + offset
```

**Key Outputs:**
- `t2_time`: True coherence time in nanoseconds
- `amplitude`: Echo efficiency
- `r_squared`: Fit quality (0-1)""",
    }

    return descriptions.get(
        experiment_name, f"Analysis details for {experiment_name} experiment."
    )


def generate_usage_examples(class_name: str, experiment_name: str) -> str:
    """Generate practical usage examples."""
    examples = f"""### Basic Usage

```python
exp = {class_name}()
result = exp.run(backend=backend, shots=1000)
df = result.analyze(plot=True)
```

### Multiple Qubits

```python
results = {{}}
for qubit in [0, 1, 2]:
    exp = {class_name}(physical_qubit=qubit)
    result = exp.run(backend=backend, shots=1000)
    df = result.analyze(plot=True)
    results[qubit] = df
```

### High Precision

```python
exp = {class_name}(
    {generate_example_params(experiment_name)}
)
result = exp.run(backend=backend, shots=2000)
df = result.analyze(plot=True, save_data=True)
```"""

    return examples


def generate_example_params(experiment_name: str) -> str:
    """Generate example parameters for high precision experiments."""
    param_examples = {
        "rabi": "amplitude_points=50,\n    max_amplitude=4.0",
        "t1": "delay_points=30,\n    max_delay=100000.0",
        "ramsey": "delay_points=40,\n    max_delay=20000.0,\n    detuning_frequency=1e6",
        "chsh": "shots_per_circuit=2000",
        "t2_echo": "delay_points=25,\n    max_delay=100000.0",
    }

    return param_examples.get(experiment_name, "# High precision parameters")


def main():
    """Generate experiment documentation from source files."""

    experiments_dir = Path("src/oqtopus_experiments/experiments")

    # Generate files directly in experiments directory
    generated_pages = []

    # Process each experiment file
    experiment_files = list(experiments_dir.glob("*.py"))
    experiment_files = [f for f in experiment_files if f.name != "__init__.py"]

    generated_pages = []

    for exp_file in experiment_files:
        experiment_name = exp_file.stem
        print(f"Processing {experiment_name}...")

        # Extract class information
        class_info = extract_class_info(exp_file)
        if not class_info:
            print(f"  No experiment class found in {exp_file}")
            continue

        # Generate documentation
        doc_content = generate_experiment_page(class_info, experiment_name)

        # Write to file
        output_file = f"experiments/{experiment_name}.md"
        with mkdocs_gen_files.open(output_file, "w") as f:
            f.write(doc_content)

        generated_pages.append(experiment_name)
        print(f"  Generated {output_file}")

    print(f"\\nGenerated documentation for {len(generated_pages)} experiments:")
    for page in sorted(generated_pages):
        print(f"  - {page}")


if __name__ == "__main__":
    main()
