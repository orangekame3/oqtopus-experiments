# New Experiment Creation Prompt

Use this prompt to request Claude to create a new quantum experiment:

---

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
- Follow existing code patterns from Rabi, T1, CHSH experiments
- Use proper type hints and docstrings
- Include meaningful analysis in the analyze() method
- Create appropriate visualizations
- Add comprehensive tests

---

## Example Usage

"Please create a new quantum experiment called `bell_state` with the following specifications:

**Experiment Details:**
- Name: bell_state
- Purpose: Create and measure Bell states to verify quantum entanglement
- Required qubits: 2
- Key parameters: 
  - state_type: str (options: "phi_plus", "phi_minus", "psi_plus", "psi_minus")
  - measure_basis: str (options: "computational", "bell")

**Expected behavior:**
The experiment should create the specified Bell state and measure correlations between the two qubits. It should calculate the fidelity of the prepared state and visualize the measurement results as a correlation matrix."