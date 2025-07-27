#!/usr/bin/env python3
"""
Tests for BernsteinVazirani experiment
"""

from unittest.mock import patch

import pytest
from qiskit import QuantumCircuit

from oqtopus_experiments.experiments.bernstein_vazirani import BernsteinVazirani
from oqtopus_experiments.models.bernstein_vazirani_models import (
    BernsteinVaziraniParameters,
)


class TestBernsteinVazirani:
    """Test BernsteinVazirani experiment functionality"""

    def test_init_with_secret_string(self):
        """Test initialization with explicit secret string"""
        experiment = BernsteinVazirani(secret_string="1011")

        assert experiment.experiment_name == "bernstein_vazirani_experiment"
        assert experiment.secret_string == "1011"
        assert experiment.n_bits == 4
        assert experiment.secret_bits == [1, 1, 0, 1]  # Reversed for little-endian
        assert isinstance(experiment.params, BernsteinVaziraniParameters)

    def test_init_with_custom_name(self):
        """Test initialization with custom experiment name"""
        experiment = BernsteinVazirani(
            experiment_name="my_bv_experiment", secret_string="101"
        )

        assert experiment.experiment_name == "my_bv_experiment"
        assert experiment.secret_string == "101"
        assert experiment.n_bits == 3

    def test_init_with_random_secret(self):
        """Test initialization with random secret string"""
        experiment = BernsteinVazirani(n_bits=6)

        assert len(experiment.secret_string) == 6
        assert all(bit in "01" for bit in experiment.secret_string)
        assert experiment.n_bits == 6

    def test_init_invalid_secret_string(self):
        """Test initialization with invalid secret string raises error"""
        with pytest.raises(ValueError, match="must contain only 0s and 1s"):
            BernsteinVazirani(secret_string="1021")

        with pytest.raises(ValueError, match="must contain only 0s and 1s"):
            BernsteinVazirani(secret_string="abc")

    def test_circuits_creation(self):
        """Test that circuit is created correctly"""
        experiment = BernsteinVazirani(secret_string="1101")
        circuits = experiment.circuits()

        # BV algorithm creates only one circuit
        assert len(circuits) == 1
        circuit = circuits[0]

        assert isinstance(circuit, QuantumCircuit)
        # n_bits + 1 ancilla qubit
        assert circuit.num_qubits == 5
        # Only measure n_bits qubits
        assert circuit.num_clbits == 4

    def test_circuit_structure(self):
        """Test the structure of BV circuit"""
        experiment = BernsteinVazirani(secret_string="101")
        circuit = experiment.circuits()[0]

        # Verify basic structure
        assert circuit.num_qubits == 4  # 3 input + 1 ancilla
        assert circuit.num_clbits == 3  # Measure only input qubits

        # Get operations
        operations = [(inst.operation.name, inst.qubits) for inst in circuit.data]
        op_names = [op[0] for op in operations]

        # Should have Z on ancilla (after H), H gates, CNOTs for secret bits, H gates, measurements
        assert "z" in op_names  # Initialize ancilla to |-⟩ state
        assert op_names.count("h") >= experiment.n_bits * 2  # H on all qubits twice
        assert op_names.count("cx") == 2  # Two CNOTs for secret "101"
        assert op_names.count("measure") == experiment.n_bits

    def test_analyze_with_correct_result(self):
        """Test analyze method with simulated correct results"""
        experiment = BernsteinVazirani(secret_string="110")

        # Simulate perfect measurement results
        results = {
            "device1": [
                {
                    "counts": {
                        "011": 1000
                    },  # Result matches secret "110" -> "011" (reversed)
                    "backend": "test_backend",
                }
            ]
        }

        df = experiment.analyze(results, plot=False, save_data=False)

        assert not df.empty
        assert len(df) == 1
        assert df.iloc[0]["outcome"] == "011"
        assert df.iloc[0]["probability"] == 1.0
        assert df.iloc[0]["is_correct"]
        assert df.iloc[0]["success_probability"] == 1.0

    def test_analyze_with_noisy_results(self):
        """Test analyze method with noisy measurement results"""
        experiment = BernsteinVazirani(secret_string="10")

        # Simulate noisy measurement results
        results = {
            "device1": [
                {
                    "counts": {
                        "01": 800,  # Correct result for secret "10"
                        "00": 100,  # Wrong
                        "11": 50,  # Wrong
                        "10": 50,  # Wrong
                    },
                    "backend": "noisy_backend",
                }
            ]
        }

        df = experiment.analyze(results, plot=False, save_data=False)

        assert not df.empty
        assert len(df) == 4

        # Check the most probable outcome
        top_result = df.iloc[0]
        assert top_result["outcome"] == "01"
        assert top_result["probability"] == 0.8
        assert top_result["is_correct"]
        assert top_result["success_probability"] == 0.8

    def test_analyze_empty_results(self):
        """Test analyze with empty results"""
        experiment = BernsteinVazirani(secret_string="111")

        df = experiment.analyze({}, plot=False, save_data=False)
        assert df.empty

    def test_analyze_with_plotting(self):
        """Test analyze with plotting enabled"""
        experiment = BernsteinVazirani(secret_string="10")

        results = {
            "device1": [
                {
                    "counts": {"01": 1000},  # Correct result for secret "10"
                    "backend": "test_backend",
                }
            ]
        }

        with patch("oqtopus_experiments.utils.visualization.show_plotly_figure"):
            df = experiment.analyze(
                results, plot=True, save_data=False, save_image=False
            )
            assert not df.empty

    def test_process_results_error_handling(self):
        """Test error handling in _process_results"""
        experiment = BernsteinVazirani(secret_string="11")

        # Test with invalid results structure
        result = experiment._process_results([{"invalid": "data"}])
        assert result is None

    def test_get_circuit_params(self):
        """Test _get_circuit_params method"""
        experiment = BernsteinVazirani(secret_string="1010")
        experiment.circuits()  # Initialize experiment_params

        params = experiment._get_circuit_params()
        assert params is not None
        assert len(params) == 1
        assert params[0]["secret_string"] == "1010"
        assert params[0]["n_bits"] == 4

    def test_large_secret_string(self):
        """Test with larger secret strings"""
        secret = "110101011010"
        experiment = BernsteinVazirani(secret_string=secret)

        assert experiment.secret_string == secret
        assert experiment.n_bits == 12
        assert len(experiment.secret_bits) == 12

        circuits = experiment.circuits()
        assert circuits[0].num_qubits == 13  # 12 + 1 ancilla
        assert circuits[0].num_clbits == 12

    def test_all_zeros_secret(self):
        """Test with all-zeros secret string"""
        experiment = BernsteinVazirani(secret_string="0000")
        circuit = experiment.circuits()[0]

        # Should have no CNOTs for all-zeros secret
        operations = [(inst.operation.name, inst.qubits) for inst in circuit.data]
        op_names = [op[0] for op in operations]
        assert op_names.count("cx") == 0

    def test_all_ones_secret(self):
        """Test with all-ones secret string"""
        experiment = BernsteinVazirani(secret_string="1111")
        circuit = experiment.circuits()[0]

        # Should have 4 CNOTs for all-ones secret
        operations = [(inst.operation.name, inst.qubits) for inst in circuit.data]
        op_names = [op[0] for op in operations]
        assert op_names.count("cx") == 4
