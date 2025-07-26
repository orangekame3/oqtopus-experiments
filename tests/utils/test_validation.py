#!/usr/bin/env python3
"""
Tests for validation utilities
"""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from oqtopus_experiments.utils.validation import (
    validate_and_convert_oqtopus_result,
    validate_config_migration,
    validate_oqtopus_result_legacy,
)


class TestValidateOqtopusResultLegacy:
    """Test validate_oqtopus_result_legacy function"""

    def test_valid_completed_result(self):
        """Test validation of completed result"""
        result = {"status": "completed", "counts": {"0": 50, "1": 50}}
        assert validate_oqtopus_result_legacy(result) is True

    def test_valid_succeeded_result(self):
        """Test validation of succeeded result"""
        result = {"status": "succeeded", "data": "some_data"}
        assert validate_oqtopus_result_legacy(result) is True

    def test_valid_success_result(self):
        """Test validation of success result"""
        result = {"status": "success", "measurement": "result"}
        assert validate_oqtopus_result_legacy(result) is True

    def test_valid_case_insensitive(self):
        """Test validation is case insensitive"""
        result = {"status": "COMPLETED", "data": "test"}
        assert validate_oqtopus_result_legacy(result) is True

        result = {"status": "Success", "data": "test"}
        assert validate_oqtopus_result_legacy(result) is True

        result = {"status": "SUCCEEDED", "data": "test"}
        assert validate_oqtopus_result_legacy(result) is True

    def test_invalid_status(self):
        """Test validation with invalid status"""
        result = {"status": "failed", "error": "something went wrong"}
        assert validate_oqtopus_result_legacy(result) is False

        result = {"status": "pending", "data": "still processing"}
        assert validate_oqtopus_result_legacy(result) is False

        result = {"status": "unknown", "data": "test"}
        assert validate_oqtopus_result_legacy(result) is False

    def test_missing_status(self):
        """Test validation with missing status field"""
        result = {"data": "some_data", "counts": {"0": 50}}
        assert validate_oqtopus_result_legacy(result) is False

    def test_empty_result(self):
        """Test validation with empty result"""
        result = {}
        assert validate_oqtopus_result_legacy(result) is False

    def test_none_result(self):
        """Test validation with None result"""
        result = None
        assert validate_oqtopus_result_legacy(result) is False

    def test_non_dict_result(self):
        """Test validation with non-dictionary result"""
        result = "not a dictionary"
        assert validate_oqtopus_result_legacy(result) is False

        result = 12345
        assert validate_oqtopus_result_legacy(result) is False

        result = ["list", "instead", "of", "dict"]
        assert validate_oqtopus_result_legacy(result) is False

    def test_empty_status(self):
        """Test validation with empty status"""
        result = {"status": "", "data": "test"}
        assert validate_oqtopus_result_legacy(result) is False

    def test_none_status(self):
        """Test validation with None status"""
        result = {"status": None, "data": "test"}
        assert validate_oqtopus_result_legacy(result) is False


class TestValidateAndConvertOqtopusResult:
    """Test validate_and_convert_oqtopus_result function"""

    @patch('oqtopus_experiments.utils.validation.ExperimentResult')
    def test_successful_conversion(self, mock_experiment_result):
        """Test successful result conversion"""
        mock_result = MagicMock()
        mock_experiment_result.from_oqtopus_result.return_value = mock_result

        result = {"status": "completed", "counts": {"0": 50, "1": 50}}
        task_id = "test_task_123"

        converted = validate_and_convert_oqtopus_result(result, task_id)

        assert converted == mock_result
        mock_experiment_result.from_oqtopus_result.assert_called_once_with(result, task_id)

    @patch('oqtopus_experiments.utils.validation.ExperimentResult')
    def test_conversion_failure(self, mock_experiment_result):
        """Test handling of conversion failure"""
        mock_experiment_result.from_oqtopus_result.side_effect = Exception("Conversion failed")

        result = {"invalid": "data"}
        task_id = "test_task_123"

        with patch('builtins.print') as mock_print:
            converted = validate_and_convert_oqtopus_result(result, task_id)

        assert converted is None
        mock_print.assert_called_once()
        assert "Failed to convert OQTOPUS result" in mock_print.call_args[0][0]
        assert "Conversion failed" in mock_print.call_args[0][0]

    @patch('oqtopus_experiments.utils.validation.ExperimentResult')
    def test_conversion_with_different_exceptions(self, mock_experiment_result):
        """Test handling of different types of exceptions"""
        # Test with ValueError
        mock_experiment_result.from_oqtopus_result.side_effect = ValueError("Invalid value")

        result = {"data": "test"}
        task_id = "task_1"

        converted = validate_and_convert_oqtopus_result(result, task_id)
        assert converted is None

        # Test with KeyError
        mock_experiment_result.from_oqtopus_result.side_effect = KeyError("Missing key")

        converted = validate_and_convert_oqtopus_result(result, task_id)
        assert converted is None

        # Test with TypeError
        mock_experiment_result.from_oqtopus_result.side_effect = TypeError("Type error")

        converted = validate_and_convert_oqtopus_result(result, task_id)
        assert converted is None


class TestValidateConfigMigration:
    """Test validate_config_migration function"""

    def test_valid_config_file(self):
        """Test validation of valid config file"""
        valid_config = {
            "experiment_type": "rabi",
            "shots": 1024,
            "parameters": {"amplitude": 1.0}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_config, f)
            config_path = f.name

        try:
            with patch('oqtopus_experiments.utils.validation.DefaultConfig') as mock_config:
                mock_config.load_from_json.return_value = MagicMock()

                result = validate_config_migration(config_path)

                assert result is True
                mock_config.load_from_json.assert_called_once_with(config_path)
        finally:
            import os
            os.unlink(config_path)

    def test_invalid_config_file(self):
        """Test validation of invalid config file"""
        config_path = "nonexistent_config.json"

        with patch('oqtopus_experiments.utils.validation.DefaultConfig') as mock_config:
            mock_config.load_from_json.side_effect = FileNotFoundError("File not found")

            with patch('builtins.print') as mock_print:
                result = validate_config_migration(config_path)

            assert result is False
            mock_print.assert_called_once()
            assert "Configuration validation failed" in mock_print.call_args[0][0]
            assert "File not found" in mock_print.call_args[0][0]

    def test_config_validation_error(self):
        """Test handling of config validation errors"""
        config_path = "test_config.json"

        with patch('oqtopus_experiments.utils.validation.DefaultConfig') as mock_config:
            mock_config.load_from_json.side_effect = ValueError("Invalid configuration format")

            with patch('builtins.print') as mock_print:
                result = validate_config_migration(config_path)

            assert result is False
            mock_print.assert_called_once()
            assert "Configuration validation failed" in mock_print.call_args[0][0]
            assert "Invalid configuration format" in mock_print.call_args[0][0]

    def test_config_parsing_error(self):
        """Test handling of JSON parsing errors"""
        # Create a file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            config_path = f.name

        try:
            with patch('oqtopus_experiments.utils.validation.DefaultConfig') as mock_config:
                mock_config.load_from_json.side_effect = json.JSONDecodeError("Invalid JSON", "doc", 0)

                with patch('builtins.print') as mock_print:
                    result = validate_config_migration(config_path)

                assert result is False
                mock_print.assert_called_once()
                assert "Configuration validation failed" in mock_print.call_args[0][0]
        finally:
            import os
            os.unlink(config_path)

    def test_permission_error(self):
        """Test handling of permission errors"""
        config_path = "/root/inaccessible_config.json"

        with patch('oqtopus_experiments.utils.validation.DefaultConfig') as mock_config:
            mock_config.load_from_json.side_effect = PermissionError("Permission denied")

            with patch('builtins.print') as mock_print:
                result = validate_config_migration(config_path)

            assert result is False
            mock_print.assert_called_once()
            assert "Configuration validation failed" in mock_print.call_args[0][0]
            assert "Permission denied" in mock_print.call_args[0][0]


class TestIntegration:
    """Test integration between validation functions"""

    def test_legacy_validation_before_conversion(self):
        """Test that legacy validation can be used before conversion"""
        # Valid result
        valid_result = {"status": "completed", "counts": {"0": 50, "1": 50}}
        assert validate_oqtopus_result_legacy(valid_result) is True

        # Invalid result
        invalid_result = {"status": "failed", "error": "test error"}
        assert validate_oqtopus_result_legacy(invalid_result) is False

        # Only attempt conversion for valid results
        if validate_oqtopus_result_legacy(valid_result):
            with patch('oqtopus_experiments.utils.validation.ExperimentResult') as mock_result:
                mock_result.from_oqtopus_result.return_value = MagicMock()
                converted = validate_and_convert_oqtopus_result(valid_result, "task_1")
                assert converted is not None

    def test_validation_workflow(self):
        """Test complete validation workflow"""
        results = [
            {"status": "completed", "counts": {"0": 100}},
            {"status": "failed", "error": "timeout"},
            {"status": "succeeded", "data": "result"},
            {"invalid": "data"}
        ]

        valid_results = []
        converted_results = []

        for result in results:
            # Step 1: Legacy validation
            if validate_oqtopus_result_legacy(result):
                valid_results.append(result)

                # Step 2: Conversion
                with patch('oqtopus_experiments.utils.validation.ExperimentResult') as mock_result:
                    if result.get("status") in ["completed", "succeeded"]:
                        mock_result.from_oqtopus_result.return_value = MagicMock()
                    else:
                        mock_result.from_oqtopus_result.side_effect = Exception("Conversion failed")

                    converted = validate_and_convert_oqtopus_result(result, "task")
                    if converted is not None:
                        converted_results.append(converted)

        # Should have 2 valid results (completed and succeeded)
        assert len(valid_results) == 2
        # Should have 2 successfully converted results
        assert len(converted_results) == 2


if __name__ == "__main__":
    pytest.main([__file__])

