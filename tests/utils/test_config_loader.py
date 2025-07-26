#!/usr/bin/env python3
"""
Tests for config_loader utilities
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from oqtopus_experiments.models.config import DefaultConfig
from oqtopus_experiments.utils.config_loader import ConfigLoader


class TestConfigLoader:
    """Test ConfigLoader functionality"""

    def test_load_default_config_with_valid_file(self):
        """Test loading default config with valid file"""
        valid_config_data = {
            "workspace_info": {
                "description": "Test workspace",
                "version": "1.0.0",
                "library_version": "0.1.0",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config_data, f)
            config_path = f.name

        try:
            config = ConfigLoader.load_default_config(config_path)

            assert isinstance(config, DefaultConfig)
            assert config.workspace_info.description == "Test workspace"
            assert config.workspace_info.version == "1.0.0"
            assert config.workspace_info.library_version == "0.1.0"
        finally:
            Path(config_path).unlink()

    def test_load_default_config_file_not_found(self):
        """Test loading default config with non-existent file"""
        non_existent_path = "/nonexistent/path/config.json"

        with pytest.raises(FileNotFoundError) as exc_info:
            ConfigLoader.load_default_config(non_existent_path)

        assert "Configuration file not found" in str(exc_info.value)
        assert non_existent_path in str(exc_info.value)

    def test_load_default_config_with_none_path(self):
        """Test loading default config with None path (default behavior)"""
        # This should try to load from the default path which likely doesn't exist
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_default_config(None)

    def test_load_default_config_with_pathlib_path(self):
        """Test loading default config with pathlib.Path object"""
        valid_config_data = {
            "workspace_info": {
                "description": "Path test workspace",
                "version": "2.0.0",
                "library_version": "0.2.0",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config_data, f)
            config_path = Path(f.name)

        try:
            config = ConfigLoader.load_default_config(config_path)

            assert isinstance(config, DefaultConfig)
            assert config.workspace_info.description == "Path test workspace"
            assert config.workspace_info.version == "2.0.0"
        finally:
            config_path.unlink()

    def test_load_default_config_invalid_json(self):
        """Test loading default config with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            config_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                ConfigLoader.load_default_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_load_default_config_invalid_data(self):
        """Test loading default config with invalid data structure"""
        # Data that doesn't match DefaultConfig model
        invalid_config_data = {
            "invalid_field": "value",
            "shots": "not_a_number",  # Invalid type
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_config_data, f)
            config_path = f.name

        try:
            from pydantic import ValidationError

            with pytest.raises(
                ValidationError
            ):  # Should raise ValidationError from Pydantic
                ConfigLoader.load_default_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_load_legacy_config_as_dict(self):
        """Test loading legacy config as dictionary"""
        legacy_data = {
            "old_field": "value",
            "another_field": 123,
            "nested": {"key": "value"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(legacy_data, f)
            config_path = f.name

        try:
            result = ConfigLoader.load_legacy_config_as_dict(config_path)

            assert result == legacy_data
            assert isinstance(result, dict)
            assert result["old_field"] == "value"
            assert result["another_field"] == 123
            assert result["nested"]["key"] == "value"
        finally:
            Path(config_path).unlink()

    def test_load_legacy_config_as_dict_with_pathlib(self):
        """Test loading legacy config as dictionary with pathlib.Path"""
        legacy_data = {"test": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(legacy_data, f)
            config_path = Path(f.name)

        try:
            result = ConfigLoader.load_legacy_config_as_dict(config_path)
            assert result == legacy_data
        finally:
            config_path.unlink()

    def test_load_legacy_config_file_not_found(self):
        """Test loading legacy config with non-existent file"""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_legacy_config_as_dict("/nonexistent/config.json")

    def test_load_legacy_config_invalid_json(self):
        """Test loading legacy config with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            config_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                ConfigLoader.load_legacy_config_as_dict(config_path)
        finally:
            Path(config_path).unlink()

    def test_migrate_config_to_pydantic_without_output(self):
        """Test migrating config to Pydantic without saving to file"""
        legacy_data = {
            "workspace_info": {
                "description": "Migrated experiment",
                "version": "1.0.0",
                "library_version": "0.1.0",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(legacy_data, f)
            legacy_path = f.name

        try:
            config = ConfigLoader.migrate_config_to_pydantic(legacy_path)

            assert isinstance(config, DefaultConfig)
            assert config.workspace_info.description == "Migrated experiment"
            assert config.workspace_info.version == "1.0.0"
        finally:
            Path(legacy_path).unlink()

    def test_migrate_config_to_pydantic_with_output(self):
        """Test migrating config to Pydantic with saving to output file"""
        legacy_data = {
            "workspace_info": {
                "description": "Migrated with output",
                "version": "1.0.0",
                "library_version": "0.1.0",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(legacy_data, f)
            legacy_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_f:
            output_path = output_f.name

        try:
            with patch.object(DefaultConfig, "save_to_json") as mock_save:
                config = ConfigLoader.migrate_config_to_pydantic(
                    legacy_path, output_path
                )

                assert isinstance(config, DefaultConfig)
                assert config.workspace_info.description == "Migrated with output"
                assert config.workspace_info.version == "1.0.0"
                mock_save.assert_called_once_with(output_path)
        finally:
            Path(legacy_path).unlink()
            Path(output_path).unlink()

    def test_migrate_config_to_pydantic_with_pathlib_paths(self):
        """Test migrating config with pathlib.Path objects"""
        legacy_data = {
            "workspace_info": {
                "description": "Pathlib migration",
                "version": "1.0.0",
                "library_version": "0.1.0",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(legacy_data, f)
            legacy_path = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as output_f:
            output_path = Path(output_f.name)

        try:
            with patch.object(DefaultConfig, "save_to_json") as mock_save:
                config = ConfigLoader.migrate_config_to_pydantic(
                    legacy_path, output_path
                )

                assert isinstance(config, DefaultConfig)
                assert config.workspace_info.description == "Pathlib migration"
                mock_save.assert_called_once_with(str(output_path))
        finally:
            legacy_path.unlink()
            output_path.unlink()

    def test_migrate_config_invalid_legacy_data(self):
        """Test migrating config with invalid legacy data"""
        invalid_legacy_data = {"shots": "invalid_type"}  # Should be int

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(invalid_legacy_data, f)
            legacy_path = f.name

        try:
            from pydantic import ValidationError

            with pytest.raises(ValidationError):  # Should raise ValidationError
                ConfigLoader.migrate_config_to_pydantic(legacy_path)
        finally:
            Path(legacy_path).unlink()

    def test_migrate_config_legacy_file_not_found(self):
        """Test migrating config with non-existent legacy file"""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.migrate_config_to_pydantic("/nonexistent/legacy.json")

    @patch("builtins.open", new_callable=mock_open, read_data='{"test": "data"}')
    @patch("json.load")
    def test_load_methods_use_json_load(self, mock_json_load, mock_file):
        """Test that loading methods properly use json.load"""
        mock_json_load.return_value = {"experiment_name": "test", "shots": 1024}

        # Test load_legacy_config_as_dict
        ConfigLoader.load_legacy_config_as_dict("test_path")
        mock_file.assert_called_with("test_path")
        mock_json_load.assert_called()

        # Reset mocks
        mock_file.reset_mock()
        mock_json_load.reset_mock()

        # Test via migrate_config_to_pydantic
        with patch.object(DefaultConfig, "__init__", return_value=None):
            ConfigLoader.migrate_config_to_pydantic("test_path")
            mock_file.assert_called_with("test_path")
            mock_json_load.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
