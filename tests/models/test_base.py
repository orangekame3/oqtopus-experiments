#!/usr/bin/env python3
"""
Tests for base model classes
"""

import pytest
from pydantic import ValidationError

from oqtopus_experiments.models.base import (
    BaseConfigModel,
    BaseExperimentModel,
    BaseResultModel,
)


class MockConfigModel(BaseConfigModel):
    """Test model for BaseConfigModel"""

    name: str
    value: int
    optional_field: str = "default"


class MockExperimentModel(BaseExperimentModel):
    """Test model for BaseExperimentModel"""

    experiment_name: str
    shots: int = 1024


class MockResultModel(BaseResultModel):
    """Test model for BaseResultModel"""

    success: bool
    counts: dict


class TestBaseConfigModel:
    """Test BaseConfigModel functionality"""

    def test_init_valid_data(self):
        """Test initialization with valid data"""
        model = MockConfigModel(name="test", value=42)

        assert model.name == "test"
        assert model.value == 42
        assert model.optional_field == "default"

    def test_init_with_optional_field(self):
        """Test initialization with optional field override"""
        model = MockConfigModel(name="test", value=42, optional_field="custom")

        assert model.name == "test"
        assert model.value == 42
        assert model.optional_field == "custom"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden"""
        with pytest.raises(ValidationError) as exc_info:
            MockConfigModel(name="test", value=42, extra_field="not_allowed")

        assert "extra_field" in str(exc_info.value)
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_validate_assignment(self):
        """Test that assignment validation works"""
        model = MockConfigModel(name="test", value=42)

        # Valid assignment
        model.name = "new_name"
        assert model.name == "new_name"

        # Invalid assignment should raise ValidationError
        with pytest.raises(ValidationError):
            model.value = "not_an_int"

    def test_to_dict(self):
        """Test conversion to dictionary"""
        model = MockConfigModel(name="test", value=42, optional_field="custom")
        data = model.to_dict()

        expected = {"name": "test", "value": 42, "optional_field": "custom"}
        assert data == expected

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {"name": "test", "value": 42, "optional_field": "custom"}
        model = MockConfigModel.from_dict(data)

        assert model.name == "test"
        assert model.value == 42
        assert model.optional_field == "custom"

    def test_from_dict_with_missing_optional(self):
        """Test creation from dictionary with missing optional field"""
        data = {"name": "test", "value": 42}
        model = MockConfigModel.from_dict(data)

        assert model.name == "test"
        assert model.value == 42
        assert model.optional_field == "default"

    def test_from_dict_with_missing_required(self):
        """Test creation from dictionary with missing required field"""
        data = {"value": 42}
        with pytest.raises(ValidationError) as exc_info:
            MockConfigModel.from_dict(data)

        assert "name" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)

    def test_round_trip_conversion(self):
        """Test that to_dict/from_dict round trip works"""
        original = MockConfigModel(name="test", value=42, optional_field="custom")
        data = original.to_dict()
        restored = MockConfigModel.from_dict(data)

        assert original.name == restored.name
        assert original.value == restored.value
        assert original.optional_field == restored.optional_field


class TestBaseExperimentModel:
    """Test BaseExperimentModel functionality"""

    def test_init_valid_data(self):
        """Test initialization with valid data"""
        model = MockExperimentModel(experiment_name="test_exp")

        assert model.experiment_name == "test_exp"
        assert model.shots == 1024  # default value

    def test_init_with_all_fields(self):
        """Test initialization with all fields"""
        model = MockExperimentModel(experiment_name="test_exp", shots=2048)

        assert model.experiment_name == "test_exp"
        assert model.shots == 2048

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed"""
        model = MockExperimentModel(
            experiment_name="test_exp",
            shots=1024,
            extra_field="allowed",
            another_extra=42,
        )

        assert model.experiment_name == "test_exp"
        assert model.shots == 1024
        # Extra fields should be accessible
        assert hasattr(model, "extra_field")
        assert hasattr(model, "another_extra")
        assert model.extra_field == "allowed"
        assert model.another_extra == 42

    def test_validate_assignment(self):
        """Test that assignment validation works"""
        model = MockExperimentModel(experiment_name="test_exp")

        # Valid assignment
        model.experiment_name = "new_name"
        assert model.experiment_name == "new_name"

        # Invalid assignment should raise ValidationError
        with pytest.raises(ValidationError):
            model.shots = "not_an_int"

    def test_required_field_validation(self):
        """Test validation of required fields"""
        with pytest.raises(ValidationError) as exc_info:
            MockExperimentModel()  # Missing required experiment_name

        assert "experiment_name" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)


class TestBaseResultModel:
    """Test BaseResultModel functionality"""

    def test_init_valid_data(self):
        """Test initialization with valid data"""
        model = MockResultModel(success=True, counts={"0": 50, "1": 50})

        assert model.success is True
        assert model.counts == {"0": 50, "1": 50}

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed"""
        model = MockResultModel(
            success=True,
            counts={"0": 50, "1": 50},
            metadata={"shots": 100},
            backend_name="test_backend",
        )

        assert model.success is True
        assert model.counts == {"0": 50, "1": 50}
        assert hasattr(model, "metadata")
        assert hasattr(model, "backend_name")
        assert model.metadata == {"shots": 100}
        assert model.backend_name == "test_backend"

    def test_arbitrary_types_allowed(self):
        """Test that arbitrary types are allowed"""
        import numpy as np

        # Custom object that normally wouldn't be allowed in Pydantic
        custom_array = np.array([1, 2, 3])

        model = MockResultModel(
            success=True, counts={"0": 50, "1": 50}, numpy_data=custom_array
        )

        assert model.success is True
        assert hasattr(model, "numpy_data")
        assert isinstance(model.numpy_data, np.ndarray)
        np.testing.assert_array_equal(model.numpy_data, custom_array)

    def test_validate_assignment(self):
        """Test that assignment validation works"""
        model = MockResultModel(success=True, counts={"0": 50, "1": 50})

        # Valid assignment
        model.success = False
        assert model.success is False

        # Valid assignment of dict
        model.counts = {"0": 100}
        assert model.counts == {"0": 100}

    def test_required_field_validation(self):
        """Test validation of required fields"""
        with pytest.raises(ValidationError) as exc_info:
            MockResultModel(success=True)  # Missing required counts

        assert "counts" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            MockResultModel(counts={"0": 50})  # Missing required success

        assert "success" in str(exc_info.value)
        assert "Field required" in str(exc_info.value)


class TestModelConfigSettings:
    """Test model configuration settings"""

    def test_base_config_model_settings(self):
        """Test BaseConfigModel configuration"""
        model = MockConfigModel(name="test", value=42)
        config = model.model_config

        assert config["extra"] == "forbid"
        assert config["validate_assignment"] is True
        assert config["use_enum_values"] is True
        assert config["frozen"] is False

    def test_base_experiment_model_settings(self):
        """Test BaseExperimentModel configuration"""
        model = MockExperimentModel(experiment_name="test")
        config = model.model_config

        assert config["extra"] == "allow"
        assert config["validate_assignment"] is True
        assert config["use_enum_values"] is True

    def test_base_result_model_settings(self):
        """Test BaseResultModel configuration"""
        model = MockResultModel(success=True, counts={"0": 50})
        config = model.model_config

        assert config["extra"] == "allow"
        assert config["validate_assignment"] is True
        assert config["arbitrary_types_allowed"] is True


if __name__ == "__main__":
    pytest.main([__file__])
