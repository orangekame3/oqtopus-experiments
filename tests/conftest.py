#!/usr/bin/env python3
"""
Test configuration and shared fixtures
"""

import importlib
import sys
import types

import pytest


@pytest.fixture
def import_oqtopus_backend():
    """
    Factory fixture to import oqtopus_backend with controlled OQTOPUS availability
    """
    def _import_with_availability(monkeypatch, available: bool):
        """
        Import oqtopus_backend with the environment tweaked so that
        quri_parts_oqtopus.backend *is* or *is not* importable.
        """
        dep = "quri_parts_oqtopus.backend"

        if available:
            # Inject a fake module so the import succeeds
            fake = types.ModuleType(dep)

            class MockOqtopusSamplingBackend:
                def __init__(self):
                    self.device_name = "test_device"

                def sample_qasm(self, *args, **kwargs):
                    return MockJob()

                def submit(self, *args, **kwargs):
                    return "mock_job_id"

                def get(self, job_id):
                    return {"counts": {"0": 50, "1": 50}, "success": True}

                def retrieve_job(self, job_id):
                    return MockJob()

            class MockJob:
                def __init__(self):
                    self.job_id = "mock_job_123"
                    self.description = None
                    self._job = self

                def to_dict(self):
                    return {"status": "succeeded"}

                def result(self):
                    return MockResult()

            class MockResult:
                def __init__(self):
                    self.counts = {"0": 50, "1": 50}

            fake.OqtopusSamplingBackend = MockOqtopusSamplingBackend
            monkeypatch.setitem(sys.modules, dep, fake)
        else:
            # Ensure the dependency cannot be imported
            if dep in sys.modules:
                monkeypatch.delitem(sys.modules, dep)

            # Make any new import attempt fail
            def _raise_import_error(name, *args, **kwargs):
                if name == dep or name.startswith(dep + "."):
                    raise ImportError(f"simulated absence of {dep}")
                return orig_import(name, *args, **kwargs)

            orig_import = __builtins__['__import__']
            monkeypatch.setattr(__builtins__, "__import__", _raise_import_error)

        # Force reload of the module under test
        module_name = "oqtopus_experiments.backends.oqtopus_backend"
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])

        return importlib.import_module(module_name)

    return _import_with_availability


@pytest.fixture
def mock_device_info():
    """Mock DeviceInfo for testing"""
    class MockDeviceInfo:
        def __init__(self, device_name="test_device"):
            self.device_name = device_name
            self._device_info = None
            self.available = False

        def _load_device_info(self):
            pass

    return MockDeviceInfo
