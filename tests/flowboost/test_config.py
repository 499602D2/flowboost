"""Tests for config save/load round-trip."""

from pathlib import Path

import pytest

from flowboost.config import config


class TestConfigSaveLoad:
    def test_round_trip(self, tmp_path):
        state = {
            "session": {
                "name": "test",
                "data_dir": str(tmp_path),
                "created_at": "2025-01-01T00:00:00+00:00",
            },
            "template": {
                "path": "/some/path",
                "additional_files": ["0", "processor0"],
            },
            "optimizer": {
                "type": "AxBackend",
                "offload_acquisition": False,
            },
            "scheduler": {
                "type": "Local",
                "job_limit": 4,
            },
        }
        config.save(state, tmp_path)
        loaded = config.load(tmp_path)

        assert loaded["session"]["name"] == "test"
        assert loaded["template"]["additional_files"] == ["0", "processor0"]
        assert loaded["optimizer"]["type"] == "AxBackend"
        assert loaded["scheduler"]["job_limit"] == 4

    def test_round_trip_preserves_nested_structure(self, tmp_path):
        state = {
            "session": {"name": "nested", "created_at": "2025-01-01T00:00:00"},
            "optimizer": {"type": "AxBackend", "offload_acquisition": False, "random_seed": 42},
            "template": {"path": "", "additional_files": []},
            "scheduler": {"type": "", "job_limit": 1},
        }
        config.save(state, tmp_path)
        loaded = config.load(tmp_path)
        assert loaded["optimizer"]["random_seed"] == 42

    def test_round_trip_preserves_numeric_types(self, tmp_path):
        state = {
            "session": {"max_evaluations": 50, "target_value": 0.01},
            "optimizer": {"type": "Ax", "offload_acquisition": False},
            "template": {"path": "", "additional_files": []},
            "scheduler": {"type": "", "job_limit": 1},
        }
        config.save(state, tmp_path)
        loaded = config.load(tmp_path)
        assert loaded["session"]["max_evaluations"] == 50
        assert loaded["session"]["target_value"] == 0.01


class TestConfigValidate:
    def test_valid_config(self):
        assert config.validate({"scheduler": {"offload_acquisition": False}}) is True

    def test_empty_config_valid(self):
        assert config.validate({}) is True

    def test_offload_without_acquisition_config_invalid(self):
        cfg = {"scheduler": {"offload_acquisition": True}}
        assert config.validate(cfg) is False
