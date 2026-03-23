"""
Tests for TOML support added to apyrobo/config.py.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from apyrobo.config import ApyroboConfig, _simple_toml_dumps


# ---------------------------------------------------------------------------
# from_file — auto-detect by extension
# ---------------------------------------------------------------------------

class TestFromFileAutoDetect:
    def test_yaml_extension_loads_yaml(self, tmp_path):
        cfg_file = tmp_path / "apyrobo.yaml"
        cfg_file.write_text("robot:\n  uri: mock://yaml_bot\n")
        cfg = ApyroboConfig.from_file(cfg_file)
        assert cfg.robot_uri == "mock://yaml_bot"

    def test_yml_extension_loads_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yml"
        cfg_file.write_text("robot:\n  uri: mock://yml_bot\n")
        cfg = ApyroboConfig.from_file(cfg_file)
        assert cfg.robot_uri == "mock://yml_bot"

    def test_nonexistent_file_returns_default(self, tmp_path):
        cfg = ApyroboConfig.from_file(tmp_path / "missing.yaml")
        assert cfg.robot_uri == "mock://turtlebot4"

    def test_toml_extension_calls_toml_loader(self, tmp_path):
        cfg_file = tmp_path / "apyrobo.toml"
        # Write valid TOML
        cfg_file.write_text('[robot]\nuri = "mock://toml_bot"\n')
        try:
            cfg = ApyroboConfig.from_file(cfg_file)
            assert cfg.robot_uri == "mock://toml_bot"
        except ImportError:
            pytest.skip("No TOML library available")


# ---------------------------------------------------------------------------
# from_toml_file
# ---------------------------------------------------------------------------

class TestFromTomlFile:
    def test_toml_basic_load(self, tmp_path):
        cfg_file = tmp_path / "cfg.toml"
        cfg_file.write_text(textwrap.dedent("""\
            [robot]
            uri = "mock://toml_robot"
            [agent]
            provider = "rule"
            [safety]
            max_speed = 0.8
        """))
        try:
            cfg = ApyroboConfig.from_toml_file(cfg_file)
            assert cfg.robot_uri == "mock://toml_robot"
            assert cfg.agent_provider == "rule"
            assert cfg.safety_policy().max_speed == 0.8
        except ImportError:
            pytest.skip("No TOML library available")

    def test_toml_missing_file_returns_default(self, tmp_path):
        try:
            cfg = ApyroboConfig.from_toml_file(tmp_path / "missing.toml")
            assert cfg.robot_uri == "mock://turtlebot4"
        except ImportError:
            pytest.skip("No TOML library available")

    def test_toml_import_error_when_no_library(self, tmp_path):
        cfg_file = tmp_path / "cfg.toml"
        cfg_file.write_text('[robot]\nuri = "mock://x"\n')
        import apyrobo.config as config_module
        with patch.object(config_module, "_tomllib", None):
            with pytest.raises(ImportError, match="TOML support requires"):
                ApyroboConfig.from_toml_file(cfg_file)


# ---------------------------------------------------------------------------
# from_yaml_file
# ---------------------------------------------------------------------------

class TestFromYamlFile:
    def test_yaml_explicit_load(self, tmp_path):
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text("robot:\n  uri: mock://explicit_yaml\n")
        cfg = ApyroboConfig.from_yaml_file(cfg_file)
        assert cfg.robot_uri == "mock://explicit_yaml"

    def test_yaml_missing_returns_default(self, tmp_path):
        cfg = ApyroboConfig.from_yaml_file(tmp_path / "nope.yaml")
        assert cfg.robot_uri == "mock://turtlebot4"


# ---------------------------------------------------------------------------
# from_env — TOML candidate
# ---------------------------------------------------------------------------

class TestFromEnvToml:
    def test_from_env_discovers_toml_file(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "apyrobo.toml"
        cfg_file.write_text('[robot]\nuri = "mock://env_toml"\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("APYROBO_CONFIG", raising=False)
        try:
            cfg = ApyroboConfig.from_env()
            assert cfg.robot_uri == "mock://env_toml"
        except ImportError:
            pytest.skip("No TOML library available")

    def test_from_env_yaml_takes_priority_over_toml(self, tmp_path, monkeypatch):
        (tmp_path / "apyrobo.yaml").write_text("robot:\n  uri: mock://yaml_wins\n")
        (tmp_path / "apyrobo.toml").write_text('[robot]\nuri = "mock://toml_loses"\n')
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("APYROBO_CONFIG", raising=False)
        cfg = ApyroboConfig.from_env()
        assert cfg.robot_uri == "mock://yaml_wins"


# ---------------------------------------------------------------------------
# to_toml
# ---------------------------------------------------------------------------

class TestToToml:
    def test_to_toml_returns_string(self):
        cfg = ApyroboConfig()
        result = cfg.to_toml()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_to_toml_roundtrip(self, tmp_path):
        cfg_file = tmp_path / "cfg.toml"
        cfg_file.write_text('[robot]\nuri = "mock://roundtrip"\n')
        try:
            cfg = ApyroboConfig.from_toml_file(cfg_file)
            toml_str = cfg.to_toml()
            assert "roundtrip" in toml_str
        except ImportError:
            pytest.skip("No TOML library available")


# ---------------------------------------------------------------------------
# save — TOML path
# ---------------------------------------------------------------------------

class TestSaveToml:
    def test_save_toml_extension(self, tmp_path):
        cfg = ApyroboConfig({"robot": {"uri": "mock://save_toml"}})
        out = tmp_path / "saved.toml"
        cfg.save(out)
        assert out.exists()
        content = out.read_text()
        assert len(content) > 0  # non-empty

    def test_save_yaml_extension_still_works(self, tmp_path):
        cfg = ApyroboConfig({"robot": {"uri": "mock://save_yaml"}})
        out = tmp_path / "saved.yaml"
        cfg.save(out)
        assert out.exists()
        import yaml
        loaded = yaml.safe_load(out.read_text())
        assert loaded["robot"]["uri"] == "mock://save_yaml"


# ---------------------------------------------------------------------------
# _simple_toml_dumps helper
# ---------------------------------------------------------------------------

class TestSimpleTomlDumps:
    def test_string_value(self):
        result = _simple_toml_dumps({"key": "value"})
        assert 'key = "value"' in result

    def test_int_value(self):
        result = _simple_toml_dumps({"count": 42})
        assert "count = 42" in result

    def test_float_value(self):
        result = _simple_toml_dumps({"speed": 1.5})
        assert "speed = 1.5" in result

    def test_bool_true(self):
        result = _simple_toml_dumps({"enabled": True})
        assert "enabled = true" in result

    def test_bool_false(self):
        result = _simple_toml_dumps({"enabled": False})
        assert "enabled = false" in result

    def test_none_omitted(self):
        result = _simple_toml_dumps({"key": None})
        assert "key" not in result

    def test_list_of_strings(self):
        result = _simple_toml_dumps({"tags": ["a", "b"]})
        assert "tags" in result
        assert '"a"' in result

    def test_list_of_ints(self):
        result = _simple_toml_dumps({"nums": [1, 2, 3]})
        assert "nums = [1, 2, 3]" in result

    def test_nested_dict_creates_section(self):
        result = _simple_toml_dumps({"robot": {"uri": "mock://x"}})
        assert "[robot]" in result
        assert 'uri = "mock://x"' in result

    def test_escaped_string(self):
        result = _simple_toml_dumps({"msg": 'say "hello"'})
        assert '\\"hello\\"' in result
