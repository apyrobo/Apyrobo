"""Tests for compute profiles schema and registry."""
from __future__ import annotations

import json
import os
import sys
import subprocess

import pytest

from apyrobo.profiles import ComputeProfile, ProfileRegistry, get_profile


# ---------------------------------------------------------------------------
# ComputeProfile dataclass
# ---------------------------------------------------------------------------

class TestComputeProfile:
    def test_required_name(self):
        p = ComputeProfile(name="my-profile")
        assert p.name == "my-profile"

    def test_defaults(self):
        p = ComputeProfile(name="test")
        assert p.llm_provider == "anthropic"
        assert p.gpu_available is False
        assert p.edge_inference is False
        assert p.streaming is True
        assert p.extra == {}

    def test_to_dict_keys(self):
        p = ComputeProfile(name="x", description="desc")
        d = p.to_dict()
        assert "name" in d
        assert "llm_model" in d
        assert "gpu_available" in d
        assert "edge_inference" in d

    def test_to_dict_values(self):
        p = ComputeProfile(name="x", gpu_available=True, gpu_vram_gb=16.0)
        d = p.to_dict()
        assert d["gpu_available"] is True
        assert d["gpu_vram_gb"] == 16.0

    def test_extra_field(self):
        p = ComputeProfile(name="x", extra={"temperature": 0.7})
        assert p.extra["temperature"] == 0.7


# ---------------------------------------------------------------------------
# ProfileRegistry — built-ins
# ---------------------------------------------------------------------------

class TestProfileRegistryBuiltins:
    def test_five_builtin_profiles(self):
        reg = ProfileRegistry()
        assert len(reg.all()) == 5

    def test_builtin_names(self):
        reg = ProfileRegistry()
        names = reg.names()
        assert "jetson-orin" in names
        assert "raspberry-pi" in names
        assert "workstation-gpu" in names
        assert "cloud" in names
        assert "cpu-only" in names

    def test_jetson_orin_edge_inference(self):
        reg = ProfileRegistry()
        p = reg.get("jetson-orin")
        assert p is not None
        assert p.edge_inference is True
        assert p.gpu_available is True

    def test_raspberry_pi_small_context(self):
        reg = ProfileRegistry()
        p = reg.get("raspberry-pi")
        assert p is not None
        assert p.max_context_tokens <= 4_096

    def test_cloud_large_context(self):
        reg = ProfileRegistry()
        p = reg.get("cloud")
        assert p is not None
        assert p.max_context_tokens >= 100_000

    def test_workstation_gpu_has_vram(self):
        reg = ProfileRegistry()
        p = reg.get("workstation-gpu")
        assert p is not None
        assert p.gpu_vram_gb > 0

    def test_cpu_only_no_gpu(self):
        reg = ProfileRegistry()
        p = reg.get("cpu-only")
        assert p is not None
        assert p.gpu_available is False
        assert p.edge_inference is False


# ---------------------------------------------------------------------------
# ProfileRegistry — register / get
# ---------------------------------------------------------------------------

class TestProfileRegistryCustom:
    def test_register_custom(self):
        reg = ProfileRegistry()
        custom = ComputeProfile(name="my-board", llm_model="phi3:mini")
        reg.register(custom)
        assert reg.get("my-board") is not None

    def test_register_overwrites(self):
        reg = ProfileRegistry()
        reg.register(ComputeProfile(name="cpu-only", llm_model="custom-model"))
        p = reg.get("cpu-only")
        assert p.llm_model == "custom-model"

    def test_get_nonexistent_returns_none(self):
        reg = ProfileRegistry()
        assert reg.get("nonexistent-profile-xyz") is None

    def test_all_returns_list(self):
        reg = ProfileRegistry()
        profiles = reg.all()
        assert isinstance(profiles, list)
        assert all(isinstance(p, ComputeProfile) for p in profiles)


# ---------------------------------------------------------------------------
# get_profile convenience function
# ---------------------------------------------------------------------------

class TestGetProfile:
    def test_get_by_name(self):
        p = get_profile("jetson-orin")
        assert p.name == "jetson-orin"

    def test_get_cpu_only_default(self):
        p = get_profile(None)
        assert p.name == "cpu-only"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("APYROBO_PROFILE", "cloud")
        p = get_profile(None)
        assert p.name == "cloud"

    def test_env_var_overridden_by_explicit(self, monkeypatch):
        monkeypatch.setenv("APYROBO_PROFILE", "cloud")
        p = get_profile("raspberry-pi")
        assert p.name == "raspberry-pi"

    def test_unknown_name_falls_back_to_cpu_only(self):
        p = get_profile("completely-unknown-profile-xyz")
        assert p.name == "cpu-only"

    def test_empty_env_var_falls_back(self, monkeypatch):
        monkeypatch.setenv("APYROBO_PROFILE", "")
        p = get_profile(None)
        assert p.name == "cpu-only"


# ---------------------------------------------------------------------------
# CLI — apyrobo profiles list
# ---------------------------------------------------------------------------

class TestProfilesCLI:
    def test_profiles_list_runs(self):
        result = subprocess.run(
            [sys.executable, "-m", "apyrobo", "profiles"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_profiles_list_contains_all_names(self):
        result = subprocess.run(
            [sys.executable, "-m", "apyrobo", "profiles"],
            capture_output=True, text=True,
        )
        out = result.stdout
        assert "jetson-orin" in out
        assert "raspberry-pi" in out
        assert "cloud" in out

    def test_profiles_list_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "apyrobo", "profiles", "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 5

    def test_profiles_show_known(self):
        result = subprocess.run(
            [sys.executable, "-m", "apyrobo", "profiles", "show", "jetson-orin"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "jetson-orin" in result.stdout

    def test_profiles_show_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "apyrobo", "profiles", "show", "cloud", "--json"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["name"] == "cloud"

    def test_profiles_show_unknown_exits_1(self):
        result = subprocess.run(
            [sys.executable, "-m", "apyrobo", "profiles", "show", "nonexistent-xyz"],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
