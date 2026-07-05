"""Tests for `apyrobo init --adapter` (v8 Phase 2 adapter authoring kit).

The bar for the scaffold: a generated package must pass strict protocol
conformance (zero warnings) before the author writes a single line.
"""
from __future__ import annotations

import argparse
import importlib
import sys

import pytest

from apyrobo.cli import cmd_init
from apyrobo.conformance import run_conformance
from apyrobo.core.adapters import _ADAPTER_REGISTRY, register_adapter_class


def _ns(name: str, out: str, **overrides) -> argparse.Namespace:
    defaults = dict(
        name=name, adapter=True, description="", author="",
        directory=out, force=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def scaffold(tmp_path):
    """Generate an adapter package and import its adapter class."""
    created: list[tuple[str, str]] = []  # (module_name, scheme)

    def _make(scheme: str):
        out = tmp_path / scheme
        cmd_init(_ns(scheme, str(out)))
        module_name = "apyrobo_adapter_" + scheme.replace("-", "_")
        sys.path.insert(0, str(out / "src"))
        module = importlib.import_module(module_name)
        created.append((module_name, scheme))
        class_name = (
            "".join(p.capitalize() for p in scheme.split("-")) + "Adapter"
        )
        return out, getattr(module, class_name)

    yield _make
    for module_name, scheme in created:
        sys.modules.pop(module_name, None)
        sys.modules.pop(module_name + ".adapter", None)
        _ADAPTER_REGISTRY.pop(scheme, None)
    sys.path[:] = [p for p in sys.path if str(tmp_path) not in p]


class TestAdapterScaffold:
    def test_creates_package_structure(self, scaffold, tmp_path):
        out, _ = scaffold("zenith")
        assert (out / "pyproject.toml").exists()
        assert (out / "src" / "apyrobo_adapter_zenith" / "adapter.py").exists()
        assert (out / "tests" / "test_adapter.py").exists()
        assert (out / ".github" / "workflows" / "ci.yml").exists()
        assert (out / "README.md").exists()

    def test_generated_adapter_passes_strict_conformance(self, scaffold):
        _, adapter_cls = scaffold("zenith")
        register_adapter_class("zenith", adapter_cls)
        report = run_conformance("zenith://scaffold-test")
        assert report.conformant, report.render_text()
        assert report.summary["warn"] == 0, report.render_text()
        assert report.summary["skip"] == 0, report.render_text()

    def test_pyproject_declares_adapter_entry_point(self, scaffold):
        tomllib = pytest.importorskip("tomllib")
        out, _ = scaffold("zenith")
        with open(out / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        eps = data["project"]["entry-points"]["apyrobo.adapters"]
        assert eps["zenith"] == "apyrobo_adapter_zenith:ZenithAdapter"
        assert data["build-system"]["build-backend"] == "setuptools.build_meta"

    def test_hyphenated_scheme_normalizes_module_and_class(self, scaffold):
        out, adapter_cls = scaffold("acme-arm")
        assert adapter_cls.__name__ == "AcmeArmAdapter"
        assert (out / "src" / "apyrobo_adapter_acme_arm").is_dir()

    def test_first_party_scheme_collision_is_refused(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as excinfo:
            cmd_init(_ns("mock", str(tmp_path / "out")))
        assert excinfo.value.code == 1
        assert "collides with a first-party adapter" in capsys.readouterr().err

    def test_invalid_scheme_is_refused(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            cmd_init(_ns("9bad scheme!", str(tmp_path / "out")))
        assert "not a valid URI scheme" in capsys.readouterr().err

    def test_refuses_existing_dir_without_force(self, tmp_path):
        out = tmp_path / "taken"
        out.mkdir()
        with pytest.raises(SystemExit):
            cmd_init(_ns("taken", str(out)))


class TestEntryPointFallback:
    def test_get_adapter_loads_from_entry_point(self, scaffold, monkeypatch):
        """Unknown schemes resolve via the apyrobo.adapters entry-point
        group — the pip-install experience, without the pip install."""
        import apyrobo.core.adapters as adapters_mod

        _, adapter_cls = scaffold("zenith")

        class FakeEntryPoint:
            name = "zenith"
            value = "apyrobo_adapter_zenith:ZenithAdapter"

            def load(self):
                return adapter_cls

        monkeypatch.setattr(
            adapters_mod, "_adapter_entry_points", lambda: [FakeEntryPoint()]
        )
        assert "zenith" not in _ADAPTER_REGISTRY
        adapter = adapters_mod.get_adapter("zenith", "ep-test")
        assert isinstance(adapter, adapter_cls)
        assert _ADAPTER_REGISTRY["zenith"] is adapter_cls  # cached

    def test_unknown_scheme_still_raises_value_error(self, monkeypatch):
        import apyrobo.core.adapters as adapters_mod

        monkeypatch.setattr(adapters_mod, "_adapter_entry_points", lambda: [])
        with pytest.raises(ValueError, match="No adapter registered"):
            adapters_mod.get_adapter("nonexistent-fuzz-scheme", "x")

    def test_broken_entry_point_degrades_to_value_error(self, monkeypatch):
        import apyrobo.core.adapters as adapters_mod

        class BrokenEntryPoint:
            name = "broken-ep"
            value = "nope:Nope"

            def load(self):
                raise ImportError("module vanished")

        monkeypatch.setattr(
            adapters_mod, "_adapter_entry_points", lambda: [BrokenEntryPoint()]
        )
        with pytest.raises(ValueError, match="No adapter registered"):
            adapters_mod.get_adapter("broken-ep", "x")


class TestSkillScaffoldBackendFix:
    def test_skill_scaffold_uses_real_build_backend(self, tmp_path):
        tomllib = pytest.importorskip("tomllib")
        cmd_init(
            argparse.Namespace(
                name="buildable", adapter=False, description="", author="",
                directory=str(tmp_path / "out"), force=False,
            )
        )
        with open(tmp_path / "out" / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["build-system"]["build-backend"] == "setuptools.build_meta"
