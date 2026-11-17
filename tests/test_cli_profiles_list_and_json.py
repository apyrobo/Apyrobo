"""apyrobo profiles: 'list' works as a subcommand, and --json given before a
subcommand ('apyrobo profiles --json show NAME') is kept, not reset."""
from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from apyrobo import cli as cli_mod


def _run(capsys, *argv: str) -> str:
    with patch.object(sys, "argv", ["apyrobo", *argv]):
        cli_mod.main()
    return capsys.readouterr().out


def test_profiles_list_subcommand_lists_profiles(capsys):
    out = _run(capsys, "profiles", "list")
    assert out.splitlines()[0].startswith("NAME")
    assert "cpu-only" in out


def test_profiles_list_matches_bare_profiles(capsys):
    assert _run(capsys, "profiles", "list") == _run(capsys, "profiles")


@pytest.mark.parametrize("argv", [
    ("profiles", "list", "--json"),
    ("profiles", "--json", "list"),
    ("profiles", "--json"),
])
def test_profiles_list_json(capsys, argv):
    data = json.loads(_run(capsys, *argv))
    assert "cpu-only" in {p["name"] for p in data}


@pytest.mark.parametrize("argv", [
    ("profiles", "show", "cpu-only", "--json"),
    ("profiles", "--json", "show", "cpu-only"),
])
def test_profiles_show_json_either_side(capsys, argv):
    data = json.loads(_run(capsys, *argv))
    assert data["name"] == "cpu-only"


def test_profiles_show_without_json_is_text(capsys):
    out = _run(capsys, "profiles", "show", "cpu-only")
    assert out.startswith("Profile: cpu-only")


@pytest.mark.parametrize("argv", [
    ("profiles", "detect", "--json"),
    ("profiles", "--json", "detect"),
])
def test_profiles_detect_json_either_side(capsys, argv):
    import apyrobo.profiles.schema as schema

    class _Fake:
        def to_dict(self):
            return {"recommended_profile": "cpu-only", "confidence": "low"}

    with patch.object(schema, "detect_profile", return_value=_Fake()):
        out = _run(capsys, *argv)
    assert json.loads(out)["recommended_profile"] == "cpu-only"
