"""apyrobo pkg validate / pack / install must not drop a skill silently.

A skill id listed in skill-package.json with no skills/<id>.json, or a
skills/*.json file the manifest does not list, used to pass validation and
pack into an archive without that skill.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apyrobo import cli
from apyrobo.skills.package import SkillPackage
from apyrobo.skills.registry import SkillRegistry


def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["apyrobo", *argv])
    cli.main()


def _make_pkg(tmp_path: Path, listed: list[str], files: list[str]) -> Path:
    d = tmp_path / "mypkg"
    SkillPackage.init(name="mypkg", directory=str(d))
    manifest = json.loads((d / "skill-package.json").read_text())
    manifest["skills"] = listed
    (d / "skill-package.json").write_text(json.dumps(manifest))
    for sid in files:
        (d / "skills" / f"{sid}.json").write_text(
            json.dumps({"skill_id": sid, "name": sid.replace("_", " ").title()})
        )
    return d


def test_complete_package_is_valid(tmp_path: Path) -> None:
    d = _make_pkg(tmp_path, ["navigate_to"], ["navigate_to"])
    assert SkillPackage.load(d).validate() == []


def test_listed_skill_without_file_is_an_error(tmp_path: Path) -> None:
    d = _make_pkg(tmp_path, ["navigate_to", "pick_up"], ["navigate_to"])
    errors = SkillPackage.load(d).validate()
    assert errors == [
        "skills/pick_up.json not found (the skill is listed in skill-package.json)"
    ]


def test_unlisted_skill_file_is_an_error_naming_the_fix(tmp_path: Path) -> None:
    d = _make_pkg(tmp_path, [], ["navigate_to"])
    errors = SkillPackage.load(d).validate()
    assert any(
        'skills/navigate_to.json is not listed in skill-package.json "skills"' in e
        and 'add "navigate_to" there' in e
        for e in errors
    ), errors


def test_only_missing_files_does_not_also_say_no_skills(tmp_path: Path) -> None:
    d = _make_pkg(tmp_path, ["pick_up"], [])
    errors = SkillPackage.load(d).validate()
    assert errors == [
        "skills/pick_up.json not found (the skill is listed in skill-package.json)"
    ]


def test_empty_package_still_says_no_skills(tmp_path: Path) -> None:
    d = _make_pkg(tmp_path, [], [])
    assert SkillPackage.load(d).validate() == [
        "Package must contain at least one skill"
    ]


def test_cli_validate_and_pack_refuse_a_missing_skill(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    d = _make_pkg(tmp_path, ["navigate_to", "pick_up"], ["navigate_to"])
    for argv in (["pkg", "validate", str(d)], ["pkg", "pack", str(d)]):
        with pytest.raises(SystemExit) as exc:
            _run(monkeypatch, argv)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "skills/pick_up.json not found" in out
    assert not list(tmp_path.rglob("*.skillpkg"))


def test_install_refuses_a_directory_missing_a_listed_skill(tmp_path: Path) -> None:
    d = _make_pkg(tmp_path, ["navigate_to", "pick_up"], ["navigate_to"])
    registry = SkillRegistry(tmp_path / "registry")
    with pytest.raises(Exception, match="pick_up.json not found"):
        registry.install(str(d))


def test_pkg_init_next_steps_name_export_and_validate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = tmp_path / "newpkg"
    _run(monkeypatch, ["pkg", "init", "newpkg", "--directory", str(d)])
    out = capsys.readouterr().out
    assert f"apyrobo skills --export navigate_to > {d}/skills/navigate_to.json" in out
    assert f"apyrobo pkg validate {d}" in out
