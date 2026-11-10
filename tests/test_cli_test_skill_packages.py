"""apyrobo test-skill finds skills in installed packages and in module files.

A package is "installed" here by putting it and a dist-info folder with an
apyrobo.skills entry point on PYTHONPATH, which importlib.metadata reads the
same way it reads site-packages.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _write_pkg(site: Path, module: str, ep_target: str, files: dict[str, str]) -> None:
    pkg = site / module
    pkg.mkdir(parents=True)
    for name, body in files.items():
        (pkg / name).write_text(textwrap.dedent(body), encoding="utf-8")
    dist = site / f"{module}-0.1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {module}\nVersion: 0.1.0\n", encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(
        f"[apyrobo.skills]\n{module} = {ep_target}\n", encoding="utf-8",
    )


def _run(cwd: Path, *argv: str, site: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if site is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(site), env.get("PYTHONPATH", "")) if p
        )
    return subprocess.run(
        [sys.executable, "-m", "apyrobo", "test-skill", *argv],
        cwd=cwd, capture_output=True, text=True, timeout=120, env=env,
    )


_SKILLS = """\
    from apyrobo.skills.decorators import skill

    @skill(description="Wave", capability="custom")
    def wave_hello(times: int = 1) -> bool:
        print(f"waving {times}")
        return True
"""

_REGISTER = """\
    from . import skills  # noqa: F401  (runs the @skill decorators)

    def register() -> None:
        from apyrobo.skills.library import SkillLibrary
        SkillLibrary.from_decorated()
"""


def test_installed_package_skill_runs_by_id(tmp_path):
    site = tmp_path / "site"
    _write_pkg(site, "apyrobo_skills_wave", "apyrobo_skills_wave:register",
               {"__init__.py": _REGISTER, "skills.py": _SKILLS})
    proc = _run(tmp_path, "wave_hello", "--params", '{"times": 3}', site=site)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Skill:    wave_hello" in proc.stdout
    assert "waving 3" in proc.stdout


def test_same_id_without_the_package_is_not_found(tmp_path):
    proc = _run(tmp_path, "wave_hello")
    assert proc.returncode != 0
    assert "'wave_hello' not found" in proc.stderr
    assert "apyrobo.skills entry point" in proc.stderr


def test_a_broken_package_is_reported_and_the_others_still_load(tmp_path):
    site = tmp_path / "site"
    _write_pkg(site, "apyrobo_skills_broken", "apyrobo_skills_broken:register",
               {"__init__.py": "def register():\n    raise RuntimeError('boom')\n"})
    _write_pkg(site, "apyrobo_skills_wave", "apyrobo_skills_wave:register",
               {"__init__.py": _REGISTER, "skills.py": _SKILLS})
    proc = _run(tmp_path, "wave_hello", site=site)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "'apyrobo_skills_broken' failed to load: boom" in proc.stderr
    assert "waving 1" in proc.stdout


def test_entry_point_naming_a_module_loads_by_import(tmp_path):
    site = tmp_path / "site"
    _write_pkg(site, "apyrobo_skills_wave", "apyrobo_skills_wave.skills",
               {"__init__.py": "", "skills.py": _SKILLS})
    proc = _run(tmp_path, "wave_hello", site=site)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "waving 1" in proc.stdout


def test_builtin_skill_does_not_load_packages(tmp_path):
    site = tmp_path / "site"
    _write_pkg(site, "apyrobo_skills_broken", "apyrobo_skills_broken:register",
               {"__init__.py": "def register():\n    raise RuntimeError('boom')\n"})
    proc = _run(tmp_path, "navigate_to", site=site)
    assert "failed to load" not in proc.stderr


def test_file_with_one_skill_not_named_after_it_runs_that_skill(tmp_path):
    (tmp_path / "moves.py").write_text(textwrap.dedent(_SKILLS), encoding="utf-8")
    proc = _run(tmp_path, "moves.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Skill:    wave_hello" in proc.stdout


def test_file_with_several_skills_none_named_after_it_is_refused(tmp_path):
    body = textwrap.dedent(_SKILLS) + textwrap.dedent("""\

        @skill(description="Nod", capability="custom")
        def nod() -> bool:
            return True
    """)
    (tmp_path / "moves.py").write_text(body, encoding="utf-8")
    proc = _run(tmp_path, "moves.py")
    assert proc.returncode == 1
    assert "defines 2 skills (nod, wave_hello)" in proc.stderr
