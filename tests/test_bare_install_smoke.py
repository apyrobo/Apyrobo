"""Tests for scripts/bare_install_smoke.py, the CI bare-install guard."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bare_install_smoke.py"


def _run(pythonpath: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, env=env, timeout=120,
        cwd=str(pythonpath) if pythonpath is not None else None,
    )


def _fake_apyrobo(tmp_path: Path, body: str) -> Path:
    pkg = tmp_path / "apyrobo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(body)
    return tmp_path


def test_real_package_passes_import_checks() -> None:
    out = _run().stdout
    assert "PASS  import under 1500 ms — fastest of 3 fresh interpreters" in out
    assert "PASS  import pulls in no optional heavy dependency" in out


def test_heavy_optional_dependency_on_import_path_fails(tmp_path: Path) -> None:
    root = _fake_apyrobo(tmp_path, "import numpy\n")
    (root / "numpy.py").write_text("")  # stand-in, so the test needs no numpy
    proc = _run(root)
    assert "FAIL  import pulls in no optional heavy dependency — numpy" in proc.stdout
    assert proc.returncode != 0


def test_slow_import_fails_budget_in_every_interpreter(tmp_path: Path) -> None:
    root = _fake_apyrobo(tmp_path, "import time\ntime.sleep(1.6)\n")
    proc = _run(root)
    assert "FAIL  import under 1500 ms" in proc.stdout
    assert proc.returncode != 0
