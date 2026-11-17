"""
examples/06_mcp_client_sdk.py on a plain install (no MCP SDK) must say what
to install instead of stopping with a ModuleNotFoundError traceback.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "06_mcp_client_sdk.py"

# Hide the SDK and anyio, then run the example as a script.
RUNNER = """
import runpy, sys
for name in ("mcp", "anyio"):
    sys.modules[name] = None
runpy.run_path(sys.argv[1], run_name="__main__")
"""


def _run_without_sdk():
    return subprocess.run(
        [sys.executable, "-c", RUNNER, str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_exits_1_without_a_traceback():
    result = _run_without_sdk()
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_names_the_install_command_and_the_no_extras_demo():
    err = _run_without_sdk().stderr
    assert 'pip install "mcp>=2.2"' in err
    assert "examples/05_mcp_safety_demo.py" in err
    assert (EXAMPLE.parent / "05_mcp_safety_demo.py").exists()
