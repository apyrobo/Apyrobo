"""Tests for scripts/release_smoke.py, the release-day check of a published package."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "release_smoke", ROOT / "scripts" / "release_smoke.py"
)
rs = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["release_smoke"] = rs
_spec.loader.exec_module(rs)


def _good() -> dict:
    return {
        "server_name": "apyrobo",
        "server_version": "9.9.9",
        "tools": ["move_to", "robot_status", "stop"],
        "clamped": {"is_error": False, "text": "", "payload": {
            "safety_interventions": [
                {"type": "speed_clamped", "requested": 2.0, "enforced": 0.5}]}},
        "refused": {"is_error": True, "text": "REFUSED by safety policy: ...",
                    "structured": {"code": "collision_zone"}},
        "before": {"x": 1.0, "y": 0.0},
        "after": {"x": 1.0, "y": 0.0},
    }


def test_wheel_version_from_file_name():
    assert rs.wheel_version(Path("apyrobo-4.2.0-py3-none-any.whl")) == "4.2.0"
    assert rs.wheel_version(Path("apyrobo-4.2.0.tar.gz")) is None


def test_server_is_started_with_the_promised_limits():
    args = rs.server_args()
    assert args[:3] == ["mcp", "--robot", "mock://turtlebot4"]
    assert args[args.index("--max-speed") + 1] == "0.5"
    assert args[args.index("--no-go") + 1] == "2,2,4,4"


def test_evaluate_passes_a_good_report():
    checks = rs.evaluate(_good())
    assert [c.ok for c in checks] == [True, True, True, True], [c.line() for c in checks]
    assert "collision_zone" in checks[2].detail


def test_evaluate_fails_when_the_client_could_not_run():
    checks = rs.evaluate("Connection closed")
    assert not any(c.ok for c in checks)
    assert checks[0].detail == "Connection closed"


def test_evaluate_fails_an_unclamped_move():
    report = _good()
    report["clamped"]["payload"]["safety_interventions"] = []
    assert not rs.evaluate(report)[1].ok


def test_evaluate_fails_a_move_that_was_not_refused():
    report = _good()
    report["refused"] = {"is_error": False, "text": '{"position": {"x": 3.0}}',
                         "structured": None}
    assert not rs.evaluate(report)[2].ok


def test_evaluate_fails_when_the_robot_moved_on_a_refusal():
    report = _good()
    report["after"] = {"x": 3.0, "y": 3.0}
    check = rs.evaluate(report)[3]
    assert not check.ok
    assert "->" in check.detail


def test_evaluate_fails_without_a_position():
    report = _good()
    report["before"] = report["after"] = None
    assert not rs.evaluate(report)[3].ok


def test_last_line_skips_exception_group_boxes():
    err = ("Traceback ...\n  | ExceptionGroup: x\n  +-+---- 1 ----\n"
           "    | MCPError: Connection closed\n    +------\n")
    assert rs._last_line(err) == "MCPError: Connection closed"


def test_needs_a_version_or_a_wheel(capsys):
    with pytest.raises(SystemExit) as exc:
        rs.main([])
    assert exc.value.code == 2
    assert "VERSION" in capsys.readouterr().err


def test_missing_wheel_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        rs.main(["--wheel", str(tmp_path / "nope.whl")])
    assert exc.value.code == 2


def test_smoke_passes_on_a_wheel_built_from_this_checkout(tmp_path, capsys):
    """End to end: build a wheel, install it in a fresh venv, drive it with the SDK."""
    pytest.importorskip("mcp")
    pytest.importorskip("setuptools")
    build = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "--no-build-isolation",
         "-w", str(tmp_path), str(ROOT)],
        capture_output=True, text=True, timeout=300,
    )
    assert build.returncode == 0, build.stderr
    [wheel] = tmp_path.glob("apyrobo-*.whl")

    code = rs.main(["--wheel", str(wheel), "--system-site-packages"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "FAIL" not in out
    assert out.count("PASS") == 6
    assert "installed version" in out
    assert "the robot did not move on the refusal" in out
