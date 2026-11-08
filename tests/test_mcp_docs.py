"""Keep docs/mcp.md in step with what apyrobo/mcp_server.py implements."""

from __future__ import annotations

import re
from pathlib import Path

from apyrobo import mcp_server

DOC = Path(__file__).resolve().parent.parent / "docs" / "mcp.md"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_every_supported_protocol_version_is_documented():
    text = _doc()
    versions = mcp_server.SUPPORTED_PROTOCOL_VERSIONS + mcp_server.MODERN_PROTOCOL_VERSIONS
    missing = [v for v in versions if v not in text]
    assert not missing, f"docs/mcp.md does not mention protocol versions {missing}"


def test_no_unsupported_protocol_version_is_documented():
    text = _doc()
    supported = set(mcp_server.SUPPORTED_PROTOCOL_VERSIONS + mcp_server.MODERN_PROTOCOL_VERSIONS)
    documented = set(re.findall(r"\b20\d\d-\d\d-\d\d\b", text))
    extra = documented - supported
    assert not extra, f"docs/mcp.md names protocol versions the server does not support: {extra}"


def test_documented_methods_are_handled():
    source = Path(mcp_server.__file__).read_text(encoding="utf-8")
    for method in ("initialize", "ping", "tools/list", "tools/call", "server/discover"):
        assert f"`{method}`" in _doc(), f"docs/mcp.md does not list {method}"
        assert f'"{method}"' in source, f"mcp_server.py does not handle {method}"


def test_relative_links_point_at_existing_files():
    broken = []
    for target in re.findall(r"\]\((?!https?://|#)([^)#]+)\)", _doc()):
        if not (DOC.parent / target).exists():
            broken.append(target)
    assert not broken, f"docs/mcp.md links to missing files: {broken}"


def test_the_uvx_ros2_pin_is_documented():
    """51b: the isolated-environment trap and the verified pin, not just sourcing."""
    text = _doc()
    for needed in (
        "--no-managed-python --python 3.12",
        "--with 'numpy<2' --with transforms3d",
        "/usr/lib/python3/dist-packages",
        "ros-$ROS_DISTRO-nav2-msgs",
    ):
        assert needed in text, f"docs/mcp.md no longer documents {needed!r}"


def test_the_tools_table_is_the_whole_tool_list():
    server = mcp_server.build_server(robot_uri="mock://docs-tools")
    served = {tool["name"] for tool in server.toolbox.list_tools()}
    section = _doc().split("## Tools")[1].split("\n## ")[0]
    documented = set(re.findall(r"^\| `(\w+)", section, flags=re.MULTILINE))
    assert documented == served
    assert not any("topic" in name or "service" in name for name in served)


def test_the_enforcer_reads_no_live_sensors_as_the_docs_say():
    server = mcp_server.build_server(robot_uri="mock://docs-sensors")
    assert server.toolbox.enforcer._world_state is None
    text = " ".join(_doc().split())
    assert "It checks policy, not live sensors." in text
    assert "human-proximity check does not run" in text


def test_docs_say_how_the_speed_limit_reaches_a_nav2_drive():
    text = _doc()
    assert "### Speed on `ros2://`" in text
    assert "`speed_limit` topic" in text
    # The fallback when nothing takes the limit stays documented, and so do
    # the limits of what CI checks.
    assert '"speed_limit_applied": false' in text
    assert "Humble with DWB" in text
    assert "Clamps and no-go refusals happen" not in text
    llms = (DOC.parent.parent / "llms.txt").read_text(encoding="utf-8")
    assert "`speed_limit` topic" in llms
    assert "not applied" in llms
    robot = (DOC.parent / "first-real-robot.md").read_text(encoding="utf-8")
    assert "mcp.md#speed-on-ros2" in robot
