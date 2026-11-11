"""The ros2:// setup messages and guides point at a setup that exists.

They used to send users into ``docker compose ... exec apyrobo bash`` (no such
service) or into the ``apyrobo-api`` image "where ROS 2 is pre-installed",
which is a slim Python image without ROS 2.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

CHECKED = [
    "apyrobo/__init__.py",
    "apyrobo/core/ros2_bridge.py",
    "docs/QUICKSTART.md",
    "docs/TURTLEBOT4.md",
    "docs/deployment.md",
]

EXEC_RE = re.compile(r"docker compose -f docker/docker-compose\.yml (?:exec|up -d) ([\w-]+)")


def _services() -> set[str]:
    data = yaml.safe_load((ROOT / "docker/docker-compose.yml").read_text())
    return set(data["services"])


def test_api_image_has_no_ros2():
    # The premise of these messages: the gateway image is plain Python.
    text = (ROOT / "docker/Dockerfile").read_text()
    runtime = re.search(r"^FROM (\S+) AS runtime", text, re.M)
    assert runtime and runtime.group(1).startswith("python:")


@pytest.mark.parametrize("rel", CHECKED)
def test_compose_commands_name_real_services(rel):
    services = _services()
    for svc in EXEC_RE.findall((ROOT / rel).read_text()):
        assert svc in services, f"{rel} names compose service {svc!r}"


@pytest.mark.parametrize("rel", CHECKED)
def test_no_claim_that_the_api_container_has_ros2(rel):
    text = (ROOT / rel).read_text().lower()
    for claim in (
        "where ros 2 is pre-installed",
        "with ros 2 sourced",
        "which has ros 2",
        "inside the docker container",
        "exec apyrobo-api bash",
        "exec apyrobo bash",
    ):
        assert claim not in text, f"{rel}: {claim!r}"


def test_turtlebot4_has_no_worker_service_claim():
    assert "apyrobo-worker" not in (ROOT / "docs/TURTLEBOT4.md").read_text()


def test_quickstart_links_resolve():
    text = (ROOT / "docs/QUICKSTART.md").read_text()
    for target in re.findall(r"\]\(([\w./-]+\.md)\)", text):
        assert (ROOT / "docs" / target).exists(), target


def test_discover_without_ros2_points_at_the_real_guide():
    import apyrobo.core.ros2_bridge as bridge
    from apyrobo import Robot

    if bridge._HAS_ROS2:
        pytest.skip("ROS 2 is importable here")
    with pytest.raises(RuntimeError) as exc:
        Robot.discover("ros2://turtlebot4")
    msg = str(exc.value)
    assert "docs/first-real-robot.md" in msg
    assert "docker compose" not in msg
