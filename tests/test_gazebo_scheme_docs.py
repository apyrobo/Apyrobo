"""Docs and examples don't present gazebo:// as a real robot or a live Gazebo.

``gazebo://`` selects ``GazeboAdapter``, an in-memory stand-in that never
connects to Gazebo. A live Gazebo (and a real robot) is driven with ``ros2://``.
Several next-step lines and a guide used to say otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apyrobo.core.adapters import GazeboAdapter
from apyrobo.core.robot import Robot

ROOT = Path(__file__).resolve().parent.parent

CHECKED = [
    "docs/TURTLEBOT4.md",
    "docs/conformance_results.md",
    "examples/demo_full.py",
    "examples/demo_mock.py",
    "examples/workflows/patrol_loop.py",
]

CLAIMS = (
    "is an alias for `ros2://`",
    "identical adapter, different environment",
    "gazebo:// or ros2://",
    "ros2:// or\ngazebo://",
    "real robot, use: robot.discover('gazebo://",
    "gazebo classic rig",
)


def test_gazebo_scheme_is_the_in_memory_stand_in():
    # The premise: gazebo:// never reaches a simulator.
    robot = Robot.discover("gazebo://scheme-check")
    assert isinstance(robot._adapter, GazeboAdapter)
    assert "no Gazebo required or used" in (GazeboAdapter.__doc__ or "")


@pytest.mark.parametrize("rel", CHECKED)
def test_no_claim_that_gazebo_scheme_is_real(rel):
    text = (ROOT / rel).read_text().lower()
    for claim in CLAIMS:
        assert claim.lower() not in text, f"{rel} says {claim!r}"


def test_turtlebot4_guide_points_gazebo_users_at_ros2():
    text = (ROOT / "docs/TURTLEBOT4.md").read_text()
    assert "`gazebo://` is not an alias for `ros2://`" in text
    assert 'Robot.discover("gazebo://' not in text


def test_conformance_table_marks_gazebo_as_stand_in():
    rows = [
        line
        for line in (ROOT / "docs/conformance_results.md").read_text().splitlines()
        if line.startswith("| `gazebo://`")
    ]
    assert len(rows) == 1
    assert "in-memory stand-in" in rows[0]
