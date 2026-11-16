"""A negative speed on unitree:// keeps the asked direction and the model's cap.

UnitreeAdapter.move and rotate scaled the velocity by ``min(speed, cap)``, so a
negative speed (which the rotate tool and SafetyEnforcer pass on with its sign)
reversed the walk or the turn, and one below minus the cap was not capped.
"""

from __future__ import annotations

import pytest

from apyrobo.core import unitree_adapter
from apyrobo.core.unitree_adapter import UnitreeAdapter


class FakeSportClient:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float, float]] = []

    def Move(self, vx: float, vy: float, vyaw: float) -> None:  # noqa: N802 - SDK name
        self.moves.append((vx, vy, vyaw))


@pytest.fixture
def go2(monkeypatch: pytest.MonkeyPatch) -> tuple[UnitreeAdapter, FakeSportClient]:
    monkeypatch.setattr(unitree_adapter, "_UNITREE_SDK_AVAILABLE", True)
    adapter = UnitreeAdapter("unitree://go2")
    client = FakeSportClient()
    adapter._sport_client = client
    return adapter, client


@pytest.mark.parametrize("speed", [-0.5, -100.0])
def test_a_negative_turn_speed_turns_the_asked_way_within_the_cap(go2, speed):
    adapter, client = go2
    adapter.rotate(1.57, speed=speed)
    _, _, vyaw = client.moves[-1]
    assert vyaw > 0
    assert vyaw <= 1.0
    adapter.rotate(-1.57, speed=speed)
    _, _, vyaw = client.moves[-1]
    assert vyaw < 0
    assert vyaw >= -1.0


def test_a_negative_turn_speed_turns_as_fast_as_the_same_positive_one(go2):
    adapter, client = go2
    adapter.rotate(1.57, speed=0.4)
    adapter.rotate(1.57, speed=-0.4)
    assert client.moves[0] == client.moves[1]


@pytest.mark.parametrize("speed", [-0.5, -100.0])
def test_a_negative_walk_speed_walks_toward_the_target_within_the_cap(go2, speed):
    adapter, client = go2
    adapter.move(2.0, -1.0, speed=speed)
    vx, vy, _ = client.moves[-1]
    assert vx > 0
    assert vy < 0
    assert abs(vx) <= 1.5
    assert abs(vy) <= 1.5


def test_a_negative_walk_speed_walks_as_fast_as_the_same_positive_one(go2):
    adapter, client = go2
    adapter.move(2.0, 1.0, speed=0.6)
    adapter.move(2.0, 1.0, speed=-0.6)
    assert client.moves[0] == client.moves[1]


def test_positive_speeds_are_unchanged(go2):
    adapter, client = go2
    adapter.rotate(1.57, speed=0.5)
    assert client.moves[-1] == (0.0, 0.0, 0.5)
    adapter.rotate(1.57, speed=5.0)
    assert client.moves[-1] == (0.0, 0.0, 1.0)
    adapter.move(3.0, 0.0, speed=0.75)
    assert client.moves[-1] == pytest.approx((0.75, 0.0, 0.0))
