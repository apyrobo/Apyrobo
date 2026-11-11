"""The dashboard escapes text it did not write before putting it in HTML.

Robot names, URIs, error messages, safety details and skill descriptions come
from adapters, policy files and installed skill packages. Rendered raw, a
``<script>`` in any of them would run in the browser of whoever opens the
dashboard.
"""

from __future__ import annotations

from apyrobo.core.robot import Robot
from apyrobo.dashboard import (
    RobotDashboard,
    _render_dashboard_html,
    _render_history_partial,
    _render_live_dashboard_html,
    _render_safety_partial,
    _render_skills_partial,
    _render_status_partial,
)

EVIL = '<script>alert("x")</script>'
ESCAPED = "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"


def _assert_escaped(page: str) -> None:
    assert "<script>" not in page
    assert ESCAPED in page


def test_status_partial_escapes_robot_fields() -> None:
    page = _render_status_partial({
        "connected": True,
        "name": EVIL,
        "uri": EVIL,
        "position": {"x": 1.0, "y": 2.0},
        "battery_pct": 80,
        "capabilities": [{"type": EVIL}],
        "max_speed": 1.5,
    })
    assert page.count(ESCAPED) == 3
    assert "<script>" not in page
    assert "(1.0, 2.0)" in page
    assert "1.5 m/s" in page


def test_status_partial_escapes_connection_error() -> None:
    _assert_escaped(_render_status_partial({"connected": False, "error": EVIL}))


def test_history_partial_escapes_skill_id_and_status() -> None:
    page = _render_history_partial([
        {"skill_id": EVIL, "status": EVIL, "elapsed_ms": 1.0, "timestamp": 0.0},
    ])
    assert page.count(ESCAPED) == 2
    assert "<script>" not in page


def test_safety_partial_escapes_event_and_detail() -> None:
    page = _render_safety_partial([
        {"event_type": EVIL, "detail": EVIL, "timestamp": 0.0},
    ])
    assert page.count(ESCAPED) == 2
    assert "<script>" not in page


def test_skills_partial_escapes_package_supplied_text() -> None:
    page = _render_skills_partial([
        {"skill_id": EVIL, "name": EVIL, "description": "<img src=x onerror=alert(1)>"},
    ])
    assert page.count(ESCAPED) == 2
    assert "<img" not in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page


def test_live_page_escapes_robot_uri() -> None:
    page = _render_live_dashboard_html(EVIL)
    assert ESCAPED in page
    # The page still loads htmx; only the URI text is escaped.
    assert '<script src="https://unpkg.com/htmx.org@1.9.12"></script>' in page
    assert page.count("<script") == 1


def test_metrics_page_escapes_tier_and_status() -> None:
    page = _render_dashboard_html({
        "status": EVIL,
        "router": {"tiers": [{"name": EVIL, "circuit_state": EVIL}]},
    })
    assert page.count(ESCAPED) == 3
    assert "<script>" not in page


def test_recorded_safety_event_is_escaped_end_to_end() -> None:
    dash = RobotDashboard(Robot.discover("mock://turtlebot4"), robot_uri="mock://turtlebot4")
    dash.record_safety_event("rejected", EVIL)
    dash.record_skill(EVIL, "ok", 3.0)
    _assert_escaped(_render_safety_partial(dash.get_safety_events()))
    _assert_escaped(_render_history_partial(dash.get_skill_history()))


def test_plain_text_is_unchanged() -> None:
    page = _render_safety_partial([
        {"event_type": "speed_clamped", "detail": "speed 2.0 -> 1.5 m/s", "timestamp": 0.0},
    ])
    assert "speed_clamped" in page
    assert "speed 2.0 -&gt; 1.5 m/s" in page
