"""Tests for multi-site fleet management."""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from apyrobo.fleet.multisite import (
    MultiSiteError,
    MultiSiteManager,
    SiteConfig,
    SiteStatus,
)


def make_site(site_id: str = "site-a", api_url: str = "http://site-a.example.com") -> SiteConfig:
    return SiteConfig(
        site_id=site_id,
        name=f"Site {site_id}",
        location="Building A",
        api_url=api_url,
        api_key="secret",
        timezone="UTC",
    )


def make_status(site_id: str, active_tasks: int = 0, robot_count: int = 5, online: bool = True):
    return SiteStatus(
        site_id=site_id,
        online=online,
        robot_count=robot_count,
        active_tasks=active_tasks,
        last_heartbeat=datetime.utcnow(),
    )


class TestSiteRegistration:
    def test_register_and_list(self):
        mgr = MultiSiteManager("local")
        cfg = make_site("site-a")
        mgr.register_site(cfg)
        assert len(mgr.list_sites()) == 1
        assert mgr.list_sites()[0].site_id == "site-a"

    def test_register_multiple_sites(self):
        mgr = MultiSiteManager("local")
        for i in range(3):
            mgr.register_site(make_site(f"site-{i}", f"http://site-{i}.example.com"))
        assert len(mgr.list_sites()) == 3

    def test_unregister_site(self):
        mgr = MultiSiteManager("local")
        mgr.register_site(make_site("site-a"))
        mgr.unregister_site("site-a")
        assert len(mgr.list_sites()) == 0

    def test_unregister_nonexistent_is_noop(self):
        mgr = MultiSiteManager("local")
        mgr.unregister_site("ghost-site")  # should not raise

    def test_list_empty(self):
        mgr = MultiSiteManager("local")
        assert mgr.list_sites() == []

    def test_get_site_status_before_sync(self):
        mgr = MultiSiteManager("local")
        mgr.register_site(make_site("site-a"))
        assert mgr.get_site_status("site-a") is None

    def test_get_site_status_after_manual_set(self):
        mgr = MultiSiteManager("local")
        mgr.register_site(make_site("site-a"))
        status = make_status("site-a", active_tasks=3)
        mgr._site_statuses["site-a"] = status
        result = mgr.get_site_status("site-a")
        assert result is not None
        assert result.active_tasks == 3


class TestRouting:
    def _mgr_with_statuses(self) -> MultiSiteManager:
        mgr = MultiSiteManager("local")
        for sid, tasks in [("site-a", 5), ("site-b", 2), ("site-c", 8)]:
            mgr.register_site(make_site(sid, f"http://{sid}.example.com"))
            mgr._site_statuses[sid] = make_status(sid, active_tasks=tasks)
        return mgr

    def test_select_site_least_loaded(self):
        mgr = self._mgr_with_statuses()
        selected = mgr._select_site("least_loaded")
        assert selected == "site-b"  # fewest active_tasks=2

    def test_select_site_round_robin_cycles(self):
        mgr = self._mgr_with_statuses()
        sites = [mgr._select_site("round_robin") for _ in range(6)]
        site_ids = list(mgr._sites.keys())
        assert sites[0] == site_ids[0]
        assert sites[1] == site_ids[1]
        assert sites[2] == site_ids[2]
        assert sites[3] == site_ids[0]

    def test_select_site_closest_returns_first(self):
        mgr = self._mgr_with_statuses()
        selected = mgr._select_site("closest")
        assert selected == list(mgr._sites.keys())[0]

    def test_select_site_unknown_strategy_raises(self):
        mgr = self._mgr_with_statuses()
        with pytest.raises(MultiSiteError, match="Unknown routing strategy"):
            mgr._select_site("teleport")

    def test_select_site_no_sites_returns_none(self):
        mgr = MultiSiteManager("local")
        assert mgr._select_site("least_loaded") is None

    def test_route_task_no_sites_raises(self):
        mgr = MultiSiteManager("local")
        with pytest.raises(MultiSiteError, match="No sites available"):
            mgr.route_task({"type": "move"})


class TestTaskSubmission:
    def _mock_urlopen(self, response_body: dict):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_body).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_submit_task_returns_task_id(self):
        mgr = MultiSiteManager("local")
        mgr.register_site(make_site("site-a", "http://site-a.example.com"))
        mock_resp = self._mock_urlopen({"task_id": "task-123"})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            task_id = mgr.submit_task_to_site("site-a", {"type": "navigate"})
        assert task_id == "task-123"

    def test_submit_task_unknown_site_raises(self):
        mgr = MultiSiteManager("local")
        with pytest.raises(MultiSiteError, match="Unknown site"):
            mgr.submit_task_to_site("ghost", {"type": "move"})

    def test_get_task_status(self):
        mgr = MultiSiteManager("local")
        mgr.register_site(make_site("site-a", "http://site-a.example.com"))
        mock_resp = self._mock_urlopen({"status": "running", "progress": 0.5})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            status = mgr.get_task_status("site-a", "task-123")
        assert status["status"] == "running"

    def test_get_task_status_unknown_site_raises(self):
        mgr = MultiSiteManager("local")
        with pytest.raises(MultiSiteError, match="Unknown site"):
            mgr.get_task_status("ghost", "task-1")


class TestSyncState:
    def test_sync_state_populates_statuses(self):
        mgr = MultiSiteManager("local")
        mgr.register_site(make_site("site-a", "http://site-a.example.com"))
        payload = {
            "online": True,
            "robot_count": 4,
            "active_tasks": 2,
            "last_heartbeat": datetime.utcnow().isoformat(),
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(payload).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            results = mgr.sync_state()
        assert "site-a" in results
        assert results["site-a"].robot_count == 4
        assert mgr.get_site_status("site-a").active_tasks == 2

    def test_sync_state_marks_offline_on_error(self):
        import urllib.error

        mgr = MultiSiteManager("local")
        mgr.register_site(make_site("site-a", "http://site-a.example.com"))
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            results = mgr.sync_state()
        assert results["site-a"].online is False
