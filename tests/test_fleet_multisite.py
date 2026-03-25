"""Tests for multi-site fleet management."""
import json
import pytest
from unittest.mock import patch, MagicMock
from apyrobo.fleet.multisite import MultiSiteManager, SiteConfig, SiteStatus, MultiSiteError


@pytest.fixture
def manager():
    m = MultiSiteManager("local")
    return m


@pytest.fixture
def sites():
    return [
        SiteConfig(site_id="site-a", name="Site A", location="NYC", api_url="http://site-a:8080"),
        SiteConfig(site_id="site-b", name="Site B", location="LA", api_url="http://site-b:8080"),
        SiteConfig(site_id="site-c", name="Site C", location="Chicago", api_url="http://site-c:8080"),
    ]


def test_register_site(manager, sites):
    manager.register_site(sites[0])
    assert len(manager.list_sites()) == 1


def test_unregister_site(manager, sites):
    manager.register_site(sites[0])
    manager.unregister_site("site-a")
    assert len(manager.list_sites()) == 0


def test_list_sites(manager, sites):
    for s in sites:
        manager.register_site(s)
    assert len(manager.list_sites()) == 3


def test_get_site_status_initial(manager, sites):
    manager.register_site(sites[0])
    status = manager.get_site_status("site-a")
    assert status is not None
    assert status.online is False


def test_get_site_status_unknown(manager):
    assert manager.get_site_status("nonexistent") is None


def test_route_task_least_loaded(manager, sites):
    for s in sites:
        manager.register_site(s)
    # Manually set site statuses
    manager._site_statuses["site-a"] = SiteStatus("site-a", True, 3, 10)
    manager._site_statuses["site-b"] = SiteStatus("site-b", True, 2, 2)
    manager._site_statuses["site-c"] = SiteStatus("site-c", True, 1, 5)

    with patch.object(manager, "submit_task_to_site", return_value="task-123"):
        site_id, task_id = manager.route_task({"skill": "navigate"}, "least_loaded")
    assert site_id == "site-b"  # least active_tasks


def test_route_task_round_robin(manager, sites):
    for s in sites:
        manager.register_site(s)
    manager._site_statuses["site-a"] = SiteStatus("site-a", True)
    manager._site_statuses["site-b"] = SiteStatus("site-b", True)

    with patch.object(manager, "submit_task_to_site", return_value="task-rr"):
        sid1, _ = manager.route_task({}, "round_robin")
        sid2, _ = manager.route_task({}, "round_robin")
    assert sid1 != sid2 or len(manager.list_sites()) == 1


def test_route_task_no_sites(manager):
    with pytest.raises(MultiSiteError):
        manager.route_task({})


def test_submit_task_unknown_site(manager):
    with pytest.raises(MultiSiteError, match="Unknown site"):
        manager.submit_task_to_site("nonexistent", {})


def test_submit_task_http_success(manager, sites):
    manager.register_site(sites[0])
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"task_id": "task-abc"}).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        task_id = manager.submit_task_to_site("site-a", {"skill": "move_to"})
    assert task_id == "task-abc"


def test_submit_task_http_failure(manager, sites):
    manager.register_site(sites[0])
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        with pytest.raises(MultiSiteError):
            manager.submit_task_to_site("site-a", {})


def test_sync_state_marks_offline(manager, sites):
    manager.register_site(sites[0])
    with patch("urllib.request.urlopen", side_effect=Exception("unreachable")):
        result = manager.sync_state()
    assert result["site-a"] is False


def test_sync_state_marks_online(manager, sites):
    manager.register_site(sites[0])
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"robot_count": 2, "active_tasks": 3}).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = manager.sync_state()
    assert result["site-a"] is True
    assert manager.get_site_status("site-a").robot_count == 2


def test_site_config_fields(sites):
    assert sites[0].site_id == "site-a"
    assert sites[0].timezone == "UTC"


def test_get_task_status(manager, sites):
    manager.register_site(sites[0])
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"status": "running"}).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        status = manager.get_task_status("site-a", "task-123")
    assert status["status"] == "running"
