"""Tests for registry CLI commands: apyrobo registry start / skill search / skill publish."""
from __future__ import annotations

import json
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from apyrobo.cli import (
    cmd_registry_start,
    cmd_skill_search,
    cmd_skill_publish,
    _cmd_registry_dispatch,
    _cmd_skill_dispatch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _search_args(query="navigation", registry="http://localhost:8080") -> SimpleNamespace:
    return SimpleNamespace(query=query, registry=registry)


def _publish_args(**kwargs) -> SimpleNamespace:
    defaults = dict(
        name="apyrobo-skills-myrobot",
        version="1.0.0",
        description="My robot skills",
        author="Test Author",
        download_url="https://example.com/pkg.whl",
        token="secret-token",
        registry="http://localhost:8080",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _registry_start_args(**kwargs) -> SimpleNamespace:
    defaults = dict(port=8080, host="0.0.0.0", db="./registry.db")
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _fake_urlopen(response_data, status=200):
    """Return a context manager that yields a fake HTTP response."""
    class _Resp:
        def read(self):
            return json.dumps(response_data).encode()
        def __enter__(self): return self
        def __exit__(self, *a): pass

    return MagicMock(return_value=_Resp())


# ---------------------------------------------------------------------------
# skill search
# ---------------------------------------------------------------------------

class TestSkillSearch:
    def test_search_returns_results(self, capsys):
        results = [
            {"name": "apyrobo-skills-patrol", "version": "1.0.0",
             "description": "Patrol skill", "tags": ["navigate"]}
        ]
        with patch("urllib.request.urlopen", _fake_urlopen(results)):
            cmd_skill_search(_search_args(query="patrol"))
        out = capsys.readouterr().out
        assert "apyrobo-skills-patrol" in out
        assert "1.0.0" in out

    def test_search_shows_result_count(self, capsys):
        results = [
            {"name": "nav-skills", "version": "2.0.0", "description": "Nav", "tags": []},
            {"name": "nav-extra", "version": "1.5.0", "description": "Extra", "tags": []},
        ]
        with patch("urllib.request.urlopen", _fake_urlopen(results)):
            cmd_skill_search(_search_args(query="nav"))
        out = capsys.readouterr().out
        assert "2 skill" in out

    def test_search_no_results_message(self, capsys):
        with patch("urllib.request.urlopen", _fake_urlopen([])):
            cmd_skill_search(_search_args(query="zzznomatch"))
        out = capsys.readouterr().out
        assert "No skills found" in out

    def test_search_registry_unreachable_exits_1(self, capsys):
        with patch("urllib.request.urlopen", side_effect=URLError("connection refused")):
            with pytest.raises(SystemExit) as exc_info:
                cmd_skill_search(_search_args())
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "registry" in err.lower() or "reach" in err.lower()

    def test_search_shows_hint_on_unreachable(self, capsys):
        with patch("urllib.request.urlopen", side_effect=URLError("refused")):
            with pytest.raises(SystemExit):
                cmd_skill_search(_search_args())
        err = capsys.readouterr().err
        assert "apyrobo registry start" in err

    def test_search_uses_custom_registry_url(self):
        captured_urls = []
        original = __import__("urllib.request", fromlist=["urlopen"]).urlopen

        def _capture(req, **kw):
            captured_urls.append(req.full_url)
            class _R:
                def read(self): return b"[]"
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _R()

        with patch("urllib.request.urlopen", side_effect=_capture):
            cmd_skill_search(_search_args(registry="http://myregistry.example.com:9000"))

        assert any("myregistry.example.com" in u for u in captured_urls)

    def test_search_encodes_query_in_url(self):
        captured = []
        def _cap(req, **kw):
            captured.append(req.full_url)
            class _R:
                def read(self): return b"[]"
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _R()

        with patch("urllib.request.urlopen", side_effect=_cap):
            cmd_skill_search(_search_args(query="my robot skill"))
        assert "my+robot+skill" in captured[0] or "my%20robot%20skill" in captured[0]


# ---------------------------------------------------------------------------
# skill publish
# ---------------------------------------------------------------------------

class TestSkillPublish:
    def test_publish_success_prints_name(self, capsys):
        response = {"status": "published", "name": "apyrobo-skills-myrobot", "version": "1.0.0"}
        with patch("urllib.request.urlopen", _fake_urlopen(response)):
            cmd_skill_publish(_publish_args())
        out = capsys.readouterr().out
        assert "apyrobo-skills-myrobot" in out

    def test_publish_sends_post_request(self):
        captured = []
        def _cap(req, **kw):
            captured.append(req)
            class _R:
                def read(self): return json.dumps({"status": "published", "name": "x", "version": "1"}).encode()
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _R()

        with patch("urllib.request.urlopen", side_effect=_cap):
            cmd_skill_publish(_publish_args())

        assert len(captured) == 1
        assert captured[0].get_method() == "POST"

    def test_publish_sends_name_and_version_in_payload(self):
        captured_body = []
        def _cap(req, **kw):
            captured_body.append(json.loads(req.data))
            class _R:
                def read(self): return json.dumps({"status": "published", "name": "x", "version": "1"}).encode()
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _R()

        with patch("urllib.request.urlopen", side_effect=_cap):
            cmd_skill_publish(_publish_args(name="test-pkg", version="2.3.4"))

        body = captured_body[0]
        assert body["package"]["name"] == "test-pkg"
        assert body["package"]["version"] == "2.3.4"

    def test_publish_http_error_exits_1(self, capsys):
        err_resp = b'{"detail": "already exists"}'
        exc = HTTPError(url="http://x", code=409, msg="Conflict", hdrs={}, fp=None)
        exc.read = lambda: err_resp
        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(SystemExit) as ei:
                cmd_skill_publish(_publish_args())
        assert ei.value.code == 1

    def test_publish_url_error_exits_1(self, capsys):
        with patch("urllib.request.urlopen", side_effect=URLError("refused")):
            with pytest.raises(SystemExit) as ei:
                cmd_skill_publish(_publish_args())
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert "apyrobo registry start" in err

    def test_publish_uses_bearer_token(self):
        captured = []
        def _cap(req, **kw):
            captured.append(dict(req.headers))
            class _R:
                def read(self): return json.dumps({"status": "published", "name": "x", "version": "1"}).encode()
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _R()

        with patch("urllib.request.urlopen", side_effect=_cap):
            cmd_skill_publish(_publish_args(token="my-secret"))

        auth = captured[0].get("Authorization", "")
        assert "my-secret" in auth


# ---------------------------------------------------------------------------
# registry start
# ---------------------------------------------------------------------------

def _mock_server_module():
    """Return a fake apyrobo.registry.server module for patching."""
    fake_server = MagicMock()
    fake_server.create_app = MagicMock(return_value=MagicMock())
    return fake_server


class TestRegistryStart:
    def test_calls_uvicorn_run(self):
        fake_uvicorn = MagicMock()
        fake_server = _mock_server_module()

        with patch.dict("sys.modules", {"uvicorn": fake_uvicorn,
                                         "apyrobo.registry.server": fake_server}):
            with patch("builtins.print"):
                cmd_registry_start(_registry_start_args())

        fake_uvicorn.run.assert_called_once()

    def test_uvicorn_called_with_correct_port(self):
        fake_uvicorn = MagicMock()
        fake_server = _mock_server_module()

        with patch.dict("sys.modules", {"uvicorn": fake_uvicorn,
                                         "apyrobo.registry.server": fake_server}):
            with patch("builtins.print"):
                cmd_registry_start(_registry_start_args(port=9999))

        _, kwargs = fake_uvicorn.run.call_args
        assert kwargs.get("port") == 9999

    def test_uvicorn_called_with_correct_host(self):
        fake_uvicorn = MagicMock()
        fake_server = _mock_server_module()

        with patch.dict("sys.modules", {"uvicorn": fake_uvicorn,
                                         "apyrobo.registry.server": fake_server}):
            with patch("builtins.print"):
                cmd_registry_start(_registry_start_args(host="127.0.0.1"))

        _, kwargs = fake_uvicorn.run.call_args
        assert kwargs.get("host") == "127.0.0.1"

    def test_uvicorn_missing_exits_1(self, capsys):
        import builtins
        real_import = builtins.__import__
        def _fake_import(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("No module named 'uvicorn'")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=_fake_import):
            with pytest.raises(SystemExit) as ei:
                cmd_registry_start(_registry_start_args())
        assert ei.value.code == 1
        err = capsys.readouterr().err
        assert "registry" in err.lower()

    def test_prints_startup_message(self, capsys):
        fake_uvicorn = MagicMock()
        fake_server = _mock_server_module()

        with patch.dict("sys.modules", {"uvicorn": fake_uvicorn,
                                         "apyrobo.registry.server": fake_server}):
            cmd_registry_start(_registry_start_args(port=8080))

        out = capsys.readouterr().out
        assert "8080" in out


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_registry_dispatch_start(self):
        args = SimpleNamespace(registry_command="start", port=8080,
                               host="0.0.0.0", db="./registry.db")
        fake_uvicorn = MagicMock()
        fake_server = _mock_server_module()
        with patch.dict("sys.modules", {"uvicorn": fake_uvicorn,
                                         "apyrobo.registry.server": fake_server}):
            with patch("builtins.print"):
                _cmd_registry_dispatch(args)
        fake_uvicorn.run.assert_called_once()

    def test_registry_dispatch_unknown_exits_1(self, capsys):
        args = SimpleNamespace(registry_command=None)
        with pytest.raises(SystemExit) as ei:
            _cmd_registry_dispatch(args)
        assert ei.value.code == 1

    def test_skill_dispatch_search(self, capsys):
        args = SimpleNamespace(skill_command="search", query="test",
                               registry="http://localhost:8080")
        with patch("urllib.request.urlopen", _fake_urlopen([])):
            _cmd_skill_dispatch(args)
        out = capsys.readouterr().out
        assert "No skills found" in out

    def test_skill_dispatch_unknown_exits_1(self, capsys):
        args = SimpleNamespace(skill_command=None)
        with pytest.raises(SystemExit) as ei:
            _cmd_skill_dispatch(args)
        assert ei.value.code == 1
