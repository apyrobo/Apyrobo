"""SkillRegistryClient reads what `apyrobo registry start` actually serves.

The server answers ``/search`` and ``/skills`` with a bare JSON list, returns
``{"status": "published"}`` on a publish, and checks a bearer token. The client
used to expect ``{"results": [...]}``, ``{"skills": [...]}`` and
``{"status": "ok"}`` and sent the token only in the body, so ``apyrobo registry
search`` against a running local registry said "No packages found" and
``registry publish`` never reported success.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import pytest

from apyrobo.registry import SkillPackage, SkillRegistryClient
from apyrobo.registry.client import load_seed_index


def _pkg_dict(name: str = "apyrobo-skills-demo", version: str = "0.2.0") -> dict:
    return {
        "name": name,
        "version": version,
        "description": "Demo navigation skills",
        "author": "Tester",
        "license": "Apache-2.0",
        "download_url": "https://example.com/demo-0.2.0.whl",
        "checksum": "a" * 64,
        "apyrobo_version_min": "4.0.0",
        "tags": ["navigation"],
    }


# ---------------------------------------------------------------------------
# A stdlib stand-in that answers with the server's documented shapes
# (response_model=list[SkillPackage] for /search and /skills).
# ---------------------------------------------------------------------------


class _ListShapeHandler(BaseHTTPRequestHandler):
    seen_auth: list[str] = []

    def log_message(self, *args):  # noqa: D401 - keep test output quiet
        pass

    def _send(self, code: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/search", "/skills"):
            self._send(200, [_pkg_dict()])
        elif path == "/skills/apyrobo-skills-demo":
            self._send(200, _pkg_dict())
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        type(self).seen_auth.append(self.headers.get("Authorization", ""))
        self._send(201, {"status": "published", "name": "x", "version": "1"})


@pytest.fixture
def list_shape_server():
    _ListShapeHandler.seen_auth = []
    srv = HTTPServer(("127.0.0.1", 0), _ListShapeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


class TestListShapes:
    def test_search_reads_bare_list(self, list_shape_server):
        results = SkillRegistryClient(list_shape_server).search("nav")
        assert [p.name for p in results] == ["apyrobo-skills-demo"]

    def test_list_all_reads_bare_list(self, list_shape_server):
        results = SkillRegistryClient(list_shape_server).list_all()
        assert [p.name for p in results] == ["apyrobo-skills-demo"]

    def test_get_exact_version_of_latest(self, list_shape_server):
        pkg = SkillRegistryClient(list_shape_server).get("apyrobo-skills-demo", "0.2.0")
        assert pkg is not None and pkg.version == "0.2.0"

    def test_get_unknown_version_is_none(self, list_shape_server):
        assert SkillRegistryClient(list_shape_server).get("apyrobo-skills-demo", "9.9.9") is None

    def test_publish_published_status_is_success_and_sends_bearer(self, list_shape_server):
        ok = SkillRegistryClient(list_shape_server).publish(
            SkillPackage(**_pkg_dict()), token="s3cret"
        )
        assert ok is True
        assert _ListShapeHandler.seen_auth == ["Bearer s3cret"]

    def test_wrapped_shapes_still_accepted(self, monkeypatch):
        client = SkillRegistryClient("http://registry.invalid")
        wrapped = {"results": [_pkg_dict()], "skills": [_pkg_dict()]}
        monkeypatch.setattr(client, "_get", lambda path: wrapped)
        assert [p.name for p in client.search("x")] == ["apyrobo-skills-demo"]
        assert [p.name for p in client.list_all()] == ["apyrobo-skills-demo"]


# ---------------------------------------------------------------------------
# The real registry app, served by uvicorn (skipped without the extra).
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def real_registry(monkeypatch):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from apyrobo.registry import server

    monkeypatch.setattr(server, "_store", {})
    monkeypatch.setattr(server, "_DEFAULT_AUTH_TOKEN", None)
    app = server.create_app(auth_token="tok-123", seed=True)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 10
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert srv.started, "registry did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        t.join(timeout=5)


class TestRealRegistry:
    def test_search_finds_seeded_package(self, real_registry, capsys):
        seed_names = {p.name for p in load_seed_index()}
        results = SkillRegistryClient(real_registry).search("nav")
        assert results, "search against a seeded registry returned nothing"
        assert {p.name for p in results} <= seed_names
        assert "unreachable" not in capsys.readouterr().err

    def test_list_all_returns_seed(self, real_registry):
        names = {p.name for p in SkillRegistryClient(real_registry).list_all()}
        assert names == {p.name for p in load_seed_index()}

    def test_publish_then_search(self, real_registry):
        client = SkillRegistryClient(real_registry)
        assert client.publish(SkillPackage(**_pkg_dict()), token="tok-123") is True
        assert "apyrobo-skills-demo" in {p.name for p in client.search("demo")}

    def test_publish_wrong_token_raises(self, real_registry):
        import urllib.error

        with pytest.raises(urllib.error.HTTPError):
            SkillRegistryClient(real_registry).publish(
                SkillPackage(**_pkg_dict(version="0.3.0")), token="wrong"
            )

    def test_cli_registry_search_prints_package(self, real_registry, capsys, monkeypatch):
        import sys

        from apyrobo.cli import main

        monkeypatch.setattr(
            sys, "argv", ["apyrobo", "registry", "search", "nav", "--registry-url", real_registry]
        )
        main()
        out = capsys.readouterr().out
        assert "No packages found" not in out
        assert any(p.name in out for p in load_seed_index())
