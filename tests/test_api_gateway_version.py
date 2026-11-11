"""The REST gateway reports the installed package version, not a stale literal."""

import ast
from pathlib import Path

import pytest

import apyrobo.api.app as app_module
from apyrobo.__version__ import __version__


def test_fastapi_version_argument_is_package_version():
    # Runs without fastapi installed: the version passed to FastAPI(...) must be
    # the package's __version__ name, never a hard-coded string.
    tree = ast.parse(Path(app_module.__file__).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "FastAPI"
    ]
    assert calls, "create_app no longer builds a FastAPI app"
    for call in calls:
        version = {kw.arg: kw.value for kw in call.keywords}.get("version")
        assert isinstance(version, ast.Name) and version.id == "__version__"
    assert app_module.__version__ == __version__


def test_openapi_reports_package_version():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    app = app_module.create_app(api_key="k")
    assert app.version == __version__
    schema = TestClient(app).get("/openapi.json").json()
    assert schema["info"]["version"] == __version__
