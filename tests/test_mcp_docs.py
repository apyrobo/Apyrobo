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
