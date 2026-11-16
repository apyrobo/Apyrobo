"""The GitHub issue forms parse and ask for the facts a fix needs."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

TEMPLATES = Path(__file__).resolve().parent.parent / ".github" / "ISSUE_TEMPLATE"


def _load(name: str) -> dict:
    return yaml.safe_load((TEMPLATES / name).read_text(encoding="utf-8"))


def _fields(form: dict) -> dict[str, dict]:
    return {item["id"]: item for item in form["body"] if "id" in item}


@pytest.mark.parametrize("name", ["bug_report.yml", "feature_request.yml", "rfc.yml"])
def test_form_is_valid_issue_form(name: str) -> None:
    form = _load(name)
    for key in ("name", "description", "body"):
        assert form.get(key), f"{name} lacks {key}"
    ids = [item["id"] for item in form["body"] if "id" in item]
    assert len(ids) == len(set(ids)), f"{name} repeats a field id"
    for item in form["body"]:
        assert item["type"] in {"markdown", "textarea", "input", "dropdown", "checkboxes"}
        assert item["attributes"]


def test_bug_report_requires_what_a_fix_needs() -> None:
    fields = _fields(_load("bug_report.yml"))
    for fid in ("version", "os", "robot-uri", "command", "actual", "expected"):
        assert fields[fid]["validations"]["required"] is True, fid
    assert "mcp-client" in fields
    assert "__version__" in fields["version"]["attributes"]["description"]


def test_feature_request_asks_what_and_which_robot() -> None:
    fields = _fields(_load("feature_request.yml"))
    for fid in ("building", "robot", "need"):
        assert fields[fid]["validations"]["required"] is True, fid


def test_config_keeps_blank_issues_and_links_help() -> None:
    config = _load("config.yml")
    assert config["blank_issues_enabled"] is True
    urls = [link["url"] for link in config["contact_links"]]
    assert any(u.endswith("docs/mcp.md#troubleshooting") for u in urls)
    assert any("discord" in u for u in urls)
    for link in config["contact_links"]:
        assert link["name"] and link["about"]


def test_troubleshooting_anchor_exists() -> None:
    mcp_doc = TEMPLATES.parent.parent / "docs" / "mcp.md"
    assert "\n## Troubleshooting\n" in mcp_doc.read_text(encoding="utf-8")
