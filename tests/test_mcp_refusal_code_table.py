"""docs/mcp.md 'Refusal codes' lists every code a refusal can carry.

The codes are read from the safety enforcer's source: every violation dict it
records (``violation = {"type": ...}``) or would record (``move_violation``),
plus ``safety_violation``, the code apyrobo mcp uses when none was recorded.
A new violation type without a row in the table fails here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "mcp.md"
ENFORCER = ROOT / "apyrobo" / "safety" / "enforcer.py"

FALLBACK = "safety_violation"


def _dict_type(node: ast.AST) -> tuple[str, set[str]] | None:
    """('code', keys) for a dict literal with a string 'type' key, else None."""
    if not isinstance(node, ast.Dict):
        return None
    keys: set[str] = set()
    code = None
    for key, value in zip(node.keys, node.values, strict=True):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.add(key.value)
            if key.value == "type" and isinstance(value, ast.Constant):
                code = value.value
    if not isinstance(code, str):
        return None
    return code, keys - {"type", "message"}


def _enforcer_codes() -> dict[str, set[str]]:
    """Each violation type the enforcer records, with the fields it can carry."""
    tree = ast.parse(ENFORCER.read_text(encoding="utf-8"))
    codes: dict[str, set[str]] = {}
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        found: list[tuple[str, set[str]]] = []
        updated: set[str] = set()
        for node in ast.walk(func):
            target = None
            value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            if isinstance(target, ast.Name) and target.id == "violation":
                hit = _dict_type(value)
                if hit:
                    found.append(hit)
            if func.name == "move_violation" and isinstance(node, ast.Return):
                hit = _dict_type(node.value)
                if hit:
                    found.append(hit)
            # violation.update(start=..., requested_position=...)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "violation"
            ):
                updated |= {kw.arg for kw in node.keywords if kw.arg}
        for code, keys in found:
            codes.setdefault(code, set()).update(keys | updated)
    return codes


def _doc_rows() -> dict[str, tuple[str, set[str], str]]:
    """code -> (what tripped it, fields named, next step) from the doc table."""
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## Refusal codes", 1)[1].split("\n## ", 1)[0]
    rows: dict[str, tuple[str, set[str], str]] = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not cells[0].startswith("`"):
            continue
        code = cells[0].strip("`")
        if code == "code":
            continue
        fields = set(re.findall(r"`([a-z_]+)`", cells[2]))
        rows[code] = (cells[1], fields, cells[3])
    return rows


def test_the_enforcer_records_the_codes_we_expect_to_find():
    # Guards the parser above: if it stops finding codes, the main test
    # below would pass on an empty set.
    codes = _enforcer_codes()
    for code in ("collision_zone", "outside_allowed_area", "battery_below_minimum",
                 "collision_zone_path", "allowed_area_path", "invalid_command"):
        assert code in codes
    for not_a_refusal in ("speed_clamped", "angular_speed_clamped"):
        assert not_a_refusal not in codes


def test_the_table_lists_exactly_the_enforcer_codes_plus_the_fallback():
    expected = set(_enforcer_codes()) | {FALLBACK}
    assert set(_doc_rows()) == expected


def test_the_fallback_code_is_the_one_the_server_uses():
    from apyrobo.mcp_server import _refusal_content

    assert _refusal_content("default", None)["code"] == FALLBACK
    assert _refusal_content("default", {"message": "x"})["code"] == FALLBACK


@pytest.mark.parametrize("code", sorted(_enforcer_codes()))
def test_each_row_names_only_fields_the_violation_carries(code):
    _, fields, _ = _doc_rows()[code]
    assert fields, f"{code}: the row names no field of violation"
    assert fields <= _enforcer_codes()[code], (
        f"{code}: the row names {sorted(fields - _enforcer_codes()[code])}, "
        "which the enforcer does not record"
    )


def test_every_row_says_what_tripped_it_and_what_to_do():
    for code, (tripped, _, next_step) in _doc_rows().items():
        assert tripped, code
        assert next_step, code


def test_the_code_lists_elsewhere_in_the_doc_link_to_the_table():
    text = DOC.read_text(encoding="utf-8")
    assert text.count("[Refusal codes](#refusal-codes)") >= 2
    # The old open-ended lists of codes are gone.
    assert "`outside_allowed_area`, ...)" not in text
    assert "`battery_below_minimum`, …," not in text
