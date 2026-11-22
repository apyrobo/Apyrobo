"""docs/mcp.md 'Write a Nav2 keepout mask' says what ``apyrobo mcp keepout`` does.

The section promises a default resolution of 0.05 m per pixel, a mask covering
the zones plus 1 m, allowed areas left out of the mask, and a re-run that
replaces the earlier pair. Each promise is checked here against the command a
reader runs, so the section and the tool cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _section() -> str:
    text = (ROOT / "docs" / "mcp.md").read_text(encoding="utf-8")
    start = text.index("### Write a Nav2 keepout mask")
    end = text.index("\n## ", start)
    return " ".join(text[start:end].split())


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "apyrobo", "mcp", "keepout", *args],
        capture_output=True, text=True, timeout=60,
    )


def _mask(out: Path) -> tuple[dict, int, int, bytes]:
    meta = yaml.safe_load((out / "keepout_mask.yaml").read_text())
    magic, size, maxval, pixels = (out / meta["image"]).read_bytes().split(b"\n", 3)
    assert (magic, maxval) == (b"P5", b"255")
    width, height = (int(n) for n in size.split())
    assert len(pixels) == width * height
    return meta, width, height, pixels


def _black(meta: dict, width: int, height: int, pixels: bytes) -> tuple[float, ...]:
    """The black cells' bounding box in metres, read the way Nav2 places the mask."""
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    cells = [
        (col, height - 1 - row)
        for row in range(height)
        for col in range(width)
        if pixels[row * width + col] == 0
    ]
    cols = [c for c, _ in cells]
    rows = [r for _, r in cells]
    return (
        round(ox + min(cols) * res, 6),
        round(oy + min(rows) * res, 6),
        round(ox + (max(cols) + 1) * res, 6),
        round(oy + (max(rows) + 1) * res, 6),
    )


def test_default_resolution_is_the_one_the_doc_names(tmp_path):
    assert "--resolution` (metres per pixel, default 0.05)" in _section()
    result = _run("--no-go", "2,2,4,4", "--out", str(tmp_path))
    assert result.returncode == 0, result.stderr
    meta, *_ = _mask(tmp_path)
    assert meta["resolution"] == 0.05


def test_the_mask_covers_the_zones_plus_one_metre(tmp_path):
    assert "covering the zones plus 1 m" in _section()
    result = _run("--no-go", "2,2,4,4", "--no-go", "6,-1,7,0", "--out", str(tmp_path))
    assert result.returncode == 0, result.stderr
    meta, width, height, _ = _mask(tmp_path)
    # Zones span x 2..7 and y -1..4, so the mask spans x 1..8 and y -2..5.
    assert meta["origin"][:2] == [1.0, -2.0]
    assert (width * meta["resolution"], height * meta["resolution"]) == (7.0, 7.0)


def test_allowed_areas_in_a_policy_file_are_not_drawn(tmp_path):
    assert "Allowed areas are not drawn." in _section()
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "name": "lab",
        "no_go_zones": [{"name": "bench", "x_min": 2, "y_min": 2, "x_max": 4, "y_max": 4}],
        "allowed_areas": [{"name": "floor", "x_min": 0, "y_min": 0, "x_max": 10, "y_max": 6}],
    }))
    out = tmp_path / "masks"
    result = _run("--policy-file", str(policy), "--out", str(out))
    assert result.returncode == 0, result.stderr
    meta, width, height, pixels = _mask(out)
    # Only the no-go zone is black; the allowed area neither shows nor widens the mask.
    assert _black(meta, width, height, pixels) == (2.0, 2.0, 4.0, 4.0)
    assert meta["origin"][:2] == [1.0, 1.0]


def test_a_second_run_replaces_the_earlier_pair(tmp_path):
    assert "replacing any earlier pair" in _section()
    first = _run("--no-go", "0,0,5,5", "--out", str(tmp_path))
    assert first.returncode == 0, first.stderr
    second = _run("--no-go", "10,10,11,11", "--out", str(tmp_path))
    assert second.returncode == 0, second.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keepout_mask.pgm", "keepout_mask.yaml"]
    meta, width, height, pixels = _mask(tmp_path)
    assert _black(meta, width, height, pixels) == (10.0, 10.0, 11.0, 11.0)
