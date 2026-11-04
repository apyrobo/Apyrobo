#!/usr/bin/env python3
"""
Build the animated terminal SVG in the README from the real MCP safety demo.

This runs examples/05_mcp_safety_demo.py, which starts the real ``apyrobo mcp``
server on mock://, and draws what the server answered. Nothing is typed in by
hand, so the picture cannot say more than the code does; tests/test_safety_demo_svg.py
fails if the committed file no longer matches a fresh run.

Run from the repo root:
    python scripts/make_safety_demo_svg.py          # rewrite docs/mcp_safety_demo.svg
    python scripts/make_safety_demo_svg.py --check  # exit 1 if the file is stale
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "05_mcp_safety_demo.py"
OUTPUT = ROOT / "docs" / "mcp_safety_demo.svg"

COLUMNS = 84
CHAR_WIDTH = 8.4
LINE_HEIGHT = 21
PAD_X = 18
TITLE_BAR = 34
PAD_Y = 16
FIRST_REVEAL_S = 0.6
STEP_S = 1.6
HOLD_S = 6.0

Line = tuple[str, str]


def load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mcp_safety_demo_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _at(position: dict[str, float]) -> str:
    return f"({position['x']}, {position['y']})"


def _outcome(call: dict[str, Any]) -> list[Line]:
    """What the server said about one call, in the words of its reply."""
    if call["is_error"]:
        wrapped = textwrap.wrap(call["text"], COLUMNS - 2, break_long_words=False)
        return [("refused", f"  {part}") for part in wrapped]
    reply = json.loads(call["text"])
    summary = f"  {reply['status']}, robot at {_at(reply['position'])}"
    clamps = [i for i in reply.get("safety_interventions", []) if i["type"] == "speed_clamped"]
    if not clamps:
        return [("ran", summary)]
    clamp = clamps[0]
    return [
        ("ran", summary),
        ("clamped", f"  speed_clamped: requested {clamp['requested']} m/s, "
                    f"enforced {clamp['enforced']} m/s"),
    ]


def transcript_lines(session: dict[str, Any], command: str) -> list[list[Line]]:
    """Group the demo into blocks that appear together: a call, then its answer."""
    blocks: list[list[Line]] = [[("prompt", f"$ {command}")]]
    for call in session["calls"]:
        if call["tool"] not in ("move_to", "stop"):
            continue
        arguments = json.dumps(call["arguments"]) if call["arguments"] else ""
        blocks.append([("call", f"> {call['tool']} {arguments}".rstrip()), *_outcome(call)])
    return blocks


def render_svg(blocks: list[list[Line]]) -> str:
    rows = sum(len(block) for block in blocks) + len(blocks) - 1
    width = round(PAD_X * 2 + COLUMNS * CHAR_WIDTH)
    height = TITLE_BAR + PAD_Y * 2 + rows * LINE_HEIGHT
    total = FIRST_REVEAL_S + STEP_S * (len(blocks) - 1) + HOLD_S

    keyframes = []
    texts = []
    y = TITLE_BAR + PAD_Y + LINE_HEIGHT - 5
    for index, block in enumerate(blocks):
        start = (FIRST_REVEAL_S + STEP_S * index) / total * 100
        keyframes.append(
            f"@keyframes b{index}{{0%,{start:.2f}%{{opacity:0}}"
            f"{start + 0.5:.2f}%,97%{{opacity:1}}100%{{opacity:0}}}}"
            f".b{index}{{animation:b{index} {total:.1f}s linear infinite}}"
        )
        texts.append(f'<g class="b{index}">')
        for kind, text in block:
            texts.append(f'<text x="{PAD_X}" y="{y}" class="{kind}">{escape(text)}</text>')
            y += LINE_HEIGHT
        texts.append("</g>")
        y += LINE_HEIGHT

    style = "".join([
        "text{font:14px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre}",
        ".prompt{fill:#e6edf3}.call{fill:#79c0ff}.ran{fill:#3fb950}",
        ".clamped{fill:#d29922}.refused{fill:#ff7b72}.title{fill:#8b949e;font-size:12px}",
        *keyframes,
        "@media (prefers-reduced-motion:reduce){g{animation:none!important}}",
    ])
    title = "apyrobo mcp on mock://: a move runs, an over-limit move is clamped, " \
            "a no-go move is refused"
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        f"<style>{style}</style>",
        f'<rect width="{width}" height="{height}" rx="8" fill="#0d1117"/>',
        '<circle cx="20" cy="17" r="6" fill="#ff5f56"/>',
        '<circle cx="40" cy="17" r="6" fill="#ffbd2e"/>',
        '<circle cx="60" cy="17" r="6" fill="#27c93f"/>',
        f'<text x="{width / 2}" y="21" text-anchor="middle" class="title">'
        "examples/05_mcp_safety_demo.py</text>",
        *texts,
        "</svg>",
        "",
    ])


def build(session: dict[str, Any] | None = None) -> str:
    example = load_example()
    if session is None:
        session = example.run_demo()
    command = " ".join(["apyrobo", *example.SERVER_CMD[3:]])
    return render_svg(transcript_lines(session, command))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    args = parser.parse_args(argv)
    svg = build()
    if args.check:
        if OUTPUT.read_text(encoding="utf-8") != svg:
            print(f"{OUTPUT} is stale; rerun python scripts/make_safety_demo_svg.py",
                  file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(svg.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
