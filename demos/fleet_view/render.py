"""Headless renderer for the fleet view — produces demo.gif / demo.mp4.

Reuses the *real* sim and planner from ``server.py`` (same SimRobot motion,
same FleetViewServer planning), scripts a short task sequence, draws each
frame with Pillow to match the browser canvas, and assembles a GIF + MP4
with ffmpeg. This is what ``record.sh`` runs, so the video re-renders after
any change to the demo — no browser or screen recorder needed.

    pip install -e '.[websocket]' Pillow      # ffmpeg on PATH for the mp4
    python demos/fleet_view/render.py
"""
from __future__ import annotations

import importlib.util
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

# Load the demo's server module (its motion + planning are the real thing).
_spec = importlib.util.spec_from_file_location("fleet_view_server", HERE / "server.py")
fv = importlib.util.module_from_spec(_spec)
sys.modules["fleet_view_server"] = fv
_spec.loader.exec_module(fv)

from apyrobo import Agent  # noqa: E402
from apyrobo.orchestration.adapter import OrchestrationMessage  # noqa: E402

# ---------------------------------------------------------------------------
# Layout / theme (mirrors index.html)
# ---------------------------------------------------------------------------
SCALE = 2  # supersample, downscaled on save for crisp text
W, H = 940 * SCALE, 600 * SCALE
PANEL_W = 300 * SCALE
HEADER_H = 34 * SCALE
WORLD_W = W - PANEL_W
FPS = 12
DT = 1.0 / FPS

BG = (20, 20, 31)
WORLD_BG = (24, 24, 38)
GRID = (36, 36, 56)
PANEL_BG = (28, 28, 43)
LINE = (44, 44, 64)
TEXT = (216, 216, 232)
DIM = (138, 138, 163)
DRONE = (122, 162, 247)
GROUND = (158, 206, 106)
OKC = (158, 206, 106)
ERRC = (247, 118, 142)
SENTC = (122, 162, 247)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "/System/Library/Fonts/Menlo.ttc"
    try:
        return ImageFont.truetype(path, size, index=1 if bold else 0)
    except Exception:
        return ImageFont.load_default()


F_SM = _font(11 * SCALE)
F_MD = _font(13 * SCALE)
F_HDR = _font(14 * SCALE, bold=True)
F_ZONE = _font(10 * SCALE)


class EventLog:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, tuple[int, int, int]]] = []

    def add(self, tag: str, text: str, color: tuple[int, int, int]) -> None:
        self.rows.insert(0, (tag, text, color))
        del self.rows[26:]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def world_to_px(x: float, y: float) -> tuple[float, float]:
    ww, wh = fv.WORLD
    scale = min((WORLD_W - 40 * SCALE) / ww, (H - HEADER_H - 40 * SCALE) / wh)
    ox = (WORLD_W - ww * scale) / 2
    oy = HEADER_H + (H - HEADER_H - wh * scale) / 2
    return ox + x * scale, oy + y * scale, scale  # type: ignore[return-value]


def draw_frame(fleet, log: EventLog) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    ww, wh = fv.WORLD

    # header
    d.text((14 * SCALE, 9 * SCALE),
           "APYROBO · Live Fleet View", font=F_HDR, fill=TEXT)
    d.text((250 * SCALE, 12 * SCALE),
           "tasks planned & dispatched over the wire protocol", font=F_SM, fill=DIM)
    d.line([(0, HEADER_H), (W, HEADER_H)], fill=LINE, width=1)

    # world background + grid
    ox0, oy0, scale = world_to_px(0, 0)
    d.rectangle([ox0, oy0, ox0 + ww * scale, oy0 + wh * scale], fill=WORLD_BG)
    for gx in range(0, int(ww) + 1, 10):
        px, _, _ = world_to_px(gx, 0)
        d.line([(px, oy0), (px, oy0 + wh * scale)], fill=GRID, width=1)
    for gy in range(0, int(wh) + 1, 10):
        _, py, _ = world_to_px(0, gy)
        d.line([(ox0, py), (ox0 + ww * scale, py)], fill=GRID, width=1)

    # zones
    for name, (zx, zy) in fv.ZONES.items():
        px, py, _ = world_to_px(zx, zy)
        r = 3.2 * scale
        for a in range(0, 360, 18):  # dashed ring
            a2 = a + 9
            d.arc([px - r, py - r, px + r, py + r], a, a2, fill=(58, 58, 85), width=1)
        d.text((px, py + 4.6 * scale), name.upper(), font=F_ZONE,
               fill=(90, 90, 122), anchor="mm")

    # robots
    for sim in fleet:
        color = DRONE if sim.kind == "drone" else GROUND
        px, py, _ = world_to_px(sim.x, sim.y)
        if sim.target:
            tx, ty, _ = world_to_px(*sim.target)
            d.line([(px, py), (tx, ty)], fill=color + (0,), width=1)
            _dashed(d, px, py, tx, ty, color)
            d.ellipse([tx - 4 * SCALE, ty - 4 * SCALE, tx + 4 * SCALE, ty + 4 * SCALE],
                      outline=color, width=SCALE)
        _draw_body(d, px, py, sim.heading, sim.kind, color)
        d.text((px, py - 12 * SCALE), sim.robot_id, font=F_SM, fill=(200, 200, 221),
               anchor="mm")
        if sim.task:
            d.text((px, py + 15 * SCALE), sim.task[:30], font=F_SM,
                   fill=(106, 106, 136), anchor="mm")

    # panel
    d.rectangle([W - PANEL_W, HEADER_H, W, H], fill=PANEL_BG)
    d.line([(W - PANEL_W, HEADER_H), (W - PANEL_W, H)], fill=LINE, width=1)
    x0 = W - PANEL_W + 12 * SCALE
    y = HEADER_H + 12 * SCALE
    d.text((x0, y), "EVENT STREAM", font=F_SM, fill=DIM)
    y += 20 * SCALE
    for tag, text, color in log.rows:
        d.text((x0, y), tag, font=F_SM, fill=color)
        d.text((x0 + 58 * SCALE, y), text[:30], font=F_SM, fill=TEXT)
        y += 17 * SCALE
        if y > H - 16 * SCALE:
            break
    return img


def _dashed(d, x0, y0, x1, y1, color):
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    seg, gap, pos = 4 * SCALE, 5 * SCALE, 0.0
    while pos < length:
        a = pos
        b = min(pos + seg, length)
        d.line([(x0 + ux * a, y0 + uy * a), (x0 + ux * b, y0 + uy * b)],
               fill=tuple(int(c * 0.55) for c in color), width=1)
        pos += seg + gap


def _draw_body(d, px, py, heading, kind, color):
    s = 6 * SCALE
    if kind == "drone":
        pts = [(1.6 * s, 0), (-s, 0.9 * s), (-s, -0.9 * s)]
    else:
        pts = [(-s, -0.8 * s), (s, -0.8 * s), (s, 0.8 * s), (-s, 0.8 * s)]
    ca, sa = math.cos(heading), math.sin(heading)
    rot = [(px + x * ca - y * sa, py + x * sa + y * ca) for x, y in pts]
    d.polygon(rot, fill=color)


# ---------------------------------------------------------------------------
# Scripted run
# ---------------------------------------------------------------------------

def main() -> None:
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found (brew install ffmpeg)", file=sys.stderr)
        sys.exit(1)

    fleet = fv.build_fleet()

    class _NullAdapter:
        def receive(self):
            return None

        def send(self, msg):
            pass

        def startup(self):
            pass

        def shutdown(self):
            pass

    server = fv.FleetViewServer(_NullAdapter(), Agent(provider="rule"), fleet)
    log = EventLog()

    def dispatch(task: str, robot_id: str) -> None:
        log.add("SENT", f"“{task}”", SENTC)
        resp = server._handle(OrchestrationMessage(
            task=task, robot_uri=f"mock://{robot_id}"))
        meta = resp.metadata
        if meta.get("status") == "planned":
            skills = " → ".join(s["name"] for s in meta.get("skills", []))
            log.add("PLANNED", skills or "(no steps)", OKC)
        else:
            log.add("ERROR", meta.get("error", ""), ERRC)

    # (frame_at, task, robot) — a tour that shows multi-step plans + zones.
    script = [
        (6, "deliver a package to the dock", "hauler_bot"),
        (20, "survey the warehouse shelves", "drone_00"),
        (32, "carry parts to the lab", "picker_bot"),
        (44, "go to (20, 40)", "drone_01"),
        (56, "inspect the charging station", "scout_bot"),
        (70, "patrol the perimeter", "drone_02"),
        (84, "deliver a package to the dock", "picker_bot"),
        (96, "scan the kitchen area", "hauler_bot"),
    ]
    total_frames = 150
    auto_rng = __import__("random").Random(5)

    tmp = Path(tempfile.mkdtemp(prefix="fleetgif_"))
    try:
        pending = list(script)
        for f in range(total_frames):
            while pending and pending[0][0] == f:
                _, task, rid = pending.pop(0)
                dispatch(task, rid)
            # keep idle robots busy, like the real auto-dispatcher
            if f > 4 and f % 9 == 0:
                idle = [s for s in fleet if s.state == "idle"]
                if idle:
                    sim = auto_rng.choice(idle)
                    dispatch(auto_rng.choice(fv.PATROL_TASKS), sim.robot_id)
            for sim in fleet:
                sim.tick(DT)
            frame = draw_frame(fleet, log).resize((W // SCALE, H // SCALE),
                                                  Image.LANCZOS)
            frame.save(tmp / f"f{f:04d}.png")

        gif = HERE / "demo.gif"
        mp4 = HERE / "demo.mp4"
        pattern = str(tmp / "f%04d.png")
        palette = tmp / "palette.png"
        # Downscale the GIF (keeps the repo artifact small); MP4 stays full-res.
        gif_scale = "scale=760:-1:flags=lanczos"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", pattern,
             "-vf", f"{gif_scale},palettegen=stats_mode=diff", str(palette)],
            check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", pattern,
             "-i", str(palette), "-lavfi",
             f"{gif_scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
             str(gif)], check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", pattern,
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
        print(f"wrote {gif} and {mp4}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
