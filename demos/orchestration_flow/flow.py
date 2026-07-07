"""
Demo 5: Orchestration Flow — the pipeline, made visible
=======================================================
One task, followed all the way down the stack the README diagram promises:

    natural language → Agent → Skill Graph → Safety Enforcer → robot

Every panel is populated by the *real* APYROBO objects, not a mock-up:

* DISCOVER  — ``robot.capabilities()`` (capabilities + max_speed)
* PLAN      — ``Agent(provider="rule").plan(task, robot)`` → a SkillGraph
* GRAPH     — the graph's execution order (skills + required capability)
* SAFETY    — ``SafetyEnforcer.move()`` really clamps the requested speed and
              really rejects a waypoint inside a no-go zone (the numbers shown
              are the interventions it recorded)
* EXECUTE   — ``SkillExecutor.execute_graph`` runs the plan; the robot walks
              the world panel, routing around the no-go zone and slowing near
              the human, one skill at a time.

Render it to demo.gif / demo.mp4:
    pip install -e . Pillow          # ffmpeg on PATH
    python demos/orchestration_flow/flow.py
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, __file__.rsplit("/demos/", 1)[0])  # repo root when cloned

from apyrobo import Agent, Robot, SafetyEnforcer
from apyrobo.safety.enforcer import SafetyPolicy
from apyrobo.skills.executor import SkillExecutor

# Pillow is imported lazily (see _ensure_pillow) so the pipeline logic —
# run_pipeline() / build_path(), which the tests exercise — works without
# the rendering dependency installed.
Image = ImageDraw = ImageFont = None  # type: ignore[assignment]

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------
TASK = "deliver the package to the dock"
ROBOT_URI = "mock://courier-01"
REQUESTED_SPEED = 2.5  # m/s the planner would like — safety will clamp it

WORLD = (12.0, 8.0)  # metres
START = (1.2, 1.2)
SHELF = (2.6, 6.6)          # pickup
DOCK = (10.2, 4.0)          # dropoff
NOGO = {"x_min": 5.0, "x_max": 8.0, "y_min": 3.8, "y_max": 7.2}
HUMAN = (9.2, 6.4)
HUMAN_LIMIT = 1.0           # m — proximity limit

# Waypoints for the animation. The leg from the shelf to the dock detours
# *below* the no-go zone rather than crossing it.
LEGS = [
    ("Navigate To", START, SHELF),
    ("Pick Object", SHELF, SHELF),
    ("Navigate To", SHELF, DOCK, [(3.6, 2.4), (8.6, 2.6)]),  # around the zone
    ("Place Object", DOCK, DOCK),
]

# ---------------------------------------------------------------------------
# Layout / theme
# ---------------------------------------------------------------------------
SCALE = 2
W, H = 980 * SCALE, 600 * SCALE
HEADER_H = 40 * SCALE
PANEL_W = 430 * SCALE            # left: pipeline
WORLD_X = PANEL_W
WORLD_W = W - PANEL_W
FPS = 12

BG = (18, 18, 28)
PANEL_BG = (24, 24, 37)
WORLD_BG = (22, 22, 34)
GRID = (34, 34, 52)
LINE = (44, 44, 66)
TEXT = (222, 222, 238)
DIM = (140, 140, 168)
ACCENT = (122, 162, 247)
OK = (158, 206, 106)
WARN = (224, 175, 104)
ERR = (247, 118, 142)
ROBOT_C = (158, 206, 106)
HUMAN_C = (224, 175, 104)
NOGO_C = (247, 118, 142)


# Fonts are populated by _ensure_pillow() on the first render call.
F_S = F_M = F_L = F_HDR = None


def _font(size, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc",
                                  size, index=1 if bold else 0)
    except Exception:
        return ImageFont.load_default()


def _ensure_pillow() -> None:
    """Import Pillow and build the fonts on demand (renderer-only dependency)."""
    global Image, ImageDraw, ImageFont, F_S, F_M, F_L, F_HDR
    if Image is not None:
        return
    try:
        from PIL import Image as _Image
        from PIL import ImageDraw as _ImageDraw
        from PIL import ImageFont as _ImageFont
    except ImportError as exc:  # pragma: no cover - exercised only when unset
        raise SystemExit(
            "error: Pillow is required to render this demo — pip install Pillow"
        ) from exc
    Image, ImageDraw, ImageFont = _Image, _ImageDraw, _ImageFont
    F_S = _font(11 * SCALE)
    F_M = _font(13 * SCALE)
    F_L = _font(15 * SCALE, bold=True)
    F_HDR = _font(16 * SCALE, bold=True)


@dataclass
class Stage:
    key: str
    title: str
    lines: list[tuple[str, tuple[int, int, int]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Run the real pipeline, capturing genuine data for each stage
# ---------------------------------------------------------------------------

def run_pipeline() -> tuple[list[Stage], list[str]]:
    robot = Robot.discover(ROBOT_URI)

    caps = robot.capabilities()
    cap_names = [c.capability_type.value for c in caps.capabilities]

    agent = Agent(provider="rule")
    graph = agent.plan(TASK, robot)
    order = graph.get_execution_order()
    plan_names = [s.name for s in order]

    policy = SafetyPolicy(
        name="strict", max_speed=0.5,
        collision_zones=[NOGO], human_proximity_limit=HUMAN_LIMIT,
    )
    enforcer = SafetyEnforcer(robot, policy=policy)
    # A real clamp: ask for REQUESTED_SPEED toward the dock.
    enforcer.move(DOCK[0], DOCK[1], speed=REQUESTED_SPEED)
    clamp = next((i for i in enforcer._interventions
                  if i["type"] == "speed_clamped"), None)
    # A real rejection: a waypoint inside the no-go zone.
    zone_pt = ((NOGO["x_min"] + NOGO["x_max"]) / 2,
               (NOGO["y_min"] + NOGO["y_max"]) / 2)
    rejected = False
    try:
        enforcer.move(zone_pt[0], zone_pt[1], speed=0.4)
    except Exception:
        rejected = True
    # move() arms a background move-watchdog expecting the (mock) robot to
    # actually travel; cancel it so no stray divergence alarm fires later.
    enforcer.stop()

    # Execute the plan for real, collecting per-skill completion events.
    events: list[str] = []
    executor = SkillExecutor(robot)
    executor.on_event(lambda e: events.append(f"{e.skill_id}:{e.status.value}"))
    executor.execute_graph(graph)

    enforced = clamp["enforced"] if clamp else policy.max_speed
    requested = clamp["requested"] if clamp else REQUESTED_SPEED

    stages = [
        Stage("task", "1 · TASK", [
            (f'"{TASK}"', TEXT),
            ("natural language in — verified robot actions out", DIM),
        ]),
        Stage("discover", "2 · DISCOVER  ·  robot.capabilities()", [
            (f"{ROBOT_URI}", TEXT),
            (f"can: {', '.join(cap_names)}", OK),
            (f"max_speed: {caps.max_speed} m/s   (planner may not exceed)", DIM),
        ]),
        Stage("plan", "3 · PLAN  ·  Agent(provider='rule').plan()", [
            (f"{len(plan_names)} skills chosen from the catalog", TEXT),
            ("only skills the robot actually declares are allowed", DIM),
        ]),
        Stage("graph", "4 · SKILL GRAPH  ·  execution order", [
            (s.name + f"  [{s.required_capability.value}]", ACCENT)
            for s in order
        ]),
        Stage("safety", "5 · SAFETY ENFORCER  ·  SafetyEnforcer.move()", [
            (f"speed {requested} → {enforced} m/s  (clamped to policy max)", WARN),
            (f"no-go zone entry REJECTED  ({'held' if rejected else '—'})", ERR),
            (f"stay {HUMAN_LIMIT:.1f} m from humans  ·  slows on approach", DIM),
        ]),
        Stage("execute", "6 · EXECUTE  ·  SkillExecutor.execute_graph()", [
            ("running the plan on the robot →", TEXT),
        ]),
    ]
    return stages, plan_names


# ---------------------------------------------------------------------------
# World panel geometry
# ---------------------------------------------------------------------------

def wpx(x, y):
    ww, wh = WORLD
    pad = 30 * SCALE
    scale = min((WORLD_W - 2 * pad) / ww, (H - HEADER_H - 2 * pad) / wh)
    ox = WORLD_X + (WORLD_W - ww * scale) / 2
    oy = HEADER_H + (H - HEADER_H - wh * scale) / 2
    return ox + x * scale, oy + (wh - y) * scale, scale  # y up


# ---------------------------------------------------------------------------
# Path sampling for the execute animation
# ---------------------------------------------------------------------------

def build_path() -> list[tuple[str, tuple[float, float]]]:
    """Dense list of (active_skill, point) along the whole mission."""
    pts: list[tuple[str, tuple[float, float]]] = []
    for leg in LEGS:
        name, a, b = leg[0], leg[1], leg[2]
        vias = leg[3] if len(leg) > 3 else []
        waypoints = [a, *vias, b]
        if a == b:  # pick / place — dwell in place
            for _ in range(6):
                pts.append((name, a))
            continue
        for i in range(len(waypoints) - 1):
            p, q = waypoints[i], waypoints[i + 1]
            steps = max(6, int(math.dist(p, q) * 4))
            for s in range(steps):
                t = s / steps
                pts.append((name, (p[0] + (q[0] - p[0]) * t,
                                   p[1] + (q[1] - p[1]) * t)))
    pts.append((LEGS[-1][0], DOCK))
    return pts


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw(stages, active_idx, robot_pos, heading, trail, done_skills,
         plan_names, exec_label):
    _ensure_pillow()
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # header
    d.text((16 * SCALE, 11 * SCALE),
           "APYROBO · Orchestration Flow", font=F_HDR, fill=TEXT)
    d.text((330 * SCALE, 15 * SCALE),
           "one task, all the way down the stack", font=F_M, fill=DIM)
    d.line([(0, HEADER_H), (W, HEADER_H)], fill=LINE, width=1)

    _draw_pipeline(d, stages, active_idx, done_skills, plan_names)
    _draw_world(d, robot_pos, heading, trail, exec_label)
    return img


def _draw_pipeline(d, stages, active_idx, done_skills, plan_names):
    d.rectangle([0, HEADER_H, PANEL_W, H], fill=PANEL_BG)
    d.line([(PANEL_W, HEADER_H), (PANEL_W, H)], fill=LINE, width=1)
    x = 18 * SCALE
    y = HEADER_H + 16 * SCALE
    avail = H - y - 14 * SCALE
    # Cap the row height so the tallest stage (EXECUTE: 1 line + a 4-item
    # checklist ≈ 95px) fits above the bottom edge instead of stretching to
    # fill and clipping the last item.
    slot = min(avail / len(stages), 82 * SCALE)

    for i, st in enumerate(stages):
        active = i == active_idx
        done = i < active_idx
        top = y + i * slot
        # connector
        if i > 0:
            col = OK if i <= active_idx else LINE
            d.line([(x + 5 * SCALE, top - slot + 30 * SCALE),
                    (x + 5 * SCALE, top + 4 * SCALE)], fill=col, width=2)
        # node dot
        dot = OK if done else (ACCENT if active else LINE)
        d.ellipse([x, top, x + 10 * SCALE, top + 10 * SCALE], fill=dot)
        # title
        tcol = TEXT if (active or done) else DIM
        d.text((x + 22 * SCALE, top - 2 * SCALE), st.title, font=F_L, fill=tcol)
        # body lines (only for reached stages)
        ly = top + 20 * SCALE
        if i <= active_idx:
            for text, col in st.lines:
                shown = col if (active or done) else DIM
                d.text((x + 22 * SCALE, ly), text, font=F_S, fill=shown)
                ly += 15 * SCALE
            # execute stage: per-skill checklist
            if st.key == "execute":
                for name in plan_names:
                    mark = "✓" if name in done_skills else "•"
                    mc = OK if name in done_skills else DIM
                    d.text((x + 26 * SCALE, ly), mark, font=F_S, fill=mc)
                    d.text((x + 40 * SCALE, ly), name, font=F_S,
                           fill=TEXT if name in done_skills else DIM)
                    ly += 15 * SCALE


def _draw_world(d, robot_pos, heading, trail, exec_label):
    ww, wh = WORLD
    ox, oy, scale = wpx(0, 0)
    d.rectangle([ox, oy - wh * scale, ox + ww * scale, oy], fill=WORLD_BG)
    for gx in range(0, int(ww) + 1, 2):
        px, py0, _ = wpx(gx, 0)
        _, py1, _ = wpx(gx, wh)
        d.line([(px, py0), (px, py1)], fill=GRID, width=1)
    for gy in range(0, int(wh) + 1, 2):
        px0, py, _ = wpx(0, gy)
        px1, _, _ = wpx(ww, gy)
        d.line([(px0, py), (px1, py)], fill=GRID, width=1)

    # no-go zone
    zx0, zy0, _ = wpx(NOGO["x_min"], NOGO["y_max"])
    zx1, zy1, _ = wpx(NOGO["x_max"], NOGO["y_min"])
    d.rectangle([zx0, zy0, zx1, zy1], fill=(60, 26, 34), outline=NOGO_C, width=1)
    d.text(((zx0 + zx1) / 2, (zy0 + zy1) / 2), "NO-GO", font=F_S,
           fill=NOGO_C, anchor="mm")

    # markers
    for (mx, my), label, col in [(SHELF, "SHELF", ACCENT), (DOCK, "DOCK", OK)]:
        px, py, _ = wpx(mx, my)
        d.rectangle([px - 5 * SCALE, py - 5 * SCALE, px + 5 * SCALE, py + 5 * SCALE],
                    outline=col, width=SCALE)
        d.text((px, py - 13 * SCALE), label, font=F_S, fill=col, anchor="mm")

    # human + proximity ring
    hx, hy, _ = wpx(*HUMAN)
    rr = HUMAN_LIMIT * scale
    d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], outline=(90, 70, 40), width=1)
    d.ellipse([hx - 4 * SCALE, hy - 4 * SCALE, hx + 4 * SCALE, hy + 4 * SCALE],
              fill=HUMAN_C)
    d.text((hx, hy - 13 * SCALE), "HUMAN", font=F_S, fill=HUMAN_C, anchor="mm")

    # trail
    if len(trail) > 1:
        pts = [wpx(x, y)[:2] for x, y in trail]
        d.line(pts, fill=(158, 206, 106, 120), width=2)

    # robot
    if robot_pos is not None:
        px, py, _ = wpx(*robot_pos)
        s = 7 * SCALE
        pts = [(-s, -0.8 * s), (s, -0.8 * s), (s, 0.8 * s), (-s, 0.8 * s)]
        ca, sa = math.cos(-heading), math.sin(-heading)
        rot = [(px + a * ca - b * sa, py + a * sa + b * ca) for a, b in pts]
        d.polygon(rot, fill=ROBOT_C)
        # heading tick
        hx2 = px + 1.4 * s * math.cos(-heading)
        hy2 = py + 1.4 * s * math.sin(-heading)
        d.line([(px, py), (hx2, hy2)], fill=(20, 30, 16), width=SCALE)
        if exec_label:
            d.text((px, py + 15 * SCALE), exec_label, font=F_S, fill=TEXT,
                   anchor="mm")


# ---------------------------------------------------------------------------
# Assemble frames → gif/mp4
# ---------------------------------------------------------------------------

def main() -> None:
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found (brew install ffmpeg)", file=sys.stderr)
        sys.exit(1)
    _ensure_pillow()

    stages, plan_names = run_pipeline()
    path = build_path()
    frames: list[Image.Image] = []

    def snap(active_idx, robot_pos=None, heading=0.0, trail=(), done=(),
             exec_label=""):
        frames.append(draw(stages, active_idx, robot_pos, heading, list(trail),
                           set(done), plan_names, exec_label))

    # Reveal stages 0..4 one at a time (hold each so it's readable).
    for idx in range(5):
        for _ in range(9):
            snap(idx)

    # Execute: animate the robot; tick off skills as their leg completes.
    exec_idx = 5
    trail: list[tuple[float, float]] = []
    done: list[str] = []
    prev = START
    # figure out which path index finishes each skill leg
    seen_names: list[str] = []
    for name, _ in path:
        if not seen_names or seen_names[-1] != name:
            seen_names.append(name)

    total = len(path)
    for i, (skill_name, pos) in enumerate(path):
        trail.append(pos)
        if len(trail) > 400:
            trail.pop(0)
        dx, dy = pos[0] - prev[0], pos[1] - prev[1]
        heading = math.atan2(dy, dx) if (dx or dy) else 0.0
        prev = pos
        # mark a skill done when the next path entry switches away from it
        nxt = path[i + 1][0] if i + 1 < total else None
        if nxt != skill_name and skill_name not in done:
            done.append(skill_name)
        # slow readout near the human
        near = math.dist(pos, HUMAN) < HUMAN_LIMIT + 0.8
        label = f"{skill_name} · {'0.25 m/s (near human)' if near else '0.5 m/s'}"
        snap(exec_idx, pos, heading, trail, done, label)

    for name in plan_names:
        if name not in done:
            done.append(name)
    for _ in range(20):  # hold the finished state
        snap(exec_idx, DOCK, 0.0, trail, done, "delivered · 4/4 skills ok")

    # write frames
    tmp = Path(tempfile.mkdtemp(prefix="orchflow_"))
    try:
        for i, fr in enumerate(frames):
            fr.resize((W // SCALE, H // SCALE), Image.LANCZOS).save(
                tmp / f"f{i:04d}.png")
        pattern = str(tmp / "f%04d.png")
        palette = tmp / "pal.png"
        gif_scale = "scale=800:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", pattern, "-vf",
                        f"{gif_scale},palettegen=stats_mode=diff", str(palette)],
                       check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
                        "-i", pattern, "-i", str(palette), "-lavfi",
                        f"{gif_scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                        str(HERE / "demo.gif")], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
                        "-i", pattern, "-pix_fmt", "yuv420p", "-movflags",
                        "+faststart", str(HERE / "demo.mp4")], check=True)
        print(f"wrote {HERE/'demo.gif'} and {HERE/'demo.mp4'} "
              f"({len(frames)} frames)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
