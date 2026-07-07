# APYROBO Demos

Three standalone `clone → run → wow` demos. Each runs on a laptop with no hardware.

| Demo | What it shows | Run |
|------|---------------|-----|
| [drone_survey/](drone_survey/) | 10 drones survey 10 km² in parallel | `python drone_survey/demo.py` |
| [warehouse_robots/](warehouse_robots/) | 3 robots fill orders via `TaskBus` | `python warehouse_robots/demo.py` |
| [humanoid_nlp/](humanoid_nlp/) | NL safety policies enforced at runtime | `python humanoid_nlp/demo.py` |
| [fleet_view/](fleet_view/) | **Live browser render** — a fleet moving in real time, tasks planned and dispatched over the wire protocol | `python fleet_view/server.py` → see its README |

Each directory ships a recorded run (`demo.gif` / `demo.mp4`) plus the
[vhs](https://github.com/charmbracelet/vhs) tape and `record.sh` that made it —
change a demo, run `./record.sh`, and the video regenerates. Recordings pace
their output via `APYROBO_DEMO_PACE` (the artificial sleeps are subtracted
from the printed timing stats).

## Quick start

```bash
pip install apyrobo
python demos/drone_survey/demo.py
python demos/warehouse_robots/demo.py
python demos/humanoid_nlp/demo.py
```

All use `mock://` adapters — no ROS, no real robots, no API keys required.
Swap the URI to connect real hardware.

The [fleet_view/](fleet_view/) demo is interactive (a browser canvas fed by
`apyrobo serve` over WebSocket, rendered by the reference TypeScript client);
see its README for setup.
