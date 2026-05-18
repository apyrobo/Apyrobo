# APYROBO Demos

Three standalone `clone → run → wow` demos. Each runs on a laptop with no hardware.

| Demo | What it shows | Run |
|------|---------------|-----|
| [drone_survey/](drone_survey/) | 10 drones survey 10 km² in parallel | `python drone_survey/demo.py` |
| [warehouse_robots/](warehouse_robots/) | 3 robots fill orders via `TaskBus` | `python warehouse_robots/demo.py` |
| [humanoid_nlp/](humanoid_nlp/) | NL safety policies enforced at runtime | `python humanoid_nlp/demo.py` |

## Quick start

```bash
pip install apyrobo
python demos/drone_survey/demo.py
python demos/warehouse_robots/demo.py
python demos/humanoid_nlp/demo.py
```

All three use `mock://` adapters — no ROS, no real robots, no API keys required.
Swap the URI to connect real hardware.
