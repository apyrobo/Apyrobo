# Demo: 10-Drone Coordinated Survey

<img src="demo.gif" alt="Demo: 10-Drone Coordinated Survey — terminal recording" width="720">

_Re-render: `./record.sh` (requires [vhs](https://github.com/charmbracelet/vhs))_

10 drones survey a 10 km² grid in parallel — no hardware required.

```bash
pip install apyrobo
python demo.py                   # 10 drones
python demo.py --fleet-size 100  # any fleet size, one 1 km² sector each
```

**What you'll see:** All 10 drones launch simultaneously. Each claims a 1 km² sector,
navigates, and reports back. Anomalies (thermal hotspots, debris) stream in as sectors
complete. On mock:// each sector takes about a millisecond, so the timings show
dispatch overhead, not flight time.

**Key APYROBO APIs used:**
- `MockAdapter` + `Robot` — zero-hardware robots with full capability contracts
- `Agent(provider="rule")` — offline planner, no API key needed
- `SkillExecutor.execute_graph()` — runs the skill DAG
- `ThreadPoolExecutor` — parallel fleet dispatch

**To run on real hardware:** APYROBO ships no drone adapter yet. Scaffold one, implement
its TODOs against your autopilot's API, and keep the conformance suite green
([adapter authoring guide](../../docs/adapter_authoring.md)):
```bash
apyrobo init my-drone --adapter
cd my-drone && pip install -e ".[dev]"
apyrobo conformance "my-drone://test" --strict
```
Then swap `mock://` for `my-drone://` in `survey_sector()`.
