# Demo: 10-Drone Coordinated Survey

10 drones survey a 10 km² grid in parallel — no hardware required.

```bash
pip install apyrobo
python demo.py
```

**What you'll see:** All 10 drones launch simultaneously. Each claims a 1 km² sector,
navigates, and reports back. Anomalies (thermal hotspots, debris) stream in as sectors
complete. The fleet finishes the full survey in the time it takes one drone to do a
single sector.

**Key APYROBO APIs used:**
- `MockAdapter` + `Robot` — zero-hardware robots with full capability contracts
- `Agent(provider="rule")` — offline planner, no API key needed
- `SkillExecutor.execute_graph()` — runs the skill DAG
- `ThreadPoolExecutor` — parallel fleet dispatch

**To run on real hardware:**
```python
# Just change the URI
robot = Robot("dds://drone_0@192.168.1.10", DDSAdapter("drone_0"))
```
