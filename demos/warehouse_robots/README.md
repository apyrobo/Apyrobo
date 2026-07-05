# Demo: Warehouse Multi-Robot Pick-and-Pack

<img src="demo.gif" alt="Demo: Warehouse Multi-Robot Pick-and-Pack — terminal recording" width="720">

_Re-render: `./record.sh` (requires [vhs](https://github.com/charmbracelet/vhs))_

Three specialized robots collaborate to fill orders via APYROBO's `TaskBus`.
The bus routes each step to the robot with the right capability automatically.

```bash
pip install apyrobo
python demo.py
```

**What you'll see:** 5 orders processed end-to-end. The picker navigates to each item
shelf, the packer consolidates items into a box, the hauler delivers to the dock.
`TaskBus.dispatch()` picks the right robot for each step — no routing code needed.

**Key APYROBO APIs used:**
- `TaskBus` — shared coordination bus with capability-aware routing
- `MultiAgentCoordinator` — background worker thread per robot
- `bus.dispatch(task, required_capability="PICK")` — routes to best available agent
- `bus.broadcast("emergency_stop")` — fan-out to all robots

**To scale up:**
```python
# Add more pickers for higher throughput — bus balances load automatically
for i in range(5):
    coord = MultiAgentCoordinator(agent, robot, bus,
        agent_id=f"picker_{i}", capabilities=["PICK", "NAVIGATE"])
    coord.start()
```
