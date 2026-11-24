# Demo: Warehouse Multi-Robot Pick-and-Pack

<img src="demo.gif" alt="Demo: Warehouse Multi-Robot Pick-and-Pack — terminal recording" width="720">

_Re-render: `./record.sh` (requires [vhs](https://github.com/charmbracelet/vhs))_

Three specialized robots collaborate to fill orders via APYROBO's `TaskBus`.
The bus routes each step to the robot with the right capability automatically.

```bash
pip install apyrobo
python demo.py
```

**What you'll see:** 5 orders, each step routed and planned. The picker is sent to
each item's shelf, the packer gets the boxing step, the hauler gets the delivery to
the dock; every line names the robot that took the step and the skills its agent
planned. `TaskBus.dispatch()` picks the robot by capability — no routing code needed.
The bus plans each step; it does not drive the mock robots. A step the agent
refuses (a drive with no coordinates, say) prints `✗ not planned` with its reason.

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
