# Demo: Orchestration Flow — the pipeline, made visible

The other demos show *what* APYROBO does. This one shows *how*: a single task
followed all the way down the stack the README diagram promises —

```
natural language → Agent → Skill Graph → Safety Enforcer → robot
```

![Orchestration flow](demo.gif)

Every panel on the left is filled in by the **real** APYROBO objects as the
task runs — nothing is mocked for show:

| Stage | Real call | What you see |
|-------|-----------|--------------|
| **Discover** | `robot.capabilities()` | the robot's declared capabilities + `max_speed` |
| **Plan** | `Agent(provider="rule").plan(task, robot)` | how many skills were chosen from the catalog |
| **Skill graph** | `graph.get_execution_order()` | the ordered skills, each tagged with the capability it needs |
| **Safety** | `SafetyEnforcer.move()` | a requested **2.5 m/s clamped to 0.5**, a **no-go-zone entry rejected**, and a human proximity limit — these are the interventions the enforcer actually recorded |
| **Execute** | `SkillExecutor.execute_graph()` | the plan running skill-by-skill while the robot (right) drives from shelf to dock, routing **around the no-go zone** and slowing near the human |

So the "theory" diagram and the running system are the same thing: the numbers
and skills on screen come straight from the library.

## Run it

```bash
pip install -e . Pillow      # ffmpeg on PATH for the mp4
python demos/orchestration_flow/flow.py
```

`./record.sh` regenerates `demo.gif` / `demo.mp4` — change the task, the
policy, or the world and re-run it.

## Try changing

- **The task** (`TASK` in `flow.py`): "patrol the perimeter" plans a different
  skill set; watch the graph and execute checklist change.
- **The policy** (`SafetyPolicy(...)`): lower `max_speed` and the clamp figure
  moves with it; move the `collision_zones` rectangle and the robot re-routes.
- **The provider**: swap `Agent(provider="rule")` for `Agent(provider="anthropic")`
  (with an API key) and the PLAN stage is produced by an LLM instead — every
  downstream stage is identical, which is the point.
