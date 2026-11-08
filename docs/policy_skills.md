# Policy-backed skills — run a VLA / LeRobot policy safely

Neural policies (π0, OpenVLA, anything LeRobot-shaped) can't be formally
certified, so the VLA deployment literature treats **runtime safety
monitors as mandatory**. APYROBO already is one: `apyrobo.skills.policy`
makes a learned policy just another skill-graph node, wrapped in the same
safety envelope, timeout, and recovery semantics as every other skill.

## The 60-second version

```python
from apyrobo import Robot, SafetyEnforcer
from apyrobo.skills.policy import PolicyRunner

robot = Robot.discover("ros2://base1")
enforcer = SafetyEnforcer(robot)              # clamps + audit trail

result = PolicyRunner(
    policy=my_policy,                         # anything with select_action(obs)
    robot=enforcer,                           # the enforcer is robot-shaped
    hz=5.0,
    max_duration_sec=60.0,
    success=lambda obs: obs["x"] > 2.0,
).run()

print(result.success, result.stop_reason, result.clamped_steps)
```

Or inside a plan:

```python
graph.add_skill(BUILTIN_SKILLS["run_policy"],
                parameters={"policy": my_policy, "max_duration_sec": 60.0})
```

## What the runner guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| A runaway policy cannot ask for more than one bounded step | `max_step_m` per tick, enforced by the runner itself |
| Speed limits, collision zones, audit trail | pass a `SafetyEnforcer` as `robot` — its clamps apply to every action |
| Episodes always end | `max_duration_sec`, optional `max_steps` |
| Failures fail closed | any policy/robot exception aborts the episode; `robot.stop()` runs in a `finally` |
| Observable outcome | `PolicyResult`: success, stop reason, steps, clamped-step count, trajectory |

## The Policy contract

Anything with `select_action(observation: dict) -> dict | None`
(optionally `reset()`). Observations carry `x`, `y`, `position`, `theta`
by default (override with `observe=`). Actions are position deltas
`{"dx", "dy"}` (the common VLA base-action shape) or absolute targets
`{"x", "y"}`. Returning `None` ends the episode.

## Wrapping a LeRobot policy

Torch stays **your** dependency — APYROBO core never imports it:

```python
from collections.abc import Mapping

class LeRobotBasePolicy:
    def __init__(self, lerobot_policy, to_tensor, from_tensor):
        self._p = lerobot_policy
        self._to, self._from = to_tensor, from_tensor
    def reset(self):
        self._p.reset()
    def select_action(self, obs):
        out = self._p.select_action(self._to(obs))
        if isinstance(out, Mapping):   # LeRobot >= 0.7 PolicyOutput
            out = out["action"]
        return self._from(out)
```

`to_tensor`/`from_tensor` are your observation/action mappings (camera
tensors in, base deltas out). Since LeRobot 0.7.0, `select_action` returns
either a bare action tensor or, for a policy that predicts more than actions
(a subtask, boxes, a future frame), a `PolicyOutput` dict with the action
under `"action"` and the rest under `"prediction"`. The wrapper takes the
action out either way (`from collections.abc import Mapping`); LeRobot's own
`lerobot.processor.policy_output_to_transition` does the same if you use its
processor pipeline. The wrapper and example are written against
`lerobot>=0.7,<0.8`; the stand-in policy in the example returns the
`PolicyOutput` shape on its first tick so the test covers it, but no real
LeRobot install runs in CI. Run it in sim first: the same `PolicyRunner`
drives `mock://` and the Gazebo stack unchanged.

Runnable version: [`examples/04_learned_policy_safety.py`](../examples/04_learned_policy_safety.py)
wraps a LeRobot-shaped stand-in policy (plain lists instead of tensors, no
torch needed) and runs three episodes on `mock://`: a nominal run, a
one-off 25 m runaway action clamped to a single 0.5 m step, and a drift
into a keep-out zone that the `SafetyEnforcer` rejects, ending the episode
with the robot stopped. `tests/test_example_learned_policy.py` checks all three.

## Honesty line

Verified with deterministic mock policies, including an adversarial
runaway policy that the bounds must contain (`tests/test_policy_skills.py`).
Not yet run against a real VLA checkpoint — that worked example is the
remaining piece of this roadmap wedge.
