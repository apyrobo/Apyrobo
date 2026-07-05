# Adapter Authoring Guide

From zero to a **conformance-passing adapter in under an hour**. An adapter
bridges APYROBO's semantic API to one platform: it translates high-level
commands like `move(x, y, speed)` into whatever your robot understands —
ROS 2 topics, a vendor SDK, MQTT, HTTP, serial.

```
APYROBO (semantic)  →  Adapter (translation)  →  Robot (hardware)
   robot.move(2, 3)       publish /cmd_vel          wheels turn
   robot.gripper_close()  send MQTT command          gripper closes
```

The normative contract your adapter must satisfy is
[spec/adapter-contract.md](../spec/adapter-contract.md); the
[conformance suite](conformance.md) checks it mechanically. This guide is
the loop that keeps you green while you implement.

---

## Step 1 — Scaffold (2 minutes)

```bash
pip install apyrobo
apyrobo init acme --adapter
cd acme
pip install -e ".[dev]"
```

You now have a pip-installable package:

```
acme/
├── pyproject.toml                     # entry point registers acme:// for you
├── src/apyrobo_adapter_acme/
│   └── adapter.py                     # AcmeAdapter with TODO markers
├── tests/test_adapter.py              # conformance + contract tests
└── .github/workflows/ci.yml           # pytest + strict conformance in CI
```

The scheme is registered through the `apyrobo.adapters` entry-point group,
so **any** apyrobo command resolves `acme://…` immediately — no imports, no
configuration:

```bash
apyrobo discover "acme://my-robot"
```

Scheme rules: lowercase, and it must not collide with a first-party scheme
(`mock`, `gazebo`, `mqtt`, `http`, `ros2`, `isaac`, `unitree`, `mujoco`) —
the scaffold enforces both.

## Step 2 — Prove the baseline (1 minute)

```bash
apyrobo conformance "acme://test" --strict
```

The scaffold passes all checks with zero warnings *before you write any
code*. That's the invariant to protect: from here on, you change one method
at a time and re-run conformance after each change. When something goes
red, the diff that broke it is one method long.

## Step 3 — Implement the TODOs (the actual work)

Open `src/apyrobo_adapter_acme/adapter.py`. Every TODO marks a place where
the simulated behavior should become a platform call. Priorities, in order:

### 1. `stop()` — the safety-critical path

```python
def stop(self) -> None:
    # Works in EVERY adapter state, never blocks on in-flight goals,
    # safe to call repeatedly. Everything else may fail; stop may not.
    self._client.emergency_stop()
```

Wire it to the platform's lowest-level halt (zero-velocity publish,
e-stop register, watchdog trip). Do **not** guard it behind
`_require_connected` — the conformance suite calls it while disconnected
(check `SAF-01`) and after failed commands (`FAIL-03`).

### 2. `get_capabilities()` — be truthful

Declare **only** what the hardware actually does. Planners treat the
capability list as exhaustive — this is what prevents an LLM from
hallucinating a gripper your robot doesn't have. Declaring a capability
the hardware cannot perform is itself non-conformant.

```python
def get_capabilities(self) -> RobotCapability:
    return RobotCapability(
        robot_id=self.robot_name,
        name=f"Acme-{self.robot_name}",
        capabilities=[
            Capability(capability_type=CapabilityType.NAVIGATE,
                       name="navigate_to", description="Move to (x, y)"),
            # add PICK/PLACE/ROTATE/… only if the hardware has them
        ],
        max_speed=1.5,   # planners MUST NOT exceed this — measure it
    )
```

### 3. `move()` and the fail-fast rule

```python
def move(self, x: float, y: float, speed: float | None = None) -> None:
    self._require_connected("move")          # fail fast, never queue
    effective = min(speed or self._max_speed, self._max_speed)
    self._client.goto(x, y, speed=effective)
```

Keep `_require_connected`: a queued `move` executing on reconnect is a
safety hazard, and the suite checks for it (`FAIL-01`).

### 4. Optional operations

The base class provides spec-correct defaults for everything else
(`rotate` warn+no-op, grippers return `True`, `cancel` delegates to
`stop`, state queries return zeros). Override only what your platform
supports; if you can't support one, **delete your override** rather than
raising — the defaults are the contract.

### 5. Lifecycle

Override `connect()`/`disconnect()` when you hold real resources (ROS
nodes, sockets, SDK sessions). Call `super().connect()` last (or set
`self._state` yourself) so `is_connected` stays truthful, and make sure a
lost connection invokes the disconnect callbacks — `LIF-03` checks it.
`reconnect_with_backoff()` is inherited and free.

## Step 4 — Verify like CI will (2 minutes)

```bash
pytest                                            # contract + conformance
apyrobo conformance "acme://test" --strict \
    --output conformance-report.json              # machine-readable proof
```

The generated GitHub workflow runs exactly this on every push. When the
report shows `"conformant": true` with zero warnings and zero skips
against your released version, commit `conformance-report.json` and claim
the **APYROBO Conformant badge** — see
[docs/conformance.md](conformance.md#apyrobo-conformant-badge) for the
program rules.

> **Safety:** conformance issues real commands (`move`, `stop`,
> `disconnect`…). Point it at a simulator, a bench robot, or your SDK's
> mock — never a robot near people.

## Step 5 — Ship

```bash
python -m build && twine upload dist/*
```

Because registration rides on the entry point, `pip install
apyrobo-adapter-acme` is the entire install experience for your users:

```python
from apyrobo import Robot

robot = Robot.discover("acme://arm-01")   # just works
robot.move(1.0, 2.0, speed=0.5)
```

---

## Registering without a package

For quick experiments inside one process, skip the packaging:

```python
from apyrobo.core.adapters import CapabilityAdapter, register_adapter

@register_adapter("lab")
class LabAdapter(CapabilityAdapter):
    ...
```

or imperatively for classes you don't own:
`register_adapter_class("lab", LabAdapter)`.

---

## Contract reference

Normative version: [spec/adapter-contract.md](../spec/adapter-contract.md).
Conformance check IDs in parentheses.

### Required (abstract — you must implement)

| Method | Contract |
|--------|----------|
| `get_capabilities() → RobotCapability` | Truthful, schema-valid profile (CAP-01…04) |
| `move(x, y, speed=None)` | Metres, robot frame; never exceed declared `max_speed` (OPS-01/02) |
| `stop()` | Immediate halt; every state, never blocks, repeatable (OPS-03, SAF-01, FAIL-03) |

### Optional (defaults are the contract — override or leave alone, never raise)

| Method | Default | Checks |
|--------|---------|--------|
| `rotate(angle_rad, speed=None)` | warn + no-op; positive = CCW | OPT-01 |
| `gripper_open()` / `gripper_close()` | return `True`; `False` on failure | OPT-02/03 |
| `cancel()` | delegates to `stop()` | OPT-04 |
| `get_position() → (x, y)` | `(0.0, 0.0)`; keep it fast — the watchdog polls it | OPT-05 |
| `get_orientation() → float` | `0.0` radians | OPT-06 |
| `get_health() → dict` | `{state, adapter, robot}` + `battery_pct`/`uptime_s`/`errors` when available | OPT-07, FAIL-02 |
| `connect()` / `disconnect()` | state transitions + callbacks | LIF-01…03 |

### Existing adapters to crib from

| Adapter | URI | Protocol | Source |
|---------|-----|----------|--------|
| `MockAdapter` | `mock://` | In-memory | [`apyrobo/core/adapters.py`](../apyrobo/core/adapters.py) |
| `GazeboAdapter` | `gazebo://` | Sim API | `apyrobo/core/adapters.py` |
| `MQTTAdapter` | `mqtt://` | MQTT topics | `apyrobo/core/adapters.py` |
| `HTTPAdapter` | `http://` | REST API | `apyrobo/core/adapters.py` |
| `UnitreeAdapter` | `unitree://` | Vendor SDK | [`apyrobo/core/unitree_adapter.py`](../apyrobo/core/unitree_adapter.py) |
| `IsaacAdapter` | `isaac://` | Isaac Sim | [`apyrobo/core/isaac_adapter.py`](../apyrobo/core/isaac_adapter.py) |

### Tips

1. **`stop()` before everything** — implement and hand-test it first.
2. **Report capabilities accurately** — the executor and safety layer
   trust them for precondition checks.
3. **Fail fast when disconnected** — raise, don't queue.
4. **Keep `get_position()` cheap** — the safety watchdog polls it.
5. **Run `apyrobo conformance --strict` after every method you touch** —
   a one-method diff is easy to debug; an hour of drift is not.
