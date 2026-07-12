# Capability Adapter Contract

**Spec version 1.0** · Reference implementation:
[`apyrobo/core/adapters.py`](../apyrobo/core/adapters.py)

A *capability adapter* translates APYROBO's semantic commands into one
platform's native interface — ROS 2, a vendor SDK, MQTT, HTTP, or a
simulator. This document is the behavioral contract an adapter must satisfy
to be conformant; `apyrobo conformance <scheme://name>` tests it
([docs/conformance.md](../docs/conformance.md)).

## 1. Robot URIs

Robots are addressed as `scheme://name`. The scheme selects the adapter via
the adapter registry; everything after `://` identifies the robot and MUST be
non-empty. Adapters MAY define further structure inside the name part (e.g.
`unitree://go2@192.168.1.12` parses model and host).

First-party schemes: `mock`, `gazebo`, `mqtt`, `http`, `ros2`, `isaac`,
`unitree`, `mujoco`. Third-party adapters register their own scheme; scheme
names SHOULD be lowercase ASCII and MUST NOT collide with a first-party
scheme.

## 2. Required operations

Every adapter MUST implement:

| Operation | Contract |
|-----------|----------|
| `get_capabilities() → RobotCapability` | Return the robot's full profile per [capability-model.md](capability-model.md). MUST be truthful — declaring a capability the hardware cannot perform is non-conformant. |
| `move(x, y, speed?)` | Command motion toward `(x, y)` in the robot's coordinate frame, metres. `speed` in m/s; when omitted the adapter chooses, never exceeding the declared `max_speed`. |
| `stop()` | **Immediately** halt all motion. MUST work in every adapter state, MUST NOT block on in-flight goals, and MUST be safe to call repeatedly. This is the safety-critical path: everything else may fail; `stop` may not. |

## 3. Optional operations

Optional operations have specified defaults so callers can rely on uniform
behavior; an adapter that cannot support one MUST keep the default semantics
(no-op with the documented return value), not raise:

| Operation | Default behavior | Notes |
|-----------|-----------------|-------|
| `rotate(angle_rad, speed?)` | warn + no-op | Positive = counter-clockwise (REP-103) |
| `gripper_open()` / `gripper_close()` | return `True` | Return `False` on failure, don't raise |
| `cancel()` | delegates to `stop()` | Cancels the current navigation goal only |
| `get_position() → (x, y)` | `(0.0, 0.0)` | Robot coordinate frame, metres |
| `get_orientation() → radians` | `0.0` | |
| `get_health() → object` | `{state, adapter, robot}` | SHOULD add `battery_pct`, `uptime_s`, `errors` when available |

## 4. Lifecycle and state

An adapter is always in exactly one `AdapterState`: `disconnected`,
`connecting`, `connected`, or `error`.

- `connect()` / `disconnect()` move between `connected` and `disconnected`.
- Adapters SHOULD support registering disconnect/reconnect callbacks and MUST
  invoke disconnect callbacks when an established connection is lost, however
  it was lost.
- Commands issued while not `connected` SHOULD fail fast with a clear error
  rather than queue silently — a queued `move` executing on reconnect is a
  safety hazard.
- Reconnecting adapters SHOULD use exponential backoff and MUST NOT replay
  motion commands from before the disconnect.

## 5. Failure semantics

- Command failures are reported by exception (or `False` for gripper ops);
  adapters MUST NOT swallow hardware errors and report success.
- After any command failure the adapter MUST remain in a queryable state:
  `get_health()` and `stop()` MUST still work.
- Timeouts are the caller's responsibility (skill executor); adapters SHOULD
  make blocking calls interruptible where the platform allows it.
