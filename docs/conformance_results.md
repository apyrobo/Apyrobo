# Conformance Results — First-Party Targets

Spec 1.0 results for every adapter and wire-protocol target that ships in
this repository, produced by [`apyrobo conformance`](conformance.md). The
CI jobs cited below re-run these checks continuously — this table records
*where* each result is produced, so it stays checkable rather than becoming
a stale snapshot.

Last updated: 2026-07-19 (spec 1.0).

## Adapters (adapter-contract checks, 20 MUST + 1 SHOULD)

| Target | Result | Where verified |
|--------|--------|----------------|
| `mock://` | **Conformant** — 20/20, 1 SHOULD warning¹ | `conformance` CI job, every commit |
| `vda5050://` | **Conformant** — full suite, 0 warnings, vs a simulated VDA 5050 AGV (in-process transport; the MQTT transport is real but the conformance rig doesn't need a broker) | `tests/test_vda5050_adapter.py`, every commit |
| `ros2://` | First CI run vs a live physics TurtleBot3 in Gazebo completed every check with 1 SHOULD warning¹ (rclpy then segfaulted at interpreter teardown, after the report was written — the job now reads the JSON verdict); row flips to **Conformant** on the first green run | `gazebo` profile CI job ([integration.yml](../.github/workflows/integration.yml)), every commit |
| `gazebo_native://` | Conformant — 20/20, 1 SHOULD warning¹ · **in-memory stand-in**² | `conformance` CI job, every commit |
| `isaac://` | Conformant — 20/20, 1 SHOULD warning¹ · **in-memory stand-in**² | `conformance` CI job, every commit |
| `mujoco://` | **Conformant** — 21/21, 0 warnings, vs **real MuJoCo physics** (the bridge loads and steps an actual model; fail-fast on disconnect) | `conformance` CI job + `tests/test_mujoco_bridge.py`, every commit |
| `gazebo://` | Not CI-covered — needs a live ROS-topic Gazebo Classic rig; run `apyrobo conformance gazebo://<robot>` inside the `gazebo` compose profile | — |
| `http://` | Not CI-covered — needs a live HTTP robot endpoint; run `apyrobo conformance http://<host>` against yours | — |
| `mqtt://` | Not CI-covered — needs a live MQTT broker + robot; run `apyrobo conformance mqtt://<broker>` against yours | — |

¹ **FAIL-01 (SHOULD)**: commands issued while disconnected return silently
instead of failing fast. `vda5050://` and `mujoco://` pass this check
(fail-fast, 0 warnings); the in-memory adapters — and, per its first live
CI run, `ros2://` — accept the command. Recorded, not hidden; making the
ros2 adapter fail fast is a flagged follow-up.

² The in-memory stand-ins (`gazebo_native://`, `isaac://`) satisfy the
adapter *contract* but do not drive a real simulator — that is exactly what
[Arc 3 of the roadmap](../ROADMAP.md#arc-3--modern-simulation-no-stand-ins)
exists to fix (`mujoco://` already graduated to a real bridge).
Contract-conformant ≠ physically real; this table says which is which.

## Wire-protocol servers (WP checks, 8 MUST)

| Target | Result | Where verified |
|--------|--------|----------------|
| `apyrobo serve` (stdio) | **Conformant** — 8/8 | `conformance` CI job, every commit |
| `apyrobo serve --transport websocket` | **Conformant** — 8/8 | `conformance` CI job, every commit |

## Clients

The suite tests *servers and adapters*; clients are verified by exercising
them against a conformant server:

| Client | Where verified |
|--------|----------------|
| [`apyrobo-client-ts`](../packages/apyrobo-client-ts/) | `ts-client` CI job: unit + cross-language tests + the [interop demo](../demos/ts_interop/) (plan **and execute**) against the Python reference server, every commit |

## Reproduce any row

```bash
pip install 'apyrobo[conformance,websocket]'
apyrobo conformance mock://check          # any adapter row
apyrobo conformance "stdio:apyrobo serve" # the stdio row
```

Machine-readable JSON for every CI run is uploaded as the
`conformance-reports` artifact on the `Conformance (first-party targets)`
job.
