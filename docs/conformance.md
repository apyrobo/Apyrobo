# Conformance Suite

`apyrobo conformance` runs any capability adapter or wire-protocol server —
in any language — against the [APYROBO Protocol spec](../spec/README.md) and
produces a machine-readable report. A spec nobody can test against is a blog
post; this suite is what makes an alternative implementation checkable.

Results for every first-party target are published in
[conformance_results.md](conformance_results.md) and kept current by CI.

## Quick start

```bash
pip install 'apyrobo[conformance]'   # adds jsonschema for schema checks

# A capability adapter, addressed by robot URI (runs in-process):
apyrobo conformance mock://my-robot

# A live orchestration server over WebSocket:
apyrobo conformance ws://localhost:8765

# Any command that speaks the stdio (NDJSON) wire protocol — spawned for you:
apyrobo conformance "stdio:apyrobo serve"
apyrobo conformance "stdio:node my-server.js"
```

Exit code `0` means conformant, `1` means at least one MUST-level check
failed (or, with `--strict`, a SHOULD-level warning), `2` means the run
itself could not start (bad target, missing dependency, unreachable server).

> **Safety:** adapter conformance issues real commands — `move`, `rotate`,
> `stop`, gripper open/close, `disconnect`/`connect`. Motion is minimized
> (the move target is the robot's current position), but run it against a
> simulator or mock, never a robot near people.

## Options

| Flag | Meaning |
|------|---------|
| `--json` | Print the JSON report to stdout instead of the text summary |
| `--output FILE` | Also write the JSON report to `FILE` |
| `--robot URI` | `robot_uri` used in wire-protocol probes (default `mock://conformance-probe`; pick a scheme the target server can plan against) |
| `--timeout N` | Seconds to wait for each server response (default 15) |
| `--strict` | Treat SHOULD-level warnings as failures |

## Check catalog

Each check tests one clause of the spec and carries its RFC-2119 level: a
failed **MUST** check makes the target non-conformant; a failed **SHOULD**
check is reported as a warning.

### Adapter contract (`scheme://name` targets)

Spec: [adapter-contract.md](../spec/adapter-contract.md),
[capability-model.md](../spec/capability-model.md).

| ID | Level | Checks |
|----|-------|--------|
| CAP-01 | MUST | `get_capabilities()` returns a capability profile |
| CAP-02 | MUST | `robot_id` and `name` are non-empty |
| CAP-03 | MUST | `max_speed` is absent or a non-negative number |
| CAP-04 | MUST | serialized profile validates against `robot-capability.schema.json` |
| OPS-01 | MUST | `move(x, y)` accepted and `stop()` halts it |
| OPS-02 | MUST | `move()` accepts an explicit speed argument |
| OPS-03 | MUST | `stop()` is safe to call repeatedly |
| OPT-01…04 | MUST | optional ops (`rotate`, grippers, `cancel`) keep default semantics, never raise |
| OPT-05…07 | MUST | state queries return the documented shapes |
| LIF-01 | MUST | adapter exposes a valid `AdapterState` |
| LIF-02 | MUST | `connect()`/`disconnect()` transition `is_connected` |
| LIF-03 | MUST | disconnect callbacks fire when an established connection is lost |
| SAF-01 | MUST | `stop()` works while disconnected — the safety-critical path |
| FAIL-01 | SHOULD | commands while disconnected fail fast rather than queue |
| FAIL-02/03 | MUST | after a command failure, `get_health()` and `stop()` still work |

### Wire protocol (`ws://…` and `stdio:<command>` targets)

Spec: [wire-protocol.md](../spec/wire-protocol.md).

| ID | Level | Checks |
|----|-------|--------|
| WP-01 | MUST | server responds to a valid task message |
| WP-02 | MUST | response validates against `orchestration-message.schema.json` |
| WP-03 | MUST | response echoes `task` and `robot_uri` |
| WP-04 | MUST | `metadata.status` is `planned` or `error`; planned responses carry a consistent `skills`/`count` |
| WP-05 | MUST | a task targeting an unknown robot scheme yields `status: error`, not a crash |
| WP-06 | MUST | malformed (non-JSON) input does not terminate the connection |
| WP-07 | MUST | unknown message keys are ignored |
| WP-08 | MUST | messages are processed sequentially, one response each |

Responses are correlated by the echoed `task` text (unique per probe), so
the checks tolerate the WebSocket transport's broadcast semantics.

## The report

`--json` / `--output` emit a stable JSON document (format version `1`):

```json
{
  "apyrobo_conformance_report": "1",
  "spec_version": "1.0",
  "apyrobo_version": "4.0.0",
  "target": "ws://localhost:8765",
  "kind": "wire-protocol",
  "timestamp": "2026-07-04T17:00:00+00:00",
  "checks": [
    {"id": "WP-01", "title": "server responds to a valid task message",
     "level": "MUST", "spec_ref": "wire-protocol.md §1",
     "status": "pass", "details": ""}
  ],
  "summary": {"pass": 8, "warn": 0, "fail": 0, "skip": 0},
  "conformant": true
}
```

`conformant` is true when no MUST-level check failed. `skip` marks checks
that could not run (e.g. `jsonschema` not installed) — a skipped check does
not count against conformance but a report with skips is not eligible for
the badge below.

## "APYROBO Conformant" badge

Third-party adapters and protocol implementations may display the badge
once they meet all three conditions:

1. A conformance run against the released version shows `"conformant": true`
   with **zero skipped checks**, against the spec version the badge claims.
2. The JSON report is committed to the project's repository (conventionally
   `conformance-report.json` at the root) so the claim is verifiable.
3. The run is repeated for each release that touches the protocol surface —
   CI is the honest way to do this.

```markdown
[![APYROBO Conformant](https://img.shields.io/badge/APYROBO-conformant%201.0-brightgreen)](./conformance-report.json)
```

The spec froze at `1.0` in July 2026. Badges claiming the earlier
`1.0-draft` should be refreshed by re-running conformance against 1.0. Misrepresenting
conformance (declaring capabilities the hardware cannot perform is itself
non-conformant — [adapter-contract.md §2](../spec/adapter-contract.md)) is
grounds for removal from any first-party listing.

## Testing your own implementation

- **Adapter authors:** register your adapter, then point the suite at it —
  `register_adapter_class("myscheme", MyAdapter)` in your package's import
  path, `apyrobo conformance myscheme://test-unit`. See
  [adapter_authoring.md](adapter_authoring.md).
- **Server authors (any language):** implement the wire protocol and run
  `apyrobo conformance "stdio:<your-server-command>"` or serve WebSocket
  and use `ws://`. Your server needs at least one robot scheme it can plan
  against; pass it with `--robot`.
- **Programmatic use:** `from apyrobo.conformance import run_conformance`
  returns a `ConformanceReport` — useful inside your own pytest suite.
