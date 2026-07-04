# Orchestration Wire Protocol

**Spec version 1.0-draft** · Reference implementation:
[`apyrobo/orchestration/adapter.py`](../apyrobo/orchestration/adapter.py)

The wire protocol is how external clients — shell scripts, web dashboards,
chat bots, non-Python programs — submit tasks to an APYROBO orchestration
server and receive planning results. It is deliberately minimal: one message
shape, two transports, no session state.

## 1. Message

Every message in both directions is a single JSON object
([schema](schemas/orchestration-message.schema.json)):

```json
{
  "task": "navigate to the dock",
  "robot_uri": "mock://turtlebot4",
  "metadata": {},
  "source": ""
}
```

| Field | Type | Direction | Semantics |
|-------|------|-----------|-----------|
| `task` | string | both | Natural-language task description. Required — the only field a client MUST send. |
| `robot_uri` | string | both | Target robot as a robot URI (see [adapter-contract.md](adapter-contract.md#robot-uris)). Servers MUST default it to their configured default robot when absent. |
| `metadata` | object | both | Free-form. On responses, carries the result (§3). Implementations MUST ignore unknown keys. |
| `source` | string | both | Identifier of the sender. The reference server sets `"orchestration_server"` on responses. Clients MAY leave it empty. |

## 2. Transports

### 2.1 stdio (NDJSON)

One JSON object per line ("\n"-terminated), UTF-8, requests on stdin,
responses on stdout. EOF on stdin signals shutdown. Blank lines end the
stream in the reference implementation; clients SHOULD NOT send them.

### 2.2 WebSocket

Same JSON objects as text frames. Started with
`apyrobo serve --transport websocket --ws-port 8765`.

- Multiple clients MAY be connected simultaneously.
- The server broadcasts **every response to all connected clients**; clients
  MUST be prepared to receive responses to tasks they did not submit and
  SHOULD correlate by echoing a unique key in `metadata` (the server echoes
  the request's `task` and `robot_uri` back verbatim).
- Binary frames are decoded as UTF-8 text.

### 2.3 Malformed input

A line/frame that is not valid JSON MUST NOT terminate the connection.
The reference server treats the raw text as the `task` of a new message
with all other fields defaulted. Alternative implementations MAY instead
respond with a `status: "error"` message; they MUST do one or the other.

## 3. Responses

The server echoes `task` and `robot_uri` and reports the outcome in
`metadata.status`:

**Success** — the task was planned:

```json
{
  "task": "navigate to the dock",
  "robot_uri": "mock://turtlebot4",
  "metadata": {
    "status": "planned",
    "skills": [{"skill_id": "s1", "name": "navigate_to"}],
    "count": 1
  },
  "source": "orchestration_server"
}
```

**Failure** — discovery or planning raised:

```json
{
  "task": "navigate to the dock",
  "robot_uri": "bogus://nowhere",
  "metadata": {"status": "error", "error": "No adapter registered for scheme 'bogus'"},
  "source": "orchestration_server"
}
```

`metadata.status` MUST be `"planned"` or `"error"` in spec 1.0. Additional
status values (e.g. streaming execution progress) are reserved for a minor
revision; clients MUST ignore messages whose status they do not recognize
rather than fail.

## 4. Server loop semantics

An orchestration server processes messages strictly sequentially:
receive → plan → respond, one at a time. Spec 1.0 has no request
pipelining or cancellation. A server that persists in-flight tasks (crash
recovery) MUST NOT re-execute a task whose response was already sent.

## 5. Explicit non-guarantees in 1.0

So that implementers don't infer promises the protocol doesn't make:

- No authentication — deploy behind an authenticated transport (the REST
  gateway in `apyrobo.api` provides API-key auth; the raw wire protocol does not).
- No delivery ordering guarantee across multiple WebSocket clients.
- No message size limit is specified; servers MAY impose one and SHOULD
  reject oversized frames with an `error` response rather than disconnecting.
