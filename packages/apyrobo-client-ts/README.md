# apyrobo-client

TypeScript client for the [APYROBO orchestration wire protocol](../../spec/wire-protocol.md)
(spec 1.0). Submit natural-language tasks to any conformant
orchestration server — such as `apyrobo serve` — and receive planning
results. Zero runtime dependencies; works in Node ≥ 20 and in browsers.

This package is the *reference non-Python client*: it exists to prove the
protocol is real and language-agnostic, and doubles as the SDK for web
dashboards and bots.

## Usage

```ts
import { ApyroboClient, isPlanned } from "apyrobo-client";

// Over WebSocket (browser or Node ≥ 21; pass options.webSocket on Node 20):
const client = await ApyroboClient.connect("ws://localhost:8765");

// …or spawn a server and speak stdio NDJSON (Node only):
// const client = await ApyroboClient.spawn("apyrobo", ["serve"]);

const response = await client.submitTask("navigate to the dock", {
  robotUri: "mock://turtlebot4",
});

if (isPlanned(response)) {
  for (const skill of response.metadata.skills) {
    console.log(`${skill.skill_id}: ${skill.name}`);
  }
} else {
  console.error(response.metadata.error);
}

client.close();
```

Observe every broadcast message (the WebSocket transport delivers responses
to all connected clients):

```ts
const unsubscribe = client.onMessage((msg) => console.log(msg));
```

Correlation follows the spec: the server echoes `task` back verbatim, so
responses are matched by task text. Make task text unique when multiple
clients may submit identical tasks concurrently.

Spec 1.0 has no cancellation or execution streaming on the wire; those are
reserved for a future minor revision, and the client already tolerates
unknown `metadata.status` values as the spec requires.

## Development

```bash
npm install
npm run build     # compile to dist/
npm test          # unit tests (fake transport)

# Cross-language integration against the real reference server:
pip install apyrobo
npm test          # integration tests un-skip when `apyrobo` is on PATH
# or point at a specific interpreter:
APYROBO_SERVE="/path/to/.venv/bin/apyrobo serve" npm test
```
