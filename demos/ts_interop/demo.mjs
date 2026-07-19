#!/usr/bin/env node
/**
 * Demo: TypeScript → wire protocol → Python server → robot
 * =========================================================
 * The cross-language interop proof: this Node script uses the reference
 * TypeScript client (packages/apyrobo-client-ts, zero dependencies) to
 * submit a natural-language task to `apyrobo serve --execute`. The Python
 * server plans the task and executes it on the robot; the outcome comes
 * back over the wire in `metadata.execution` — all within wire-protocol
 * spec 1.0. No Python is imported here; the only contract is the spec.
 *
 * Modes:
 *   node demo.mjs                          # spawn `apyrobo serve` (stdio), mock robot
 *   node demo.mjs --robot mock://bot2      # same, different robot URI
 *   node demo.mjs --ws ws://host:8765      # connect to a running server
 *                                          # (e.g. Gazebo: see README.md)
 *
 * Requires the client to be built once:
 *   cd packages/apyrobo-client-ts && npm install && npm run build
 */
import { parseArgs } from "node:util";

const clientPath = new URL(
  "../../packages/apyrobo-client-ts/dist/index.js",
  import.meta.url
);

let ApyroboClient, isError, isPlanned;
try {
  ({ ApyroboClient, isError, isPlanned } = await import(clientPath));
} catch {
  console.error(
    "error: TypeScript client not built.\n" +
      "  cd packages/apyrobo-client-ts && npm install && npm run build"
  );
  process.exit(2);
}

const { values: opts } = parseArgs({
  options: {
    robot: { type: "string", default: "mock://interop_bot" },
    task: { type: "string", default: "deliver package from (1, 2) to (5, 5)" },
    ws: { type: "string" },
    timeout: { type: "string", default: "120000" },
    // How long to keep retrying the initial --ws connection. The server
    // opens its port only once robot discovery succeeds (a Gazebo stack
    // can take many minutes to boot), and Docker's port proxy accepts TCP
    // before the server is really there — so retry at the protocol level.
    "connect-timeout": { type: "string", default: "600000" },
  },
});
const timeoutMs = Number(opts.timeout);
const connectTimeoutMs = Number(opts["connect-timeout"]);

const say = (line = "") => console.log(line);

say("╔══════════════════════════════════════════════════════════╗");
say("║  APYROBO — TypeScript client → wire protocol → robot     ║");
say("╚══════════════════════════════════════════════════════════╝");
say();

let client;
if (opts.ws !== undefined) {
  say(`① Connecting to a running server at ${opts.ws} …`);
  const deadline = Date.now() + connectTimeoutMs;
  for (;;) {
    try {
      client = await ApyroboClient.connect(opts.ws);
      break;
    } catch (err) {
      if (Date.now() >= deadline) {
        console.error(`   gave up after ${connectTimeoutMs} ms: ${err.message}`);
        process.exit(1);
      }
      say(`   not up yet (${err.message}) — retrying in 5 s`);
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
  }
} else {
  const serveCommand = (
    process.env.APYROBO_SERVE ?? "apyrobo serve"
  ).split(" ");
  const [command, ...baseArgs] = serveCommand;
  const args = [...baseArgs, "--execute", "--robot", opts.robot];
  say(`① Spawning the Python reference server (stdio):`);
  say(`   ${command} ${args.join(" ")}`);
  client = await ApyroboClient.spawn(command, args);
}
say();

say(`② Submitting over the wire: ${JSON.stringify(opts.task)}`);
say(`   target robot: ${opts.robot}`);
let response;
try {
  response = await client.submitTask(opts.task, {
    robotUri: opts.robot,
    timeoutMs,
  });
} catch (err) {
  console.error(`   FAILED: ${err.message}`);
  client.close();
  process.exit(1);
}
say();

if (isError(response)) {
  console.error(`③ Server reported an error: ${response.metadata.error}`);
  client.close();
  process.exit(1);
}
if (!isPlanned(response)) {
  console.error(`③ Unexpected response: ${JSON.stringify(response)}`);
  client.close();
  process.exit(1);
}

say(`③ The Python server planned ${response.metadata.count} skill(s):`);
for (const skill of response.metadata.skills) {
  say(`   • ${skill.skill_id}`);
}
say();

const execution = response.metadata.execution;
if (execution === undefined) {
  console.error(
    "④ No execution report — is the server running with --execute?"
  );
  client.close();
  process.exit(1);
}

say(`④ …and executed them on ${response.robot_uri}:`);
say(`   status:          ${execution.status}`);
say(`   steps completed: ${execution.steps_completed}`);
if (execution.error !== undefined) {
  say(`   error:           ${execution.error}`);
}
say();

client.close();

if (execution.status !== "completed") {
  console.error("Execution did not complete — failing.");
  process.exit(1);
}
say("Cross-language round trip complete: TypeScript planned nothing,");
say("imported nothing from Python — every byte crossed as spec-1.0 JSON. ✓");
