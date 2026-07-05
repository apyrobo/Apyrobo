/**
 * Cross-language integration: this TypeScript client against the real
 * Python reference server, spawned over stdio.
 *
 * Needs the apyrobo package installed. Set APYROBO_SERVE to the server
 * command (default "apyrobo serve"); the tests skip when the command
 * cannot be spawned.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { test } from "node:test";

import { ApyroboClient, isError, isPlanned } from "../src/index.js";

const serveCommand = (process.env.APYROBO_SERVE ?? "apyrobo serve").split(" ");
const [command = "apyrobo", ...args] = serveCommand;

const apyroboAvailable =
  spawnSync(command, ["--help"], { stdio: "ignore" }).status === 0;

test(
  "plans a task through the reference server",
  { skip: !apyroboAvailable && "apyrobo not installed (set APYROBO_SERVE)" },
  async () => {
    const client = await ApyroboClient.spawn(command, args);
    try {
      const response = await client.submitTask(
        "navigate to the dock and pick up the box",
        { robotUri: "mock://ts-integration", timeoutMs: 60_000 }
      );
      assert.ok(isPlanned(response), JSON.stringify(response));
      assert.equal(response.robot_uri, "mock://ts-integration");
      assert.ok(response.metadata.count >= 1);
      assert.equal(response.source, "orchestration_server");
    } finally {
      client.close();
    }
  }
);

test(
  "unknown robot scheme yields a status:error response",
  { skip: !apyroboAvailable && "apyrobo not installed (set APYROBO_SERVE)" },
  async () => {
    const client = await ApyroboClient.spawn(command, args);
    try {
      const response = await client.submitTask("go", {
        robotUri: "ts-bogus://nowhere",
        timeoutMs: 60_000,
      });
      assert.ok(isError(response), JSON.stringify(response));
      assert.match(response.metadata.error, /ts-bogus/);
    } finally {
      client.close();
    }
  }
);

test(
  "omitted robot_uri resolves to the server default",
  { skip: !apyroboAvailable && "apyrobo not installed (set APYROBO_SERVE)" },
  async () => {
    const client = await ApyroboClient.spawn(command, args);
    try {
      const response = await client.submitTask("scan the area", {
        timeoutMs: 60_000,
      });
      assert.ok(isPlanned(response), JSON.stringify(response));
      assert.match(response.robot_uri ?? "", /^[a-z][a-z0-9+.-]*:\/\/.+$/);
    } finally {
      client.close();
    }
  }
);
