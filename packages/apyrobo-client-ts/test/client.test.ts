/** Unit tests: ApyroboClient against an in-memory transport. */
import assert from "node:assert/strict";
import { test } from "node:test";

import { ApyroboClient, isError, isPlanned } from "../src/index.js";
import { Transport } from "../src/transport.js";

class FakeTransport implements Transport {
  sent: string[] = [];
  handler: (text: string) => void = () => {};
  closeHandler: (reason: string) => void = () => {};
  closed = false;

  send(text: string): void {
    if (this.closed) throw new Error("transport closed");
    this.sent.push(text);
  }
  onMessage(handler: (text: string) => void): void {
    this.handler = handler;
  }
  onClose(handler: (reason: string) => void): void {
    this.closeHandler = handler;
  }
  close(): void {
    this.closed = true;
  }

  /** Simulate a server response frame. */
  reply(message: unknown): void {
    this.handler(JSON.stringify(message));
  }
}

function plannedReply(task: string, robotUri = "mock://default") {
  return {
    task,
    robot_uri: robotUri,
    metadata: {
      status: "planned",
      skills: [{ skill_id: "s1", name: "navigate_to" }],
      count: 1,
    },
    source: "orchestration_server",
  };
}

test("submitTask resolves with the planned response", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  const promise = client.submitTask("go to dock", {
    robotUri: "mock://turtle",
  });
  assert.equal(transport.sent.length, 1);
  assert.deepEqual(JSON.parse(transport.sent[0]!), {
    task: "go to dock",
    robot_uri: "mock://turtle",
  });

  transport.reply(plannedReply("go to dock", "mock://turtle"));
  const response = await promise;
  assert.ok(isPlanned(response));
  assert.equal(response.metadata.count, 1);
  assert.equal(response.metadata.skills[0]!.name, "navigate_to");
});

test("robot_uri is omitted from the wire when not given", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  const promise = client.submitTask("go home");
  assert.equal("robot_uri" in JSON.parse(transport.sent[0]!), false);
  transport.reply(plannedReply("go home"));
  await promise;
});

test("error responses resolve and narrow via isError", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  const promise = client.submitTask("go", { robotUri: "bogus://nowhere" });
  transport.reply({
    task: "go",
    robot_uri: "bogus://nowhere",
    metadata: { status: "error", error: "No adapter registered" },
    source: "orchestration_server",
  });
  const response = await promise;
  assert.ok(isError(response));
  assert.match(response.metadata.error, /No adapter/);
});

test("broadcasts for other clients' tasks do not resolve ours", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  const promise = client.submitTask("my task", { timeoutMs: 200 });
  transport.reply(plannedReply("someone else's task"));
  transport.reply(plannedReply("my task"));
  const response = await promise;
  assert.equal(response.task, "my task");
});

test("unknown statuses are ignored for correlation but observable", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  const seen: string[] = [];
  client.onMessage((msg) => seen.push(String(msg.metadata?.status)));

  const promise = client.submitTask("long task");
  transport.reply({
    task: "long task",
    metadata: { status: "in_progress_preview" }, // future minor revision
  });
  transport.reply(plannedReply("long task"));
  const response = await promise;
  assert.ok(isPlanned(response));
  assert.deepEqual(seen, ["in_progress_preview", "planned"]);
});

test("concurrent identical tasks resolve in FIFO order", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  const first = client.submitTask("patrol");
  const second = client.submitTask("patrol");
  transport.reply(plannedReply("patrol"));
  transport.reply({
    task: "patrol",
    metadata: { status: "error", error: "boom" },
    source: "orchestration_server",
  });
  assert.ok(isPlanned(await first));
  assert.ok(isError(await second));
});

test("submitTask times out when the server never answers", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  await assert.rejects(
    client.submitTask("silence", { timeoutMs: 50 }),
    /no response to task within 50 ms/
  );
});

test("in-flight submissions reject when the connection closes", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  const promise = client.submitTask("doomed");
  transport.closeHandler("websocket closed (code 1006)");
  await assert.rejects(promise, /connection closed before response/);
});

test("submitTask after close rejects immediately", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  client.close();
  await assert.rejects(client.submitTask("late"), /client is closed/);
});

test("non-message frames are ignored", async () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  const promise = client.submitTask("robust");
  transport.handler("not json at all");
  transport.handler("[1, 2, 3]");
  transport.handler('{"no_task_field": true}');
  transport.reply(plannedReply("robust"));
  assert.ok(isPlanned(await promise));
});

test("onMessage unsubscribe stops delivery", () => {
  const transport = new FakeTransport();
  const client = new ApyroboClient(transport);

  const seen: string[] = [];
  const unsubscribe = client.onMessage((msg) => seen.push(msg.task));
  transport.reply(plannedReply("one"));
  unsubscribe();
  transport.reply(plannedReply("two"));
  assert.deepEqual(seen, ["one"]);
});
