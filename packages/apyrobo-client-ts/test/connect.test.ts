/**
 * ApyroboClient.connect must settle: it resolves on open and rejects when
 * the socket closes or errors before opening, or never answers. Uses real
 * TCP servers on 127.0.0.1 and Node's global WebSocket (Node >= 22).
 */
import assert from "node:assert/strict";
import { createServer, Server, Socket } from "node:net";
import { test } from "node:test";

import { ApyroboClient } from "../src/index.js";

const hasWebSocket =
  (globalThis as { WebSocket?: unknown }).WebSocket !== undefined;
const skip = !hasWebSocket && "no global WebSocket in this Node";

async function listen(onSocket: (socket: Socket) => void): Promise<{
  server: Server;
  url: string;
  sockets: Socket[];
}> {
  const sockets: Socket[] = [];
  const server = createServer((socket) => {
    sockets.push(socket);
    onSocket(socket);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address !== null && typeof address === "object");
  return { server, url: `ws://127.0.0.1:${address.port}`, sockets };
}

async function shutdown(s: { server: Server; sockets: Socket[] }) {
  for (const socket of s.sockets) socket.destroy();
  await new Promise<void>((resolve) => s.server.close(() => resolve()));
}

test("connect rejects when the server closes at once", { skip }, async () => {
  const s = await listen((socket) => socket.destroy());
  try {
    await assert.rejects(
      ApyroboClient.connect(s.url, { timeoutMs: 5_000 }),
      /could not connect to ws:\/\/127\.0\.0\.1:\d+/
    );
  } finally {
    await shutdown(s);
  }
});

test("connect rejects when the server never answers", { skip }, async () => {
  // Accepts TCP and reads the handshake but never replies: before the fix
  // this left connect() pending forever.
  const s = await listen(() => {});
  try {
    const started = Date.now();
    await assert.rejects(
      ApyroboClient.connect(s.url, { timeoutMs: 300 }),
      /no answer within 300 ms/
    );
    const waited = Date.now() - started;
    assert.ok(waited >= 250 && waited < 5_000, `waited ${waited} ms`);
  } finally {
    await shutdown(s);
  }
});

test("connect rejects when nothing listens on the port", { skip }, async () => {
  const s = await listen(() => {});
  const url = s.url;
  await shutdown(s);
  await assert.rejects(
    ApyroboClient.connect(url, { timeoutMs: 5_000 }),
    /could not connect to/
  );
});

test("connect refuses a negative or non-finite timeout", async () => {
  for (const timeoutMs of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
    await assert.rejects(
      ApyroboClient.connect("ws://127.0.0.1:1", { timeoutMs }),
      /timeoutMs must be a finite number >= 0/
    );
  }
});

test("connect resolves once the socket opens", async () => {
  // A minimal WebSocket stand-in that opens on the next tick.
  class OpeningSocket extends EventTarget {
    constructor(readonly url: string) {
      super();
      setTimeout(() => this.dispatchEvent(new Event("open")), 0);
    }
    send(): void {}
    close(): void {}
  }
  const client = await ApyroboClient.connect("ws://example.invalid", {
    webSocket: OpeningSocket as unknown as new (url: string) => WebSocket,
    timeoutMs: 1_000,
  });
  assert.ok(client instanceof ApyroboClient);
});

test("a late close after open does not reject connect", async () => {
  class OpenThenClose extends EventTarget {
    constructor(readonly url: string) {
      super();
      setTimeout(() => {
        this.dispatchEvent(new Event("open"));
        const close = new Event("close") as Event & { code: number };
        close.code = 1000;
        this.dispatchEvent(close);
      }, 0);
    }
    send(): void {}
    close(): void {}
  }
  const client = await ApyroboClient.connect("ws://example.invalid", {
    webSocket: OpenThenClose as unknown as new (url: string) => WebSocket,
    timeoutMs: 1_000,
  });
  assert.ok(client instanceof ApyroboClient);
});
