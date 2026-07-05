/**
 * Transports — the two ways spec 1.0 moves messages (wire-protocol.md §2).
 *
 * A transport moves opaque text frames/lines; framing and JSON handling live
 * in the client. Implement this interface to add a custom transport (e.g. a
 * test double or a tunneled connection).
 */

export interface Transport {
  /** Send one message as text. The transport adds framing (e.g. "\n"). */
  send(text: string): void;
  /** Register the single consumer of incoming frames/lines. */
  onMessage(handler: (text: string) => void): void;
  /** Register a handler for connection/process termination. */
  onClose(handler: (reason: string) => void): void;
  /** Close the connection (or terminate the spawned server). */
  close(): void;
}

/**
 * WebSocket transport (§2.2). Works in browsers and in Node >= 21 via the
 * global WebSocket; pass a constructor (e.g. from the "ws" package) for
 * older Node versions.
 */
export class WebSocketTransport implements Transport {
  private handler: (text: string) => void = () => {};
  private closeHandler: (reason: string) => void = () => {};

  private constructor(private readonly socket: WebSocket) {
    socket.addEventListener("message", (event: MessageEvent) => {
      if (typeof event.data === "string") {
        this.handler(event.data);
      }
    });
    socket.addEventListener("close", (event: CloseEvent) => {
      this.closeHandler(`websocket closed (code ${event.code})`);
    });
  }

  static connect(
    url: string,
    options?: { webSocket?: new (url: string) => WebSocket }
  ): Promise<WebSocketTransport> {
    const WebSocketImpl =
      options?.webSocket ??
      (globalThis as { WebSocket?: new (url: string) => WebSocket }).WebSocket;
    if (WebSocketImpl === undefined) {
      throw new Error(
        "No WebSocket implementation available. Use Node >= 21, a browser, " +
          "or pass one via options.webSocket."
      );
    }
    const socket = new WebSocketImpl(url);
    return new Promise((resolve, reject) => {
      socket.addEventListener("open", () =>
        resolve(new WebSocketTransport(socket))
      );
      socket.addEventListener("error", () =>
        reject(new Error(`could not connect to ${url}`))
      );
    });
  }

  send(text: string): void {
    this.socket.send(text);
  }

  onMessage(handler: (text: string) => void): void {
    this.handler = handler;
  }

  onClose(handler: (reason: string) => void): void {
    this.closeHandler = handler;
  }

  close(): void {
    this.socket.close();
  }
}

/**
 * stdio transport (§2.1): spawns a server command (e.g. `apyrobo serve`)
 * and speaks NDJSON over its stdin/stdout. Node-only.
 */
export class StdioTransport implements Transport {
  private handler: (text: string) => void = () => {};
  private closeHandler: (reason: string) => void = () => {};
  private buffer = "";

  // child_process.ChildProcess, typed loosely to keep the module loadable
  // in browsers (the dynamic import only happens inside spawn()).
  private constructor(private readonly child: {
    stdin: { write(chunk: string): void; end(): void } | null;
    stdout: { on(event: "data", cb: (chunk: Buffer) => void): void } | null;
    on(event: "exit", cb: (code: number | null) => void): void;
    kill(): void;
  }) {
    child.stdout?.on("data", (chunk) => this.feed(chunk.toString("utf-8")));
    child.on("exit", (code) => {
      this.closeHandler(`server process exited (code ${code})`);
    });
  }

  static async spawn(
    command: string,
    args: string[] = []
  ): Promise<StdioTransport> {
    const { spawn } = await import("node:child_process");
    const child = spawn(command, args, {
      stdio: ["pipe", "pipe", "ignore"],
    });
    return new Promise((resolve, reject) => {
      child.once("error", (err) =>
        reject(new Error(`could not spawn ${command}: ${err.message}`))
      );
      child.once("spawn", () => resolve(new StdioTransport(child)));
    });
  }

  /** Split incoming chunks into "\n"-terminated NDJSON lines. */
  private feed(chunk: string): void {
    this.buffer += chunk;
    let newline: number;
    while ((newline = this.buffer.indexOf("\n")) !== -1) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (line !== "") {
        this.handler(line);
      }
    }
  }

  send(text: string): void {
    if (this.child.stdin === null) {
      throw new Error("server stdin is not available");
    }
    this.child.stdin.write(text + "\n");
  }

  onMessage(handler: (text: string) => void): void {
    this.handler = handler;
  }

  onClose(handler: (reason: string) => void): void {
    this.closeHandler = handler;
  }

  close(): void {
    // EOF on stdin signals shutdown per §2.1; kill as a fallback.
    try {
      this.child.stdin?.end();
    } finally {
      this.child.kill();
    }
  }
}
