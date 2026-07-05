/**
 * ApyroboClient — submit tasks to an APYROBO orchestration server and
 * receive planning results over any Transport (spec/wire-protocol.md).
 */

import {
  OrchestrationMessage,
  TaskResponse,
  parseMessage,
} from "./messages.js";
import { StdioTransport, Transport, WebSocketTransport } from "./transport.js";

export interface SubmitOptions {
  /** Target robot as `scheme://name`; the server defaults it when omitted. */
  robotUri?: string;
  /** Reject the returned promise after this many milliseconds (default 30000). */
  timeoutMs?: number;
}

interface Pending {
  resolve: (response: TaskResponse) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class ApyroboClient {
  private readonly pending = new Map<string, Pending[]>();
  private readonly listeners = new Set<(msg: OrchestrationMessage) => void>();
  private closed: string | null = null;
  private readonly rejectOnClose = new Set<(err: Error) => void>();

  constructor(private readonly transport: Transport) {
    transport.onMessage((text) => this.dispatch(text));
    transport.onClose((reason) => this.handleClose(reason));
  }

  /** Connect to a server over WebSocket, e.g. `ws://localhost:8765`. */
  static async connect(
    url: string,
    options?: { webSocket?: new (url: string) => WebSocket }
  ): Promise<ApyroboClient> {
    return new ApyroboClient(await WebSocketTransport.connect(url, options));
  }

  /** Spawn a server command and speak stdio NDJSON, e.g. `apyrobo serve`. */
  static async spawn(
    command: string,
    args: string[] = []
  ): Promise<ApyroboClient> {
    return new ApyroboClient(await StdioTransport.spawn(command, args));
  }

  /**
   * Submit a task and resolve with the server's response.
   *
   * Responses are correlated by the echoed `task` text (the WebSocket
   * transport broadcasts every response to every client, §2.2), so
   * concurrent submissions of *identical* task text from multiple clients
   * can cross wires — make the text unique when that matters.
   */
  submitTask(task: string, options: SubmitOptions = {}): Promise<TaskResponse> {
    if (this.closed !== null) {
      return Promise.reject(new Error(`client is closed: ${this.closed}`));
    }
    const message: OrchestrationMessage = { task };
    if (options.robotUri !== undefined) {
      message.robot_uri = options.robotUri;
    }
    const timeoutMs = options.timeoutMs ?? 30_000;

    return new Promise<TaskResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.removePending(task, entry);
        this.rejectOnClose.delete(reject);
        reject(new Error(`no response to task within ${timeoutMs} ms`));
      }, timeoutMs);
      const entry: Pending = {
        resolve: (response) => {
          clearTimeout(timer);
          this.rejectOnClose.delete(reject);
          resolve(response);
        },
        timer,
      };
      const queue = this.pending.get(task) ?? [];
      queue.push(entry);
      this.pending.set(task, queue);
      this.rejectOnClose.add(reject);
      try {
        this.transport.send(JSON.stringify(message));
      } catch (err) {
        clearTimeout(timer);
        this.removePending(task, entry);
        this.rejectOnClose.delete(reject);
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  /**
   * Observe every message the server sends, including broadcasts for tasks
   * submitted by other clients. Returns an unsubscribe function.
   */
  onMessage(listener: (msg: OrchestrationMessage) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Close the underlying transport; in-flight submissions reject. */
  close(): void {
    this.transport.close();
    this.handleClose("closed by client");
  }

  private dispatch(text: string): void {
    const message = parseMessage(text);
    if (message === null) {
      return;
    }
    for (const listener of this.listeners) {
      listener(message);
    }
    const status = message.metadata?.status;
    // Spec §3: ignore (don't resolve on) statuses we don't recognize —
    // they are reserved for future revisions like execution progress.
    if (status !== "planned" && status !== "error") {
      return;
    }
    const queue = this.pending.get(message.task);
    const entry = queue?.shift();
    if (entry === undefined) {
      return; // a broadcast for some other client's task
    }
    if (queue !== undefined && queue.length === 0) {
      this.pending.delete(message.task);
    }
    entry.resolve(message as TaskResponse);
  }

  private removePending(task: string, entry: Pending): void {
    const queue = this.pending.get(task);
    if (queue === undefined) {
      return;
    }
    const index = queue.indexOf(entry);
    if (index !== -1) {
      queue.splice(index, 1);
    }
    if (queue.length === 0) {
      this.pending.delete(task);
    }
  }

  private handleClose(reason: string): void {
    if (this.closed !== null) {
      return;
    }
    this.closed = reason;
    for (const queue of this.pending.values()) {
      for (const entry of queue) {
        clearTimeout(entry.timer);
      }
    }
    this.pending.clear();
    for (const reject of this.rejectOnClose) {
      reject(new Error(`connection closed before response: ${reason}`));
    }
    this.rejectOnClose.clear();
  }
}
