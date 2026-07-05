/**
 * Wire-protocol message types.
 *
 * Mirrors spec/wire-protocol.md and spec/schemas/orchestration-message.schema.json
 * (spec version 1.0-draft). One message shape, both directions.
 */

/** A task planned by the server: one step of the returned skill plan. */
export interface PlannedSkill {
  skill_id: string;
  name: string;
}

/**
 * `metadata.status` on responses. Spec 1.0 defines "planned" and "error";
 * clients MUST ignore messages whose status they do not recognize, so the
 * type stays open to future minor revisions.
 */
export type ResponseStatus = "planned" | "error" | (string & {});

/** A single message on the wire, either direction. Only `task` is required. */
export interface OrchestrationMessage {
  task: string;
  /** Target robot as `scheme://name`. The server defaults it when absent. */
  robot_uri?: string;
  /** Free-form on requests; carries the result on responses. */
  metadata?: Record<string, unknown>;
  /** Sender identifier. The reference server sets "orchestration_server". */
  source?: string;
}

/** A server response whose outcome has been narrowed. */
export interface TaskResponse extends OrchestrationMessage {
  metadata: Record<string, unknown> & { status: ResponseStatus };
}

/** True when the response carries a successful plan. */
export function isPlanned(
  response: TaskResponse
): response is TaskResponse & {
  metadata: { status: "planned"; skills: PlannedSkill[]; count: number };
} {
  return response.metadata.status === "planned";
}

/** True when the response reports a planning/discovery failure. */
export function isError(
  response: TaskResponse
): response is TaskResponse & { metadata: { status: "error"; error: string } } {
  return response.metadata.status === "error";
}

/** Parse one wire frame/line into a message, or null when it isn't one. */
export function parseMessage(text: string): OrchestrationMessage | null {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    return null;
  }
  const record = data as Record<string, unknown>;
  if (typeof record.task !== "string") {
    return null;
  }
  return record as unknown as OrchestrationMessage;
}
