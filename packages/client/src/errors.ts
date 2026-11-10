export type ErrorKind = "domain" | "protocol" | "transport" | "timeout" | "aborted" | "execution";

export class NrlyClientError extends Error {
  readonly kind: ErrorKind;
  readonly code: string;
  readonly tool: string | undefined;
  readonly resourceId: string | undefined;

  constructor(kind: ErrorKind, code: string, message: string, options: {
    tool?: string; resourceId?: string; cause?: unknown;
  } = {}) {
    super(message, { cause: options.cause });
    this.name = "NrlyClientError";
    this.kind = kind;
    this.code = code;
    this.tool = options.tool;
    this.resourceId = options.resourceId;
  }
}

export function domainError(tool: string, message: string): NrlyClientError {
  const detail = message.replace(/^Error (?:executing|calling) tool(?: [^:]+)?:\s*/, "");
  const code = /^([a-z][a-z0-9_]+):/.exec(detail)?.[1] ?? "tool_error";
  return new NrlyClientError("domain", code, message, { tool });
}

export function protocolError(tool: string, detail: string): NrlyClientError {
  return new NrlyClientError("protocol", "invalid_response", `${tool}: ${detail}`, { tool });
}
