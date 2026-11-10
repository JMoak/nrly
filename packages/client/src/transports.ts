import { domainError, NrlyClientError, protocolError } from "./errors.js";

export interface NrlyTransport {
  call(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown>;
}

export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

export interface McpCaller {
  callTool(name: string, args: Record<string, JsonValue>): Promise<unknown>;
}

export interface KeyworkConnection {
  callTool(name: string, args: unknown): Promise<{ text: string; isError: boolean }>;
}

export function keyworkTransport(connection: KeyworkConnection): NrlyTransport {
  return {
    async call(name, args) {
      const result = await connection.callTool(name, args);
      if (result.isError) throw domainError(name, result.text);
      return parseText(name, result.text);
    },
  };
}

export function mcpTransport(caller: McpCaller): NrlyTransport {
  return {
    async call(name, args) {
      return decodeMcpResult(name, await caller.callTool(name, jsonArguments(args)));
    },
  };
}

export function vibrissaTransport(session: McpCaller): NrlyTransport {
  return {
    async call(name, args) {
      const outcome = record(await session.callTool(name, jsonArguments(args)));
      if (outcome?.kind === "error") {
        const error = record(outcome.error);
        const message = typeof error?.message === "string" ? error.message : "MCP tool failed";
        if (error?.kind === "transport" || error?.kind === "timeout" || error?.kind === "protocol") {
          const code = typeof error.codeName === "string" ? error.codeName : `mcp_${error.kind}`;
          throw new NrlyClientError(error.kind, code, message, { tool: name, cause: error });
        }
        throw domainError(name, message);
      }
      if (outcome?.kind !== "result") throw protocolError(name, "expected a completed MCP tool response");
      return decodeMcpResult(name, outcome.result);
    },
  };
}

function decodeMcpResult(name: string, value: unknown): unknown {
  const result = record(value);
  if (!result) throw protocolError(name, "expected an MCP result object");
  const text = Array.isArray(result.content)
    ? result.content.flatMap((block: unknown) => {
      const item = record(block);
      return item?.type === "text" && typeof item.text === "string" ? [item.text] : [];
    }).join("\n")
    : "";
  if (result.isError === true) throw domainError(name, text || "MCP tool failed");
  if (result.structuredContent !== undefined) return result.structuredContent;
  return parseText(name, text);
}

function parseText(name: string, text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw protocolError(name, "expected JSON text; the transport may have truncated the response");
  }
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown> : undefined;
}

function jsonArguments(args: Record<string, unknown>): Record<string, JsonValue> {
  const serialized = JSON.stringify(args, (_key: string, value: unknown) => {
    if (typeof value === "number" && !Number.isFinite(value)) throw new TypeError("MCP arguments must contain finite numbers");
    if (["bigint", "function", "symbol"].includes(typeof value)) throw new TypeError("MCP arguments must be JSON values");
    return value;
  });
  return JSON.parse(serialized) as Record<string, JsonValue>;
}
