import type { NrlyClient } from "./client.js";
import { integer, object } from "./client.js";
import { NrlyClientError, protocolError } from "./errors.js";
import type { PageOptions } from "./types.js";

/** Reassemble bounded immutable JSON records and verify their advertised UTF-8 digest. */
export async function readJsonRecord(
  client: NrlyClient, tool: string, identity: Record<string, string | null>, options: PageOptions,
): Promise<Record<string, unknown>> {
  const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
  let offset = 0;
  let content = "";
  let digest: string | undefined;
  let total: number | undefined;
  for (let count = 0; count < maximum; count++) {
    const result = await client.request(tool, { ...identity, offset, limit: 4000 }, options);
    if (Object.entries(identity).some(([key, value]) => result[key] !== value)
      || result.offset !== offset || typeof result.content !== "string"
      || Array.from(result.content).length > 4000 || !Number.isSafeInteger(result.total)
      || typeof result.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(result.sha256)
      || (digest !== undefined && result.sha256 !== digest) || (total !== undefined && result.total !== total)) {
      throw protocolError(tool, "invalid or changed evidence page");
    }
    digest = result.sha256;
    total = result.total as number;
    content += result.content;
    const next = offset + Array.from(result.content).length;
    if (result.next_offset === null) {
      if (next !== total) throw protocolError(tool, "incomplete evidence record");
      const bytes = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(content)));
      const actual = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
      if (actual !== digest) throw protocolError(tool, "evidence checksum mismatch");
      try { return object(tool, JSON.parse(content)); }
      catch { throw protocolError(tool, "malformed evidence JSON"); }
    }
    if (!result.content.length || result.next_offset !== next) throw protocolError(tool, "evidence cursor did not advance");
    offset = next;
  }
  throw new NrlyClientError("protocol", "page_limit", `Evidence exceeded ${maximum} pages`, identity.study_id ? { resourceId: identity.study_id } : {});
}
