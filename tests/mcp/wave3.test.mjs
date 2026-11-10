import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import test from 'node:test';
import { McpSession } from '@jmoak/vibrissa';
import { NrlyClient, vibrissaTransport } from '../../packages/client/dist/index.js';
import { python, repo, temporaryStore } from './support.mjs';

test('real local transformer emits teacher-forced diagnostics and activation pages over MCP', { timeout: 120000 }, async () => {
  const store = await temporaryStore();
  let session;
  try {
    const prepared = execFileSync(python, [join(repo, 'tests/mcp/provision_diagnostics.py'), store.root],
      { cwd: repo, encoding: 'utf8', timeout: 45000 });
    const model = JSON.parse(prepared);
    session = await McpSession.start(store.server, { timeoutMs: 15000, stderr: 'capture' });
    const client = new NrlyClient(vibrissaTransport(session));
    const run = await client.submit({
      schema_version: '0.3', name: 'MCP native teacher forcing', model,
      diagnostics: { top_k: 3, max_tokens: 16 },
      cases: [{ id: 'probe', messages: [{ role: 'user', content: 'hello world' }], continuation: ' yes world' }],
      captures: [{ target: 'transformer.h.0.mlp.c_proj', source: 'input', token_positions: [0, 2], invocation: 'first', max_elements: 64 }],
      budget: { max_capture_bytes: 100000 },
    });
    const state = await run.wait({ timeoutMs: 45000, requireSuccess: true });
    assert.equal(state.status, 'succeeded');
    const sample = await run.sample('probe');
    assert.equal(sample.finish_reason, 'teacher_forced');
    assert.equal(sample.output_tokens, 0);
    assert.equal(sample.diagnostic_tokens_omitted, true);
    const tokens = [];
    for await (const page of run.diagnosticPages('probe', { limit: 1, top_k_limit: 2 })) tokens.push(...page.tokens);
    assert.equal(tokens.length, 2);
    assert.ok(tokens.every((token) => Number.isFinite(token.logprob) && token.logprob <= 0));
    assert.ok(tokens.every((token) => token.top_k.length === 2));
    const captured = await run.capture('probe', 'transformer.h.0.mlp.c_proj');
    assert.equal(captured.capture.source, 'input');
    assert.deepEqual(captured.capture.token_positions, [0, 2]);
    assert.equal(captured.values.length, 64);
    assert.equal(session.stdoutNoise.lines, 0);
  } finally {
    await session?.close();
    await store.cleanup();
  }
});
