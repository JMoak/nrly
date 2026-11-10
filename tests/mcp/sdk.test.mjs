import assert from 'node:assert/strict';
import test from 'node:test';
import { McpSession, validate } from '@jmoak/vibrissa';
import { NrlyClient, NrlyClientError, vibrissaTransport } from '../../packages/client/dist/index.js';
import { temporaryStore } from './support.mjs';

test('typed SDK runs pinned pipelines, rich reads, bounded sample pages and cancellation through MCP', { timeout: 120000 }, async () => {
  const store = await temporaryStore();
  let session;
  try {
    session = await McpSession.start(store.server, { timeoutMs: 15000, stderr: 'capture' });
    const catalog = new Map((await session.listTools()).map((tool) => [tool.name, tool]));
    const transport = vibrissaTransport(session);
    const checked = {
      async call(name, args, signal) {
        const schema = catalog.get(name);
        assert.ok(schema, `Tool ${name} must be in the actual server catalog`);
        const input = validate(schema.inputSchema, args);
        assert.equal(input.ok, true, `${name} SDK arguments violate the server schema: ${JSON.stringify(input.errors)}`);
        const result = await transport.call(name, args, signal);
        if (schema.outputSchema) assert.equal(validate(schema.outputSchema, result).ok, true);
        return result;
      },
    };
    const client = new NrlyClient(checked);
    const capabilities = await client.capabilities();
    assert.ok(capabilities.weight_operations.includes('weight_slice_scale'));
    const inspected = await client.inspect({ model: { backend: 'tiny' }, weight_reads: [{ target: 'output.weight', coordinates: [[0, 1], [0, 0]], max_elements: 2 }] });
    await inspected.wait({ timeoutMs: 45000, requireSuccess: true });
    const first = await inspected.inspection({ limit: 1, value_limit: 1, weight_target: 'output.weight' });
    assert.deepEqual(first.weight_reads[0].values, [0.5]);
    assert.deepEqual(first.weight_reads[0].flat_indices, [1]);
    assert.deepEqual(first.weight_reads[0].coordinates, [[0, 1]]);
    assert.equal(first.weight_reads[0].more_values, true);
    const second = await inspected.inspection({ limit: 1, value_offset: 1, value_limit: 1, weight_target: 'output.weight' });
    assert.deepEqual(second.weight_reads[0].flat_indices, [0]);
    assert.deepEqual(second.weight_reads[0].coordinates, [[0, 0]]);
    const values = [];
    for await (const page of inspected.weightPages('output.weight', { value_limit: 1 })) values.push(...page.values);
    assert.deepEqual(values, [0.5, 0.5]);
    const tensors = [];
    for await (const page of inspected.inspectionPages({ limit: 1 })) tensors.push(...page.tensors);
    assert.ok(tensors.some((tensor) => tensor.name === 'output.weight'));

    const spec = {
      name: 'SDK pipeline',
      program: {
        kind: 'pipeline', expected_base: first.model_identity.fingerprint,
        operations: [
          { kind: 'weight_sparse_add', target: 'output.weight', shape: [1, 2], indices: [0], values: [0.5] },
          { kind: 'weight_slice_scale', target: 'output.weight', shape: [1, 2], slices: [{ stop: 1 }, { start: 1, stop: 2 }], factor: 2 },
          ...Array.from({ length: 30 }, () => ({ kind: 'weight_scale', target: 'output.weight', factor: 1 })),
        ],
      },
      cases: [
        { id: 'repair', input: `[${' '.repeat(16000)}2,4]`, expected: '6', scorer: 'numeric', role: 'target' },
        { id: 'preserve', input: '[0,0]', expected: '0', scorer: 'numeric', role: 'regression' },
      ],
    };
    const run = await client.submit(spec, 'sdk-idempotency');
    assert.equal((await client.submit(spec, 'sdk-idempotency')).id, run.id);
    const complete = await run.wait({ timeoutMs: 45000, requireSuccess: true });
    assert.equal(complete.truncated, true);
    assert.equal(complete.artifact, run.artifact);
    const samples = [];
    for await (const page of run.samplePages({ limit: 1 })) samples.push(...page.samples);
    assert.deepEqual(samples.map((sample) => sample.case_id), ['repair', 'preserve']);
    const repair = await run.sample('repair');
    assert.equal(repair.output, '6');
    assert.equal(repair.provenance, 'torch_fixture');
    assert.equal(repair.weight_changes.length, 4);
    assert.equal(repair.weight_change_count, 32);
    assert.equal(repair.weight_changes[1].operation_index, 1);
    const changes = [];
    for await (const page of run.sampleChangePages('repair', { change_limit: 4, output_limit: 1 })) changes.push(...page.weight_changes);
    assert.equal(changes.length, 32);
    assert.deepEqual(changes.map((change) => change.operation_index), Array.from({ length: 32 }, (_, index) => index));
    assert.equal(changes.at(-1).net_changed_elements, 2);
    assert.equal((await run.cancel()).status, 'succeeded');
    await assert.rejects(client.submit({ ...spec, name: 'different' }, 'sdk-idempotency'), (error) => error instanceof NrlyClientError && error.code === 'request_conflict');

    const pending = await client.submit({ name: 'SDK cancellation', cases: [{ id: 'cancel', input: '[1,2]' }] });
    const cancellation = await pending.cancel();
    assert.ok(['cancelling', 'cancelled'].includes(cancellation.status));
    assert.equal((await pending.wait({ timeoutMs: 10000 })).status, 'cancelled');
    assert.equal(session.stdoutNoise.lines, 0);
  } finally {
    await session?.close();
    await store.cleanup();
  }
});
