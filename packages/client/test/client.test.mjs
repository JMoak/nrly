import assert from 'node:assert/strict';
import test from 'node:test';
import {
  NrlyClient, NrlyClientError, keyworkConfig, keyworkTransport, mcpTransport, vibrissaTransport,
} from '../dist/index.js';

function client(call, options) { return new NrlyClient({ call }, options); }
function state(status = 'queued') { return { id: 'run-1', status, error: null }; }
function error(kind, code) { return (value) => value instanceof NrlyClientError && value.kind === kind && value.code === code; }

test('transport adapters preserve structured content and typed domain errors', async () => {
  const data = { run_id: 'run-1', status: 'queued' };
  const spec = { name: 'test', cases: [{ id: 'x', input: '[1,2]' }] };
  const adapters = [
    mcpTransport({ async callTool() { return { structuredContent: data }; } }),
    mcpTransport({ async callTool() { return { content: [{ type: 'text', text: JSON.stringify(data) }] }; } }),
    keyworkTransport({ async callTool() { return { text: JSON.stringify(data), isError: false }; } }),
    vibrissaTransport({ async callTool() { return { kind: 'result', result: { structuredContent: data } }; } }),
  ];
  for (const adapter of adapters) assert.equal((await new NrlyClient(adapter).submit(spec)).id, 'run-1');
  const failures = [
    mcpTransport({ async callTool() { return { isError: true, content: [{ type: 'text', text: 'request_conflict: reused ID' }] }; } }),
    keyworkTransport({ async callTool() { return { isError: true, text: 'Error calling tool: request_conflict: reused ID' }; } }),
    keyworkTransport({ async callTool() { return { isError: true, text: 'Error executing tool submit_experiment: request_conflict: reused ID' }; } }),
    vibrissaTransport({ async callTool() { return { kind: 'error', error: { message: 'request_conflict: reused ID' } }; } }),
  ];
  for (const adapter of failures) await assert.rejects(new NrlyClient(adapter).submit(spec), error('domain', 'request_conflict'));
});

test('malformed, truncated, wrong-identity, and oversized responses are protocol errors', async () => {
  await assert.rejects(new NrlyClient(keyworkTransport({ async callTool() {
    return { text: '{"id": [truncated 5 characters]', isError: false };
  } })).getRun('run-1'), error('protocol', 'invalid_response'));
  await assert.rejects(client(async () => ({ ...state(), id: 'another' })).getRun('run-1'), error('protocol', 'invalid_response'));
  await assert.rejects(client(async () => state('unknown')).getRun('run-1'), error('protocol', 'invalid_response'));
  await assert.rejects(client(async () => ({ ...state(), huge: 'x'.repeat(12001) })).getRun('run-1'), error('protocol', 'invalid_response'));
});

test('transport failures keep their cause and do not become domain failures', async () => {
  const cause = new Error('server exited');
  await assert.rejects(client(async () => { throw cause; }).getRun('run-1'), (value) => {
    assert.equal(value.kind, 'transport');
    assert.equal(value.cause, cause);
    return true;
  });
  for (const kind of ['transport', 'timeout', 'protocol']) {
    const adapter = vibrissaTransport({ async callTool() { return { kind: 'error', error: { kind, codeName: 'SpecificCode', message: 'failed' } }; } });
    await assert.rejects(new NrlyClient(adapter).getRun('run-1'), error(kind, 'SpecificCode'));
  }
  await assert.rejects(client(async () => Promise.reject()).getRun('run-1'), error('transport', 'transport_error'));
});

test('polling preserves failed terminal states and optional success requirement', async () => {
  const statuses = ['queued', 'running', 'failed'];
  const progress = [];
  const sdk = client(async () => state(statuses.shift() ?? 'failed'));
  assert.equal((await sdk.wait('run-1', { pollIntervalMs: 1, onProgress: (run) => progress.push(run.status) })).status, 'failed');
  assert.deepEqual(progress, ['queued', 'running', 'failed']);
  await assert.rejects(sdk.wait('run-1', { requireSuccess: true }), error('execution', 'failed'));
});

test('wait timeout bounds a hung request and does not cancel unless selected', async () => {
  const calls = [];
  const sdk = client(async (name) => { calls.push(name); return new Promise(() => {}); });
  const started = performance.now();
  await assert.rejects(sdk.wait('run-1', { timeoutMs: 20 }), error('timeout', 'wait_timeout'));
  assert.ok(performance.now() - started < 1000);
  assert.deepEqual(calls, ['get_run']);
});

test('cancel-on-timeout submits exactly one cancellation and retains timeout outcome', async () => {
  const calls = [];
  const sdk = client(async (name) => {
    calls.push(name);
    return name === 'cancel_run' ? { run_id: 'run-1', status: 'cancelled' } : state('running');
  });
  await assert.rejects(sdk.wait('run-1', { timeoutMs: 20, pollIntervalMs: 1, cancelOnTimeout: true }), error('timeout', 'wait_timeout'));
  assert.equal(calls.filter((name) => name === 'cancel_run').length, 1);
});

test('abort stops polling; pre-aborted requests do not start transport work', async () => {
  const controller = new AbortController();
  let calls = 0;
  const sdk = client(async () => { calls++; controller.abort('leave'); return state(); });
  await assert.rejects(sdk.wait('run-1', { signal: controller.signal }), error('aborted', 'aborted'));
  assert.equal(calls, 1);
  await assert.rejects(sdk.getRun('run-1', { signal: controller.signal }), error('aborted', 'aborted'));
  assert.equal(calls, 1);
  const immediate = new AbortController();
  const pending = sdk.getRun('run-1', { signal: immediate.signal });
  immediate.abort();
  await assert.rejects(pending, error('aborted', 'aborted'));
  assert.equal(calls, 1, 'abort before dispatch must prevent transport work');
});

test('cancel-on-abort is explicit and cancellation failures remain observable', async () => {
  const controller = new AbortController();
  let cancellations = 0;
  const sdk = client(async (name) => {
    if (name === 'cancel_run') { cancellations++; throw new Error('disconnected'); }
    controller.abort();
    return state();
  });
  await assert.rejects(sdk.wait('run-1', { signal: controller.signal, cancelOnAbort: true }), (value) => {
    assert.equal(value.kind, 'aborted');
    assert.equal(value.cause.cancellation.kind, 'transport');
    return true;
  });
  assert.equal(cancellations, 1);
});

test('request deadlines are distinct from experiment deadlines', async () => {
  const sdk = client(async () => new Promise(() => {}));
  await assert.rejects(sdk.getRun('run-1', { timeoutMs: 10 }), error('timeout', 'request_timeout'));
});

test('bounded inspection pages advance and expose resume cursors', async () => {
  const sdk = client(async (_name, args) => ({
    run_id: args.run_id, status: 'succeeded', tensors: [{ name: `w${args.offset}`, kind: 'parameter' }],
    weight_reads: [], offset: args.offset, limit: args.limit, total: 3,
    next_offset: args.offset < 2 ? args.offset + 1 : null,
  }));
  const pages = [];
  for await (const page of sdk.run('run-1').inspectionPages({ limit: 1 })) pages.push(page);
  assert.deepEqual(pages.map((page) => page.offset), [0, 1, 2]);
  const bounded = sdk.inspectionPages('run-1', { limit: 1 }, { maxPages: 1 });
  assert.equal((await bounded.next()).value.next_offset, 1);
  await assert.rejects(bounded.next(), error('protocol', 'page_limit'));
});

test('inspection pages reject stalled cursors and invalid local page bounds', async () => {
  const sdk = client(async () => ({ run_id: 'run-1', tensors: [], weight_reads: [], offset: 0, next_offset: 0 }));
  await assert.rejects(sdk.inspectionPages('run-1').next(), error('protocol', 'invalid_response'));
  await assert.rejects(sdk.getInspection('run-1', { limit: 41 }), RangeError);
});

test('output pagination uses Python Unicode code points, not UTF-16 units', async () => {
  const offsets = [];
  const sdk = client(async (_name, args) => {
    offsets.push(args.output_offset);
    return { case_id: 'case', output: args.output_offset === 0 ? '🌿' : 'x', output_offset: args.output_offset, more_output: args.output_offset === 0 };
  });
  let result = '';
  for await (const page of sdk.run('run-1').sampleOutputPages('case', { output_limit: 1 })) result += page.output;
  assert.equal(result, '🌿x');
  assert.deepEqual(offsets, [0, 1]);
});

test('Keywork configuration is portable, explicit, and does not enable trust', () => {
  const config = keyworkConfig({ python: 'C:/nrly/.venv/Scripts/python.exe', root: 'C:/experiments', modelCache: 'C:/models' });
  assert.deepEqual(config.mcpServers.nrly.args, ['-m', 'nrly.mcp', '--root', 'C:/experiments', '--model-cache', 'C:/models']);
  assert.equal('trusted' in config.mcpServers.nrly, false);
  assert.equal('cwd' in config.mcpServers.nrly, false);
  assert.throws(() => keyworkConfig({ python: 'python', root: './runs' }), /absolute path/);
});

test('study handles preserve namespaces and reject wrong attempt identities', async () => {
  const sdk = client(async (name) => {
    if (name === 'submit_study_candidate') return { study_id: 'study-1', attempt_id: 'attempt-1', status: 'queued' };
    return { study_id: 'study-2', attempt_id: 'attempt-1', status: 'succeeded' };
  });
  const attempt = await sdk.study('study-1').submit({ name: 'x', program: { kind: 'identity' } });
  assert.equal(attempt.id, 'attempt-1');
  assert.equal(attempt.studyId, 'study-1');
  await assert.rejects(attempt.get(), error('protocol', 'invalid_response'));
});

test('study finalization polls completed as lifecycle state without interpreting assessment', async () => {
  const statuses = ['finalizing', 'completed'];
  const sdk = client(async () => ({ id: 'study-1', status: statuses.shift(), final_result: { verdict: 'inconclusive', recommendation: 'retain_original' } }));
  const result = await sdk.study('study-1').waitFinal({ pollIntervalMs: 1, requireSuccess: true });
  assert.equal(result.status, 'completed');
  assert.equal(result.final_result.recommendation, 'retain_original');
});

test('study final cancellation is an explicit timeout policy', async () => {
  let cancelled = false;
  const sdk = client(async (name) => {
    if (name === 'cancel_study_final') { cancelled = true; return { study_id: 'study-1', status: 'interrupted' }; }
    return { id: 'study-1', status: 'finalizing' };
  });
  await assert.rejects(sdk.study('study-1').waitFinal({ timeoutMs: 10, pollIntervalMs: 1, cancelOnTimeout: true }), error('timeout', 'wait_timeout'));
  assert.equal(cancelled, true);
});

test('all registered mutable targets remain accessible through bounded study pages', async () => {
  const names = Array.from({ length: 19 }, (_, index) => `layer${index}.weight`);
  const sdk = client(async (_name, args) => ({
    id: 'study-1', status: 'open', mutable_targets: names.slice(args.target_offset, args.target_offset + args.target_limit),
    mutable_target_count: names.length, mutable_target_offset: args.target_offset,
    next_target_offset: args.target_offset + args.target_limit < names.length ? args.target_offset + args.target_limit : null,
  }));
  const read = [];
  for await (const page of sdk.study('study-1').targetPages({ target_limit: 3 })) read.push(...page.mutable_targets);
  assert.deepEqual(read, names);
});

test('development output pagination reconstructs Unicode using Python offsets', async () => {
  const output = Array.from('a🌿b');
  const offsets = [];
  const sdk = client(async (_name, args) => {
    offsets.push(args.output_offset);
    return {
      study_id: 'study-1', attempt_id: 'attempt-1', arm: 'candidate', offset: 0, next_offset: null,
      samples: [{ output: output[args.output_offset], output_offset: args.output_offset, more_output: args.output_offset < 2 }],
    };
  });
  let collected = '';
  for await (const page of sdk.study('study-1').attempt('attempt-1').sampleOutputPages('candidate', 0, { output_limit: 1 })) collected += page.samples[0].output;
  assert.equal(collected, 'a🌿b');
  assert.deepEqual(offsets, [0, 1, 2]);
});
