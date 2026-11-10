import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { McpSession, validate } from '@jmoak/vibrissa';
import { temporaryStore } from './support.mjs';

const terminal = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);

test('weight-space programming works through real MCP and leaves base weights intact', { timeout: 120000 }, async (t) => {
  const store = await temporaryStore();
  let session;
  const diagnostics = [];
  try {
    session = await McpSession.start(store.server, {
      timeoutMs: 15000,
      stderr: 'capture',
      onStderr: (text) => diagnostics.push(text),
    });
    const tools = new Map((await session.listTools()).map((tool) => [tool.name, tool]));
    for (const name of ['capabilities', 'submit_experiment', 'submit_inspection', 'get_inspection', 'get_sample', 'get_run', 'list_runs', 'cancel_run', 'compare_runs']) {
      assert.ok(tools.has(name), `missing required tool ${name}`);
    }

    async function call(name, args = {}) {
      const outcome = await session.callTool(name, args);
      assert.equal(outcome.kind, 'result', `${name}: ${JSON.stringify(outcome)}`);
      const data = outcome.result.structuredContent;
      assert.ok(data && typeof data === 'object', `${name} needs structuredContent`);
      assert.ok(JSON.stringify(data).length <= 12000, `${name} exceeded the advertised response budget`);
      const schema = tools.get(name).outputSchema;
      if (schema) {
        const checked = validate(schema, data);
        assert.equal(checked.ok, true, `${name} violated outputSchema: ${JSON.stringify(checked.errors)}`);
      }
      return data;
    }

    async function waitForRun(runId) {
      const until = Date.now() + 45000;
      let latest;
      while (Date.now() < until) {
        const run = await call('get_run', { run_id: runId });
        latest = run;
        assert.equal(run.id, runId);
        if (terminal.has(run.status)) return run;
        await delay(100);
      }
      t.diagnostic(JSON.stringify({ run_id: runId, status: latest?.status, error: latest?.error }));
      for (const file of ['worker.log', 'progress.json']) {
        const text = await readFile(join(store.root, 'runs', runId, file), 'utf8').catch(() => 'unavailable');
        t.diagnostic(`${file}: ${text.slice(-8000)}`);
      }
      assert.fail(`run ${runId} did not finish within 45 seconds`);
    }

    const inspectionJob = await call('submit_inspection', {
      model: { backend: 'tiny' },
      weight_reads: [{ target: 'output.weight', max_elements: 2 }],
      name: 'Inspect parameters before editing',
    });
    const inspectionRun = await waitForRun(inspectionJob.run_id);
    assert.equal(inspectionRun.status, 'succeeded', JSON.stringify(inspectionRun.error));
    assert.deepEqual(inspectionRun.samples, [], 'inspection must not run forward passes');
    const inspection = await call('get_inspection', { run_id: inspectionJob.run_id });
    assert.ok(inspection.tensors.some((tensor) => tensor.name === 'output.weight' && tensor.kind === 'parameter'));
    assert.equal(inspection.weight_reads.length, 1);
    assert.equal(inspection.weight_reads[0].target, 'output.weight');
    assert.deepEqual(inspection.weight_reads[0].shape, [1, 2]);
    assert.deepEqual(inspection.weight_reads[0].values, [0.5, 0.5]);
    assert.equal(inspection.weight_reads[0].truncated, false);
    const page = await call('get_inspection', {
      run_id: inspectionJob.run_id, limit: 1,
      weight_target: 'output.weight', value_offset: 1, value_limit: 1,
    });
    assert.equal(page.tensors.length, 1);
    assert.equal(page.next_offset, 1);
    assert.equal(page.weight_reads[0].captured_elements, 2);
    assert.deepEqual(page.weight_reads[0].values, [0.5]);
    assert.equal(page.weight_reads[0].more_values, false);

    const cases = [
      { id: 'repair', input: '[2, 4]', expected: '6', scorer: 'numeric', role: 'target', split: 'test' },
      { id: 'preserve-zero', input: '[0, 0]', expected: '0', scorer: 'numeric', role: 'regression', split: 'test' },
    ];
    const baselineSpec = { name: 'MCP weight repair baseline', cases };
    const candidateSpec = {
      name: 'MCP weight repair candidate',
      program: { kind: 'weight_scale', target: 'output.weight', factor: 2 },
      cases,
    };

    const baselineJob = await call('submit_experiment', { spec: baselineSpec, request_id: 'mcp-baseline' });
    assert.equal(typeof baselineJob.run_id, 'string');
    const baseline = await waitForRun(baselineJob.run_id);
    assert.equal(baseline.status, 'succeeded', JSON.stringify(baseline.error));
    assert.deepEqual(baseline.samples.map((sample) => Number(sample.output)), [3, 0]);

    const candidateJob = await call('submit_experiment', { spec: candidateSpec, request_id: 'mcp-weight-edit' });
    const duplicate = await call('submit_experiment', { spec: candidateSpec, request_id: 'mcp-weight-edit' });
    assert.equal(duplicate.run_id, candidateJob.run_id, 'idempotent retries must not create another experiment');
    const candidate = await waitForRun(candidateJob.run_id);
    assert.equal(candidate.status, 'succeeded', JSON.stringify(candidate.error));
    assert.deepEqual(candidate.samples.map((sample) => Number(sample.output)), [6, 0]);
    assert.ok(candidate.samples.every((sample) => sample.status === 'completed' && sample.provenance === 'torch_fixture'));
    for (const sample of candidate.samples) {
      assert.equal(sample.weight_changes.length, 1, 'a direct weight edit needs recorded parameter evidence');
      const change = sample.weight_changes[0];
      assert.equal(change.target, 'output.weight');
      assert.deepEqual(change.shape, [1, 2]);
      assert.equal(change.changed_elements, 2);
      assert.equal(change.total_elements, 2);
      assert.deepEqual(change.before_preview, [0.5, 0.5]);
      assert.deepEqual(change.after_preview, [1, 1]);
      assert.match(change.before_sha256, /^[a-f0-9]{64}$/);
      assert.match(change.after_sha256, /^[a-f0-9]{64}$/);
      assert.notEqual(change.before_sha256, change.after_sha256);
    }
    assert.match(candidate.model_identity.fingerprint, /^[a-f0-9]{64}$/);
    assert.equal(candidate.model_identity.fingerprint, baseline.model_identity.fingerprint);
    assert.equal(candidate.model_identity.fingerprint, inspection.model_identity.fingerprint);

    const conflicting = await session.callTool('submit_experiment', {
      spec: { ...candidateSpec, program: { ...candidateSpec.program, factor: 3 } },
      request_id: 'mcp-weight-edit',
    });
    assert.equal(conflicting.kind, 'error', 'a request ID cannot silently select a different experiment');
    assert.match(conflicting.error.message, /request_conflict/, 'retry conflicts need actionable domain errors');

    const patchJob = await call('submit_experiment', {
      spec: {
        ...candidateSpec,
        name: 'MCP sparse weight patch',
        program: { kind: 'weight_patch', target: 'output.weight', shape: [1, 2], indices: [0], values: [1.5] },
      },
    });
    const patched = await waitForRun(patchJob.run_id);
    assert.equal(patched.status, 'succeeded', JSON.stringify(patched.error));
    assert.deepEqual(patched.samples.map((sample) => Number(sample.output)), [5, 0]);
    assert.equal(patched.samples[0].weight_changes[0].changed_elements, 1);
    assert.deepEqual(patched.samples[0].weight_changes[0].after_preview, [1.5, 0.5]);

    const replayJob = await call('submit_experiment', { spec: baselineSpec, request_id: 'mcp-after-edit' });
    const restored = await waitForRun(replayJob.run_id);
    assert.equal(restored.status, 'succeeded', JSON.stringify(restored.error));
    assert.deepEqual(restored.samples.map((sample) => Number(sample.output)), [3, 0], 'the weight edit leaked into another run');
    assert.equal(restored.model_identity.fingerprint, baseline.model_identity.fingerprint);

    const sample = await call('get_sample', { run_id: candidate.id, case_id: 'repair', output_limit: 1 });
    assert.equal(sample.output, '6');
    assert.equal(sample.weight_changes[0].target, 'output.weight');
    const listed = await call('list_runs', { limit: 2 });
    assert.equal(listed.runs.length, 2);
    assert.equal(listed.runs[0].id, restored.id);

    const comparison = await call('compare_runs', { baseline_id: baseline.id, candidate_id: candidate.id });
    assert.equal(comparison.comparable, true);
    assert.equal(comparison.verdict, 'observed_improvement');
    assert.equal(comparison.accuracy_delta, 0.5);
    assert.equal(comparison.baseline.complete, true);
    assert.equal(comparison.candidate.complete, true);
    assert.equal(comparison.candidate.scope, 'engineering_fixture', 'fixture evidence must not imply real-model capability');
    assert.equal(comparison.candidate.summary.accuracy, 1);
    assert.equal(comparison.candidate.targets.passed, 1);
    assert.equal(comparison.candidate.regressions.passed, 1);
    assert.deepEqual(comparison.improved_cases, ['repair']);
    assert.deepEqual(comparison.unchanged_cases, ['preserve-zero']);
    assert.deepEqual(comparison.regressed_cases, []);
    const cancelledTerminal = await call('cancel_run', { run_id: candidate.id });
    assert.equal(cancelledTerminal.run_id, candidate.id);
    assert.equal(cancelledTerminal.status, 'succeeded', 'cancelling an already completed run cannot rewrite its outcome');

    const invalidJob = await session.callTool('submit_experiment', {
      spec: { ...candidateSpec, name: 'invalid target', program: { kind: 'weight_scale', target: 'missing.weight', factor: 2 } },
    });
    if (invalidJob.kind === 'result') {
      const failed = await waitForRun(invalidJob.result.structuredContent.run_id);
      assert.equal(failed.status, 'failed', 'an unavailable target cannot fall back to a successful identity run');
      assert.ok(failed.error || failed.samples.some((sample) => sample.error), 'failed experiment needs explicit error evidence');
    } else {
      assert.equal(invalidJob.kind, 'error', 'invalid target must be rejected or produce a failed run');
    }

    const largeJob = await call('submit_experiment', {
      spec: {
        name: 'Bounded record retrieval',
        cases: [{ id: 'large-input', input: `[${' '.repeat(16000)}2,4]`, expected: '3', scorer: 'numeric' }],
      },
    });
    const large = await waitForRun(largeJob.run_id);
    assert.equal(large.status, 'succeeded', JSON.stringify(large.error));
    assert.equal(large.truncated, true, 'a large specification must not flood the tool response');
    assert.equal(large.artifact, `nrly://runs/${largeJob.run_id}`);
    const largeSample = await call('get_sample', { run_id: largeJob.run_id, case_id: 'large-input' });
    assert.equal(largeSample.output, '3', 'bounded fallback must preserve access to useful results');
    assert.equal(session.stdoutNoise.lines, 0, 'server printed non-protocol output on stdout');
  } catch (error) {
    if (diagnostics.length) t.diagnostic(diagnostics.join('').slice(-8000));
    throw error;
  } finally {
    await session?.close();
    await store.cleanup();
  }
});
