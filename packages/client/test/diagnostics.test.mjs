import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { NrlyClient, NrlyClientError } from '../dist/index.js';

test('diagnostic token pages preserve identity, alternatives and cursors', async () => {
  const requests = [];
  const client = new NrlyClient({ async call(name, args) {
    requests.push([name, args]);
    return {
      run_id: args.run_id, case_id: args.case_id, offset: args.offset ?? 0,
      total: 2, next_offset: args.offset ? null : 1,
      diagnostics: { mode: 'teacher_forced' },
      evidence_kind: 'teacher_forced_diagnostic_not_task_accuracy',
      tokens: [{ position: args.offset + 3, token_id: 7, logprob: -1,
        top_k: [], top_k_offset: 0, top_k_total: 0, next_top_k_offset: null }],
    };
  } });
  const positions = [];
  for await (const page of client.run('run-1').diagnosticPages('case-1', { limit: 1 })) {
    positions.push(...page.tokens.map((token) => token.position));
  }
  assert.deepEqual(positions, [3, 4]);
  assert.equal(requests.length, 2);
  await assert.rejects(client.getDiagnostics('run-1', 'case-1', { limit: 9 }), /limit/);
});

test('diagnostic and capture pages reject cross-run or cross-target evidence', async () => {
  const bad = new NrlyClient({ async call() { return { run_id: 'other', case_id: 'x' }; } });
  await assert.rejects(bad.getDiagnostics('run-1', 'x'), (error) => error instanceof NrlyClientError && error.kind === 'protocol');
  const capture = new NrlyClient({ async call() {
    return { run_id: 'run-1', case_id: 'x', capture: { target: 'wrong' }, values: [], offset: 0, total: 0, next_offset: null };
  } });
  await assert.rejects(capture.getCapture('run-1', 'x', 'mlp'), /target/);
});

test('complete assessments reconstruct Unicode JSON and reject changes between pages', async () => {
  const assessment = { verdict: 'no_change', arms: [], comparisons: [], issues: ['🧠'.repeat(4500)] };
  const chars = Array.from(JSON.stringify(assessment));
  const digest = createHash('sha256').update(chars.join('')).digest('hex');
  const client = new NrlyClient({ async call(name, args) {
    assert.equal(name, 'get_study_assessment');
    return { study_id: args.study_id, attempt_id: args.attempt_id,
      sha256: digest, offset: args.offset, total: chars.length,
      content: chars.slice(args.offset, args.offset + 4000).join(''),
      next_offset: args.offset + 4000 < chars.length ? args.offset + 4000 : null };
  } });
  assert.deepEqual(await client.study('study-1').attempt('attempt-1').assessment(), assessment);
  const changed = new NrlyClient({ async call(name, args) {
    return { study_id: args.study_id, attempt_id: args.attempt_id,
      sha256: (args.offset ? 'b' : 'a').repeat(64), offset: args.offset, total: chars.length,
      content: chars.slice(args.offset, args.offset + 4000).join(''),
      next_offset: args.offset + 4000 < chars.length ? args.offset + 4000 : null };
  } });
  await assert.rejects(changed.study('study-1').finalAssessment(), /changed evidence/);
  const corrupted = new NrlyClient({ async call(name, args) {
    return { study_id: args.study_id, attempt_id: args.attempt_id,
      sha256: digest, offset: args.offset, total: chars.length,
      content: chars.slice(args.offset, args.offset + 4000).join('').replace('no_change', 'regressed'),
      next_offset: args.offset + 4000 < chars.length ? args.offset + 4000 : null };
  } });
  await assert.rejects(corrupted.study('study-1').finalAssessment(), /checksum mismatch/);
});

test('requireSuccess rejects failed final evaluation while retaining valid negative results', async () => {
  const failed = new NrlyClient({ async call() {
    return { id: 'study-1', status: 'completed', final_error_code: 'study_execution_failed', final_result: { verdict: 'inconclusive' } };
  } });
  await assert.rejects(failed.study('study-1').waitFinal({ requireSuccess: true }),
    (error) => error instanceof NrlyClientError && error.kind === 'execution' && error.code === 'study_execution_failed');
  const negative = new NrlyClient({ async call() {
    return { id: 'study-1', status: 'completed', final_error_code: null, final_result: { verdict: 'regressed' } };
  } });
  assert.equal((await negative.study('study-1').waitFinal({ requireSuccess: true })).final_result.verdict, 'regressed');
});
