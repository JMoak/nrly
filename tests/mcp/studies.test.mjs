import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { join } from 'node:path';
import { promisify } from 'node:util';
import test from 'node:test';
import { McpSession, validate } from '@jmoak/vibrissa';
import { NrlyClient, NrlyClientError, vibrissaTransport } from '../../packages/client/dist/index.js';
import { python, repo, temporaryStore } from './support.mjs';

test('study SDK completes controlled search and sealed finalization without exposing holdout', { timeout: 180000 }, async () => {
  const store = await temporaryStore();
  let session;
  try {
    const provisioned = await promisify(execFile)(python, [join(repo, 'tests/mcp/provision_study.py'), store.root], { cwd: repo, windowsHide: true });
    const { study_id: studyId } = JSON.parse(provisioned.stdout);
    session = await McpSession.start(store.server, { timeoutMs: 15000, stderr: 'capture' });
    const catalog = new Map((await session.listTools()).map((tool) => [tool.name, tool]));
    assert.equal(catalog.has('create_study'), false, 'the optimizer must not supply or read sealed cases');
    const base = vibrissaTransport(session);
    const responses = [];
    const client = new NrlyClient({
      async call(name, args, signal) {
        const schema = catalog.get(name);
        assert.ok(schema, `missing tool ${name}`);
        const checked = validate(schema.inputSchema, args);
        assert.equal(checked.ok, true, JSON.stringify(checked.errors));
        const result = await base.call(name, args, signal);
        responses.push(result);
        return result;
      },
    });
    const study = client.study(studyId);
    assert.equal((await study.get()).status, 'open');
    assert.equal((await client.listStudies())[0].id, studyId);
    const mutableTargets = [];
    for await (const page of study.targetPages({ target_limit: 1 })) mutableTargets.push(...page.mutable_targets);
    assert.deepEqual(mutableTargets, ['output.weight']);
    const cases = [];
    for await (const page of study.casePages({ limit: 1 })) cases.push(...page.cases);
    assert.deepEqual(cases.map((item) => item.id), ['repair', 'preserve']);
    const completeCase = await study.caseSpec('repair');
    assert.equal(completeCase.id, 'repair');
    assert.equal(completeCase.expected, '66');
    await assert.rejects(study.caseSpec('sealed-target-canary'), (error) => error instanceof NrlyClientError && error.kind === 'domain');
    const candidate = { name: 'Repair output weights', program: { kind: 'weight_scale', target: 'output.weight', factor: 2 } };
    const attempt = await study.submit(candidate, 'sdk-study-candidate');
    assert.equal((await study.submit(candidate, 'sdk-study-candidate')).id, attempt.id);
    const completed = await attempt.wait({ timeoutMs: 90000, requireSuccess: true });
    assert.equal(completed.assessment.verdict, 'observed_improvement', JSON.stringify(completed.assessment));
    assert.equal(completed.assessment.scope, 'engineering_fixture');
    assert.equal(completed.assessment.recommendation, 'candidate_supported');
    assert.equal((await attempt.assessment()).verdict, 'observed_improvement');
    assert.equal((await study.attempts({ limit: 1 })).attempts[0].id, attempt.id);
    const development = [];
    for await (const page of attempt.samplePages('candidate', { limit: 1 })) development.push(...page.samples);
    assert.deepEqual(development.map((sample) => sample.output), ['66', '0']);
    let fullOutput = '';
    for await (const page of attempt.sampleOutputPages('candidate', 0, { output_limit: 1 })) fullOutput += page.samples[0].output;
    assert.equal(fullOutput, '66');
    await assert.rejects(study.submit({ ...candidate, name: 'conflict' }, 'sdk-study-candidate'), (error) => error instanceof NrlyClientError && error.kind === 'domain');
    await study.finalize(attempt.id);
    const finalized = await study.waitFinal({ timeoutMs: 90000, requireSuccess: true });
    assert.equal(finalized.status, 'completed');
    assert.equal(finalized.final_result.verdict, 'observed_improvement', JSON.stringify(finalized.final_result));
    assert.equal(finalized.final_result.recommendation, 'candidate_supported');
    assert.equal((await study.finalAssessment()).verdict, 'observed_improvement');
    assert.equal(finalized.nominee_attempt_id, attempt.id);
    assert.equal(finalized.holdout.count, 2);
    assert.equal(finalized.usage.attempts, 1);
    assert.deepEqual(await client.listRuns(), [], 'private study arms must not appear in public run enumeration');
    const raw = await session.callTool('get_study_samples', { study_id: studyId, attempt_id: attempt.id, arm: 'final_candidate' });
    assert.equal(raw.kind, 'error', 'development samples cannot read a sealed final arm');
    for (const response of responses) {
      const text = JSON.stringify(response);
      assert.doesNotMatch(text, /sealed-target-canary|sealed-regression-canary|900001|900003|1800004|private_runs/);
      assert.ok(text.length <= 12000);
    }
    assert.equal(session.stdoutNoise.lines, 0);
  } finally {
    await session?.close();
    await store.cleanup();
  }
});
