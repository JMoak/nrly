import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { NrlyClient, NrlyClientError, keyworkConfig, keyworkTransport } from '../../packages/client/dist/index.js';
import { python, temporaryStore } from './support.mjs';

if (!process.env.KEYWORK_REPO) throw new Error('Set KEYWORK_REPO to the existing Keywork checkout for this optional connector acceptance test.');
const source = resolve(process.env.KEYWORK_REPO);
const [{ connectStdioServer }, { configSchema }] = await Promise.all([
  import(pathToFileURL(resolve(source, 'packages/engine/src/mcp/client.ts')).href),
  import(pathToFileURL(resolve(source, 'packages/shared/src/config/schema.ts')).href),
]);
const store = await temporaryStore();
let connection;
try {
  const fragment = keyworkConfig({ python, root: store.root });
  const accepted = configSchema.parse(fragment);
  const server = accepted.mcpServers.nrly;
  connection = await connectStdioServer(server, { requestTimeoutMs: 10000 });
  const client = new NrlyClient(keyworkTransport(connection));
  assert.equal((await client.capabilities()).name, 'nrly');
  const spec = { name: 'Keywork connector acceptance', cases: [{ id: 'sum', input: '[2,4]', expected: '3', scorer: 'numeric' }] };
  const submitted = await client.submit(spec, 'keywork-connector');
  const result = await submitted.wait({ timeoutMs: 45000, requireSuccess: true });
  assert.equal(result.status, 'succeeded');
  assert.equal((await submitted.sample('sum')).output, '3');
  assert.equal((await submitted.samples({ limit: 1 })).samples[0].case_id, 'sum');
  assert.equal((await client.submit(spec, 'keywork-connector')).id, submitted.id);
  await assert.rejects(client.submit({ ...spec, name: 'conflict' }, 'keywork-connector'), (error) => error instanceof NrlyClientError && error.code === 'request_conflict');
  console.log(JSON.stringify({ ok: true, connector: 'Keywork connectStdioServer', era: connection.era, transport: 'stdio', inference: 'torch_fixture', user_config_modified: false }));
} finally {
  await connection?.close();
  await store.cleanup();
}
