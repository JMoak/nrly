import { spawn } from 'node:child_process';
import { join } from 'node:path';
import { python, repo, temporaryStore } from './support.mjs';

const store = await temporaryStore();
try {
  const cli = join(repo, 'node_modules', '@jmoak', 'vibrissa', 'bin', 'vib-test.js');
  const child = spawn(process.execPath, [cli, ...process.argv.slice(2), '--config', join(repo, 'vibrissa.json')], {
    cwd: repo,
    stdio: 'inherit',
    env: { ...process.env, NRLY_TEST_PYTHON: python, NRLY_TEST_ROOT: store.root },
  });
  process.exitCode = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', (code, signal) => resolve(code ?? (signal ? 130 : 1)));
  });
} finally {
  await store.cleanup();
}
