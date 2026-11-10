import { mkdtemp, realpath, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
export const python = join(repo, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

export async function temporaryStore() {
  if (!existsSync(python)) throw new Error(`Python environment missing: ${python}. Run uv sync first.`);
  const parent = await realpath(tmpdir());
  const root = await mkdtemp(join(parent, 'nrly-mcp-'));
  return {
    root,
    server: {
      cmd: python,
      args: ['-m', 'nrly.mcp', '--root', root],
      cwd: repo,
      protocol: 'auto',
      spawnTimeoutMs: 15000,
    },
    async cleanup() {
      // Verify the resolved target before recursively removing only our own temp directory.
      const actual = await realpath(root);
      const child = relative(parent, actual);
      if (isAbsolute(child) || child.startsWith('..') || child.includes('/') || child.includes('\\') || !child.startsWith('nrly-mcp-')) {
        throw new Error(`Refusing to remove unexpected test directory: ${actual}`);
      }
      await rm(actual, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    },
  };
}
