export interface NrlyServerOptions {
  python: string;
  root: string;
  modelCache?: string;
  env?: Record<string, string>;
}

export interface KeyworkNrlyServer {
  transport: "stdio";
  command: string;
  args: string[];
  env?: Record<string, string>;
}

export function keyworkServer(options: NrlyServerOptions): KeyworkNrlyServer {
  absolutePath("python", options.python);
  absolutePath("root", options.root);
  const args = ["-m", "nrly.mcp", "--root", options.root];
  if (options.modelCache !== undefined) {
    absolutePath("modelCache", options.modelCache);
    args.push("--model-cache", options.modelCache);
  }
  return {
    transport: "stdio", command: options.python, args,
    ...(options.env !== undefined ? { env: { ...options.env } } : {}),
  };
}

export function keyworkConfig(options: NrlyServerOptions): { mcpServers: { nrly: KeyworkNrlyServer } } {
  return { mcpServers: { nrly: keyworkServer(options) } };
}

function absolutePath(field: string, value: string): void {
  if (!value || /[\x00\r\n]/.test(value) || !(/^[A-Za-z]:[\\/]/.test(value) || value.startsWith("/") || /^\\\\[^\\]+\\[^\\]+/.test(value))) {
    throw new TypeError(`${field} must be an absolute path; Keywork stdio configuration has no cwd field`);
  }
}
