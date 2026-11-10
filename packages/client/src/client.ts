import type { ExperimentSpec } from "./contracts.generated.js";
import { NrlyClientError, protocolError } from "./errors.js";
import type { NrlyTransport } from "./transports.js";
import { StudyHandle, listStudies } from "./studies.js";
import type { StudySummary } from "./studies.js";
import type {
  Capabilities, ClientOptions, Comparison, InspectionOptions, InspectionPage, InspectionSubmission,
  OffsetPage, PageOptions, RequestOptions, RunState, RunStatus, RunSummary, SampleOptions, SamplePage, SamplesPage,
  SubmittedRun, WaitOptions, WeightReadPage, DiagnosticOptions, DiagnosticsPage, CapturePage,
} from "./types.js";

const statuses = new Set<RunStatus>(["queued", "running", "cancelling", "succeeded", "failed", "cancelled", "interrupted"]);
const terminal = new Set<RunStatus>(["succeeded", "failed", "cancelled", "interrupted"]);

export class NrlyClient {
  private readonly transport: NrlyTransport;
  private readonly requestTimeoutMs: number;
  private readonly responseCharacterLimit: number;

  constructor(transport: NrlyTransport, options: ClientOptions = {}) {
    this.transport = transport;
    this.requestTimeoutMs = positive("requestTimeoutMs", options.requestTimeoutMs ?? 10_000);
    this.responseCharacterLimit = positive("responseCharacterLimit", options.responseCharacterLimit ?? 12_000);
  }

  async capabilities(options: RequestOptions = {}): Promise<Capabilities> {
    const value = await this.request("capabilities", {}, options);
    stringField("capabilities", value, "schema_version");
    if (!Array.isArray(value.backends)) throw protocolError("capabilities", "missing backends");
    return value as unknown as Capabilities;
  }

  run(runId: string): RunHandle { return new RunHandle(this, identifier(runId)); }
  study(studyId: string): StudyHandle { return new StudyHandle(this, studyId); }
  listStudies(limit = 20, options: RequestOptions = {}): Promise<StudySummary[]> { return listStudies(this, limit, options); }

  async submit(spec: ExperimentSpec, requestId?: string, options: RequestOptions = {}): Promise<RunHandle> {
    const result = submitted(await this.request("submit_experiment", {
      spec, ...(requestId !== undefined ? { request_id: requestId } : {}),
    }, options), "submit_experiment");
    return this.run(result.run_id);
  }

  async inspect(submission: InspectionSubmission, options: RequestOptions = {}): Promise<RunHandle> {
    const result = submitted(await this.request("submit_inspection", { ...submission }, options), "submit_inspection");
    return this.run(result.run_id);
  }

  async getRun(runId: string, options: RequestOptions = {}): Promise<RunState> {
    const result = await this.request("get_run", { run_id: identifier(runId) }, options);
    validateIdentity("get_run", result, "id", runId);
    validateStatus("get_run", result);
    return result as unknown as RunState;
  }

  async cancel(runId: string, options: RequestOptions = {}): Promise<SubmittedRun> {
    const result = submitted(await this.request("cancel_run", { run_id: identifier(runId) }, options), "cancel_run");
    validateIdentity("cancel_run", result, "run_id", runId);
    return result;
  }

  async listRuns(limit = 20, options: RequestOptions = {}): Promise<RunSummary[]> {
    integer("limit", limit, 1, 20);
    const result = await this.request("list_runs", { limit }, options);
    if (!Array.isArray(result.runs) || result.runs.length > limit) throw protocolError("list_runs", "invalid run page");
    for (const value of result.runs) {
      const run = object("list_runs", value);
      stringField("list_runs", run, "id");
      validateStatus("list_runs", run);
    }
    return result.runs as RunSummary[];
  }

  async compare(baselineId: string, candidateId: string, options: RequestOptions = {}): Promise<Comparison> {
    const result = await this.request("compare_runs", {
      baseline_id: identifier(baselineId), candidate_id: identifier(candidateId),
    }, options);
    if (typeof result.comparable !== "boolean" || typeof result.verdict !== "string") {
      throw protocolError("compare_runs", "missing comparison outcome");
    }
    return result as unknown as Comparison;
  }

  async getInspection(runId: string, page: InspectionOptions = {}, options: RequestOptions = {}): Promise<InspectionPage> {
    integer("offset", page.offset ?? 0, 0);
    integer("limit", page.limit ?? 40, 1, 40);
    integer("value_offset", page.value_offset ?? 0, 0);
    integer("value_limit", page.value_limit ?? 32, 1, 64);
    const result = await this.request("get_inspection", { run_id: identifier(runId), ...page }, options);
    validateIdentity("get_inspection", result, "run_id", runId);
    validatePage("get_inspection", result, "tensors", page.offset ?? 0, page.limit ?? 40);
    if (!Array.isArray(result.weight_reads)) throw protocolError("get_inspection", "missing weight reads");
    return result as unknown as InspectionPage;
  }

  async getSample(runId: string, caseId: string, page: SampleOptions = {}, options: RequestOptions = {}): Promise<SamplePage> {
    integer("output_offset", page.output_offset ?? 0, 0);
    integer("output_limit", page.output_limit ?? 2000, 1, 4000);
    integer("change_offset", page.change_offset ?? 0, 0);
    integer("change_limit", page.change_limit ?? 4, 1, 4);
    const result = await this.request("get_sample", { run_id: identifier(runId), case_id: caseId, ...page }, options);
    validateIdentity("get_sample", result, "case_id", caseId);
    if (typeof result.output !== "string" || Array.from(result.output).length > (page.output_limit ?? 2000)
      || result.output_offset !== (page.output_offset ?? 0) || typeof result.more_output !== "boolean") {
      throw protocolError("get_sample", "invalid output page");
    }
    return result as unknown as SamplePage;
  }

  async getSamples(runId: string, page: OffsetPage = {}, options: RequestOptions = {}): Promise<SamplesPage> {
    integer("offset", page.offset ?? 0, 0);
    integer("limit", page.limit ?? 20, 1, 20);
    const result = await this.request("get_samples", { run_id: identifier(runId), ...page }, options);
    validateIdentity("get_samples", result, "run_id", runId);
    validatePage("get_samples", result, "samples", page.offset ?? 0, page.limit ?? 20);
    return result as unknown as SamplesPage;
  }

  async getDiagnostics(runId: string, caseId: string, page: DiagnosticOptions = {}, options: RequestOptions = {}): Promise<DiagnosticsPage> {
    integer("offset", page.offset ?? 0, 0);
    integer("limit", page.limit ?? 8, 1, 8);
    integer("top_k_offset", page.top_k_offset ?? 0, 0);
    integer("top_k_limit", page.top_k_limit ?? 3, 0, 5);
    const result = await this.request("get_diagnostics", { run_id: identifier(runId), case_id: caseId, ...page }, options);
    validateIdentity("get_diagnostics", result, "run_id", runId);
    validateIdentity("get_diagnostics", result, "case_id", caseId);
    validatePage("get_diagnostics", result, "tokens", page.offset ?? 0, page.limit ?? 8);
    for (const value of result.tokens as unknown[]) {
      const token = object("get_diagnostics", value);
      if (!Array.isArray(token.top_k) || token.top_k.length > (page.top_k_limit ?? 3)
        || token.top_k_offset !== (page.top_k_offset ?? 0)) throw protocolError("get_diagnostics", "invalid token alternatives");
    }
    return result as unknown as DiagnosticsPage;
  }

  async *diagnosticPages(runId: string, caseId: string, page: DiagnosticOptions = {}, options: PageOptions = {}): AsyncGenerator<DiagnosticsPage> {
    let offset = page.offset ?? 0;
    const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
    for (let count = 0; count < maximum; count++) {
      const result = await this.getDiagnostics(runId, caseId, { ...page, offset }, options);
      yield result;
      if (result.next_offset === null) return;
      offset = result.next_offset;
    }
    throw new NrlyClientError("protocol", "page_limit", `Diagnostics exceeded ${maximum} pages; resume with next_offset`, { resourceId: runId });
  }

  async getCapture(runId: string, caseId: string, target: string, page: OffsetPage = {}, options: RequestOptions = {}): Promise<CapturePage> {
    integer("offset", page.offset ?? 0, 0);
    integer("limit", page.limit ?? 64, 1, 64);
    const result = await this.request("get_capture", { run_id: identifier(runId), case_id: caseId, target, ...page }, options);
    validateIdentity("get_capture", result, "run_id", runId);
    validateIdentity("get_capture", result, "case_id", caseId);
    validateIdentity("get_capture", object("get_capture", result.capture), "target", target);
    validatePage("get_capture", result, "values", page.offset ?? 0, page.limit ?? 64);
    return result as unknown as CapturePage;
  }

  async *samplePages(runId: string, page: OffsetPage = {}, options: PageOptions = {}): AsyncGenerator<SamplesPage> {
    let offset = page.offset ?? 0;
    const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
    for (let count = 0; count < maximum; count++) {
      const result = await this.getSamples(runId, { ...page, offset }, options);
      yield result;
      if (result.next_offset === null) return;
      offset = result.next_offset;
    }
    throw new NrlyClientError("protocol", "page_limit", `Samples exceeded ${maximum} pages; resume with the last next_offset`, { resourceId: runId });
  }

  async *inspectionPages(runId: string, page: InspectionOptions = {}, options: PageOptions = {}): AsyncGenerator<InspectionPage> {
    let offset = page.offset ?? 0;
    const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
    for (let count = 0; count < maximum; count++) {
      const result = await this.getInspection(runId, { ...page, offset }, options);
      yield result;
      if (result.next_offset === null) return;
      offset = result.next_offset;
    }
    throw new NrlyClientError("protocol", "page_limit", `Inspection exceeded ${maximum} pages; resume with the last next_offset`, { resourceId: runId });
  }

  async *sampleOutputPages(runId: string, caseId: string, page: SampleOptions = {}, options: PageOptions = {}): AsyncGenerator<SamplePage> {
    let outputOffset = page.output_offset ?? 0;
    const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
    for (let count = 0; count < maximum; count++) {
      const result = await this.getSample(runId, caseId, { ...page, output_offset: outputOffset }, options);
      yield result;
      if (!result.more_output) return;
      if (!result.output?.length) throw protocolError("get_sample", "output cursor did not advance");
      outputOffset += Array.from(result.output).length;
    }
    throw new NrlyClientError("protocol", "page_limit", `Sample output exceeded ${maximum} pages; resume with output_offset`, { resourceId: runId });
  }

  async *sampleChangePages(runId: string, caseId: string, page: SampleOptions = {}, options: PageOptions = {}): AsyncGenerator<SamplePage> {
    let changeOffset = page.change_offset ?? 0;
    const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
    for (let count = 0; count < maximum; count++) {
      const result = await this.getSample(runId, caseId, { ...page, change_offset: changeOffset }, options);
      if (result.change_offset !== changeOffset || !Array.isArray(result.weight_changes)
        || result.weight_changes.length > (page.change_limit ?? 4)
        || (result.next_change_offset !== null && (!Number.isSafeInteger(result.next_change_offset) || result.next_change_offset <= changeOffset))) {
        throw protocolError("get_sample", "invalid weight-change pagination");
      }
      yield result;
      if (result.next_change_offset === null) return;
      changeOffset = result.next_change_offset;
    }
    throw new NrlyClientError("protocol", "page_limit", `Weight changes exceeded ${maximum} pages; resume with next_change_offset`, { resourceId: runId });
  }

  async *weightPages(runId: string, target: string, page: Pick<InspectionOptions, "value_offset" | "value_limit"> = {}, options: PageOptions = {}): AsyncGenerator<WeightReadPage> {
    let valueOffset = page.value_offset ?? 0;
    const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
    for (let count = 0; count < maximum; count++) {
      const inspection = await this.getInspection(runId, { ...page, limit: 1, weight_target: target, value_offset: valueOffset }, options);
      const weight = inspection.weight_reads[0];
      if (inspection.weight_reads.length !== 1 || !weight || weight.target !== target
        || weight.value_offset !== valueOffset || !Array.isArray(weight.values)
        || weight.values.length > (page.value_limit ?? 32)
        || (weight.more_values && weight.values.length === 0)) {
        throw protocolError("get_inspection", "invalid captured-weight pagination");
      }
      yield weight;
      if (!weight.more_values) return;
      valueOffset += weight.values.length;
    }
    throw new NrlyClientError("protocol", "page_limit", `Captured weights exceeded ${maximum} pages; resume with value_offset`, { resourceId: runId });
  }

  wait(runId: string, options: WaitOptions<RunState> = {}): Promise<RunState> {
    return waitFor(runId, (signal) => this.getRun(runId, { signal }), () => this.cancel(runId), options);
  }

  async request(tool: string, args: Record<string, unknown>, options: RequestOptions = {}): Promise<Record<string, unknown>> {
    const timeoutMs = positive("timeoutMs", options.timeoutMs ?? this.requestTimeoutMs);
    try {
      const result = await within((signal) => this.transport.call(tool, args, signal), timeoutMs, options.signal,
        new NrlyClientError("timeout", "request_timeout", `${tool} exceeded ${timeoutMs} ms; a submitted job may still be running`, { tool }));
      const data = object(tool, result);
      if (Array.from(JSON.stringify(data)).length > this.responseCharacterLimit) throw protocolError(tool, "response exceeded the configured character budget");
      return data;
    } catch (cause) {
      if (cause instanceof NrlyClientError) throw cause;
      throw new NrlyClientError("transport", "transport_error", `${tool}: ${cause instanceof Error ? cause.message : String(cause)}`, { tool, cause });
    }
  }
}

export class RunHandle {
  readonly id: string;
  readonly artifact: string;
  private readonly client: NrlyClient;

  constructor(client: NrlyClient, id: string) {
    this.client = client;
    this.id = identifier(id);
    this.artifact = `nrly://runs/${encodeURIComponent(id)}`;
  }

  get(options?: RequestOptions): Promise<RunState> { return this.client.getRun(this.id, options); }
  cancel(options?: RequestOptions): Promise<SubmittedRun> { return this.client.cancel(this.id, options); }
  wait(options?: WaitOptions<RunState>): Promise<RunState> { return this.client.wait(this.id, options); }
  inspection(page?: InspectionOptions, options?: RequestOptions): Promise<InspectionPage> { return this.client.getInspection(this.id, page, options); }
  inspectionPages(page?: InspectionOptions, options?: PageOptions): AsyncGenerator<InspectionPage> { return this.client.inspectionPages(this.id, page, options); }
  sample(caseId: string, page?: SampleOptions, options?: RequestOptions): Promise<SamplePage> { return this.client.getSample(this.id, caseId, page, options); }
  samples(page?: OffsetPage, options?: RequestOptions): Promise<SamplesPage> { return this.client.getSamples(this.id, page, options); }
  samplePages(page?: OffsetPage, options?: PageOptions): AsyncGenerator<SamplesPage> { return this.client.samplePages(this.id, page, options); }
  sampleOutputPages(caseId: string, page?: SampleOptions, options?: PageOptions): AsyncGenerator<SamplePage> { return this.client.sampleOutputPages(this.id, caseId, page, options); }
  sampleChangePages(caseId: string, page?: SampleOptions, options?: PageOptions): AsyncGenerator<SamplePage> { return this.client.sampleChangePages(this.id, caseId, page, options); }
  weightPages(target: string, page?: Pick<InspectionOptions, "value_offset" | "value_limit">, options?: PageOptions): AsyncGenerator<WeightReadPage> { return this.client.weightPages(this.id, target, page, options); }
  diagnostics(caseId: string, page?: DiagnosticOptions, options?: RequestOptions): Promise<DiagnosticsPage> { return this.client.getDiagnostics(this.id, caseId, page, options); }
  diagnosticPages(caseId: string, page?: DiagnosticOptions, options?: PageOptions): AsyncGenerator<DiagnosticsPage> { return this.client.diagnosticPages(this.id, caseId, page, options); }
  capture(caseId: string, target: string, page?: OffsetPage, options?: RequestOptions): Promise<CapturePage> { return this.client.getCapture(this.id, caseId, target, page, options); }
}

export async function waitFor<T extends { status: string }>(id: string, read: (signal: AbortSignal) => Promise<T>, cancel: () => Promise<unknown>, options: WaitOptions<T>, terminalStatuses: ReadonlySet<string> = terminal, successStatus = "succeeded"): Promise<T> {
  const timeoutMs = positive("timeoutMs", options.timeoutMs ?? 120_000);
  const pollIntervalMs = positive("pollIntervalMs", options.pollIntervalMs ?? 250);
  try {
    const result = await within(async (signal) => {
      while (true) {
        const state = await read(signal);
        await options.onProgress?.(state);
        if (terminalStatuses.has(state.status)) return state;
        await sleep(pollIntervalMs, signal);
      }
    }, timeoutMs, options.signal, new NrlyClientError("timeout", "wait_timeout", `Waiting for ${id} exceeded ${timeoutMs} ms; the job may still be running`, { resourceId: id }));
    if (options.requireSuccess && result.status !== successStatus) {
      throw new NrlyClientError("execution", result.status, `${id} ended ${result.status}`, { resourceId: id });
    }
    return result;
  } catch (error) {
    if (error instanceof NrlyClientError && ((error.code === "wait_timeout" && options.cancelOnTimeout)
      || (error.kind === "aborted" && options.cancelOnAbort))) {
      try { await cancel(); }
      catch (cause) {
        throw new NrlyClientError(error.kind, error.code, `${error.message}; cancellation also failed`, { resourceId: id, cause: { waiting: error, cancellation: cause } });
      }
    }
    throw error;
  }
}

export function object(tool: string, value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw protocolError(tool, "expected an object");
  return value as Record<string, unknown>;
}

export function stringField(tool: string, value: Record<string, unknown>, field: string): string {
  const result = value[field];
  if (typeof result !== "string" || !result) throw protocolError(tool, `missing ${field}`);
  return result;
}

export function validateIdentity(tool: string, value: object, field: string, expected: string): void {
  if ((value as Record<string, unknown>)[field] !== expected) throw protocolError(tool, `response ${field} does not match the request`);
}

export function validateStatus(tool: string, value: Record<string, unknown>): void {
  if (!statuses.has(value.status as RunStatus)) throw protocolError(tool, `unknown status ${String(value.status)}`);
}

export function validatePage(tool: string, value: Record<string, unknown>, field: string, offset: number, limit: number): void {
  if (!Array.isArray(value[field]) || value[field].length > limit || value.offset !== offset
    || (value.next_offset !== null && (!Number.isSafeInteger(value.next_offset) || Number(value.next_offset) <= offset))) {
    throw protocolError(tool, `invalid ${field} pagination`);
  }
}

export function integer(name: string, value: number, minimum: number, maximum = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new RangeError(`${name} must be an integer between ${minimum} and ${maximum}`);
  return value;
}

export function identifier(value: string): string {
  if (!value || value.length > 256 || !/^[a-zA-Z0-9_.-]+$/.test(value)) throw new TypeError("Invalid nrly resource ID");
  return value;
}

function submitted(value: Record<string, unknown>, tool: string): SubmittedRun {
  stringField(tool, value, "run_id");
  validateStatus(tool, value);
  return value as unknown as SubmittedRun;
}

function positive(name: string, value: number): number {
  if (!Number.isFinite(value) || value <= 0 || value > 2_147_483_647) throw new RangeError(`${name} must be positive and at most 2147483647`);
  return value;
}

function within<T>(work: (signal: AbortSignal) => Promise<T>, timeoutMs: number, signal: AbortSignal | undefined, timeout: NrlyClientError): Promise<T> {
  if (signal?.aborted) return Promise.reject(new NrlyClientError("aborted", "aborted", "Operation aborted", { cause: signal.reason }));
  return new Promise<T>((resolve, reject) => {
    const controller = new AbortController();
    let settled = false;
    const finish = (succeeded: boolean, value: unknown): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      if (succeeded) resolve(value as T);
      else { reject(value); controller.abort(value); }
    };
    const onAbort = (): void => finish(false, new NrlyClientError("aborted", "aborted", "Operation aborted", { cause: signal?.reason }));
    const timer = setTimeout(() => finish(false, timeout), timeoutMs);
    signal?.addEventListener("abort", onAbort, { once: true });
    Promise.resolve().then(() => {
      if (controller.signal.aborted) throw controller.signal.reason;
      return work(controller.signal);
    }).then((value) => finish(true, value), (error: unknown) => finish(false, error));
  });
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise<void>((resolve, reject) => {
    const abort = (): void => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener("abort", abort); resolve(); }, ms);
    signal.addEventListener("abort", abort, { once: true });
  });
}
