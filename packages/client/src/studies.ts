import type {
  CandidateSpec, CaseSpec, ComparatorSpec, GenerationSpec, HoldoutCommitment, ModelSpec,
  StudyAssessment, StudyBudget, StudyUsage,
} from "./contracts.generated.js";
import { identifier, integer, object, stringField, validateIdentity, validatePage, validateStatus, waitFor } from "./client.js";
import type { NrlyClient } from "./client.js";
import { NrlyClientError, protocolError } from "./errors.js";
import { readJsonRecord } from "./evidence.js";
import type { ModelIdentity, OffsetPage, PageOptions, RequestOptions, RunStatus, WaitOptions } from "./types.js";

export type StudyStatus = "open" | "finalizing" | "completed" | "interrupted";
type LogicalArm = "baseline" | "no_op" | "candidate" | `control:${string}` | `comparator:${string}`;
export type DevelopmentArm = LogicalArm | `${LogicalArm}@seed=${number}`;
export interface StudySummary {
  id: string; name: string; status: StudyStatus; attempts: number; created_at: string;
  verdict: StudyAssessment["verdict"] | null;
}
export interface StudyState {
  id: string; name: string; status: StudyStatus; created_at: string; finished_at: string | null;
  spec_digest: string; holdout: HoldoutCommitment; model: ModelSpec; generation: GenerationSpec;
  model_identity: ModelIdentity | null; development_case_count: number; controls: string[];
  mutable_targets: string[] | null; mutable_target_count: number | null;
  mutable_target_offset: number; next_target_offset: number | null;
  usage: Required<StudyUsage>; budget: Required<StudyBudget>; budget_semantics: string;
  nominee_attempt_id: string | null; nominee_digest: string | null;
  final_result: StudyAssessment | null; final_error_code: string | null;
  seeds?: number[] | null; comparators?: Array<{ name: string; kind: "prompt" | "model" }>;
}
export interface AttemptSummary {
  id: string; attempt_id: string; study_id: string; sequence: number; name: string;
  status: Exclude<RunStatus, "cancelling">; program_kind: string; candidate_digest: string;
  created_at: string; finished_at: string | null; error_code: string | null;
  verdict: StudyAssessment["verdict"] | null;
  recommendation: "retain_original" | "candidate_supported" | null;
}
export interface AttemptState extends AttemptSummary { assessment: StudyAssessment | null }
export interface AttemptPage {
  study_id: string; offset: number; total: number; next_offset: number | null; attempts: AttemptSummary[];
}
export interface StudyCase extends CaseSpec {
  input_offset: number; input_characters: number; expected_characters: number; more_input: boolean;
}
export interface StudyCasePage {
  study_id: string; offset: number; total: number; next_offset: number | null; cases: StudyCase[];
}
export interface StudySample {
  case_id: string; status: "completed" | "failed";
  provenance: "torch_fixture" | "huggingface" | "unavailable";
  output: string; output_offset: number; output_characters: number; more_output: boolean;
  truncated: boolean; error_code: string | null;
  inference_truncated?: boolean | null; finish_reason?: string | null;
}
export interface StudyMessageOptions extends OffsetPage { content_offset?: number; content_limit?: number }
export interface StudyComparator { registration: ComparatorSpec; registration_digest: string; model_identity: ModelIdentity | null }
export interface StudyMessagePage {
  study_id: string; case_id: string; offset: number; total: number; next_offset: number | null;
  messages: Array<{ role: "system" | "user" | "assistant"; content: string; content_offset: number; content_characters: number; more_content: boolean }>;
}
export interface StudySamplePage {
  study_id: string; attempt_id: string; arm: DevelopmentArm; samples: StudySample[];
  offset: number; total: number; next_offset: number | null;
}
export interface StudyCaseOptions extends OffsetPage { input_offset?: number; input_limit?: number }
export interface StudySampleOptions extends OffsetPage { output_limit?: number; output_offset?: number }
export interface StudyGetOptions extends RequestOptions { target_offset?: number; target_limit?: number }
export interface SubmittedAttempt { study_id: string; attempt_id: string; status: Exclude<RunStatus, "cancelling"> }
export interface StudyTransition { study_id: string; status: StudyStatus }

export class StudyHandle {
  readonly id: string;
  private readonly client: NrlyClient;

  constructor(client: NrlyClient, id: string) { this.client = client; this.id = identifier(id); }

  async get(options: StudyGetOptions = {}): Promise<StudyState> {
    integer("target_offset", options.target_offset ?? 0, 0);
    integer("target_limit", options.target_limit ?? 16, 1, 16);
    const result = await this.client.request("get_study", {
      study_id: this.id, target_offset: options.target_offset ?? 0, target_limit: options.target_limit ?? 16,
    }, options);
    validateIdentity("get_study", result, "id", this.id);
    studyStatus("get_study", result);
    return result as unknown as StudyState;
  }

  async *targetPages(page: Pick<StudyGetOptions, "target_offset" | "target_limit"> = {}, options: PageOptions = {}): AsyncGenerator<StudyState> {
    let offset = page.target_offset ?? 0;
    const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
    for (let count = 0; count < maximum; count++) {
      const result = await this.get({ ...options, ...page, target_offset: offset });
      if (result.mutable_target_offset !== offset
        || (result.mutable_targets !== null && (!Array.isArray(result.mutable_targets) || result.mutable_targets.length > (page.target_limit ?? 16)))
        || (result.next_target_offset !== null && (!Number.isSafeInteger(result.next_target_offset) || result.next_target_offset <= offset))) {
        throw protocolError("get_study", "invalid mutable-target pagination");
      }
      yield result;
      if (result.next_target_offset === null) return;
      offset = result.next_target_offset;
    }
    throw new NrlyClientError("protocol", "page_limit", `Mutable targets exceeded ${maximum} pages; resume with next_target_offset`, { resourceId: this.id });
  }

  attempt(attemptId: string): StudyAttemptHandle { return new StudyAttemptHandle(this.client, this.id, attemptId); }

  async submit(candidate: CandidateSpec, requestId?: string, options: RequestOptions = {}): Promise<StudyAttemptHandle> {
    const result = await this.client.request("submit_study_candidate", {
      study_id: this.id, candidate, ...(requestId !== undefined ? { request_id: requestId } : {}),
    }, options);
    validateIdentity("submit_study_candidate", result, "study_id", this.id);
    validateStatus("submit_study_candidate", result);
    return this.attempt(stringField("submit_study_candidate", result, "attempt_id"));
  }

  async attempts(page: OffsetPage = {}, options: RequestOptions = {}): Promise<AttemptPage> {
    pageBounds(page, 20);
    const result = await this.client.request("list_study_attempts", { study_id: this.id, ...page }, options);
    validateIdentity("list_study_attempts", result, "study_id", this.id);
    validatePage("list_study_attempts", result, "attempts", page.offset ?? 0, page.limit ?? 20);
    return result as unknown as AttemptPage;
  }

  attemptPages(page: OffsetPage = {}, options: PageOptions = {}): AsyncGenerator<AttemptPage> {
    return pages(this.id, (offset) => this.attempts({ ...page, offset }, options), page, options);
  }

  async cases(page: StudyCaseOptions = {}, options: RequestOptions = {}): Promise<StudyCasePage> {
    pageBounds(page, 5);
    integer("input_offset", page.input_offset ?? 0, 0);
    integer("input_limit", page.input_limit ?? 1000, 1, 1000);
    const result = await this.client.request("get_study_cases", { study_id: this.id, ...page }, options);
    validateIdentity("get_study_cases", result, "study_id", this.id);
    validatePage("get_study_cases", result, "cases", page.offset ?? 0, page.limit ?? 5);
    return result as unknown as StudyCasePage;
  }

  casePages(page: StudyCaseOptions = {}, options: PageOptions = {}): AsyncGenerator<StudyCasePage> {
    return pages(this.id, (offset) => this.cases({ ...page, offset }, options), page, options);
  }

  async messages(caseId: string, page: StudyMessageOptions = {}, options: RequestOptions = {}): Promise<StudyMessagePage> {
    pageBounds(page, 4);
    integer("content_offset", page.content_offset ?? 0, 0);
    integer("content_limit", page.content_limit ?? 1000, 1, 1000);
    const result = await this.client.request("get_study_case_messages", { study_id: this.id, case_id: caseId, ...page }, options);
    validateIdentity("get_study_case_messages", result, "study_id", this.id);
    validateIdentity("get_study_case_messages", result, "case_id", caseId);
    validatePage("get_study_case_messages", result, "messages", page.offset ?? 0, page.limit ?? 4);
    return result as unknown as StudyMessagePage;
  }

  async caseSpec(caseId: string, options: PageOptions = {}): Promise<CaseSpec> {
    const result = await readJsonRecord(this.client, "get_study_case_spec", { study_id: this.id, case_id: caseId }, options);
    validateIdentity("get_study_case_spec", result, "id", caseId);
    return result as unknown as CaseSpec;
  }

  finalAssessment(options: PageOptions = {}): Promise<StudyAssessment> {
    return readAssessment(this.client, this.id, null, options);
  }

  async comparator(name: string, options: PageOptions = {}): Promise<StudyComparator> {
    const result = await readJsonRecord(this.client, "get_study_comparator", { study_id: this.id, comparator: name }, options);
    validateIdentity("get_study_comparator", object("get_study_comparator", result.registration), "name", name);
    return result as unknown as StudyComparator;
  }

  async finalize(attemptId: string, options: RequestOptions = {}): Promise<StudyTransition> {
    const result = await this.client.request("finalize_study", { study_id: this.id, attempt_id: identifier(attemptId) }, options);
    return transition("finalize_study", result, this.id);
  }

  async cancelFinal(options: RequestOptions = {}): Promise<StudyTransition> {
    const result = await this.client.request("cancel_study_final", { study_id: this.id }, options);
    return transition("cancel_study_final", result, this.id);
  }

  async waitFinal(options: WaitOptions<StudyState> = {}): Promise<StudyState> {
    const state = await waitFor(this.id, (signal) => this.get({ signal }), () => this.cancelFinal(), options,
      new Set(["completed", "interrupted"]), "completed");
    if (options.requireSuccess && (state.final_error_code || !state.final_result)) {
      throw new NrlyClientError("execution", state.final_error_code ?? "assessment_unavailable", "Final evaluation did not complete successfully", { resourceId: this.id });
    }
    return state;
  }
}

export class StudyAttemptHandle {
  readonly id: string;
  readonly studyId: string;
  private readonly client: NrlyClient;

  constructor(client: NrlyClient, studyId: string, id: string) {
    this.client = client; this.id = identifier(id); this.studyId = identifier(studyId);
  }

  async get(options: RequestOptions = {}): Promise<AttemptState> {
    const result = await this.client.request("get_study_attempt", { study_id: this.studyId, attempt_id: this.id }, options);
    attemptIdentity("get_study_attempt", result, this.studyId, this.id);
    return result as unknown as AttemptState;
  }

  async cancel(options: RequestOptions = {}): Promise<SubmittedAttempt> {
    const result = await this.client.request("cancel_study_attempt", { study_id: this.studyId, attempt_id: this.id }, options);
    attemptIdentity("cancel_study_attempt", result, this.studyId, this.id);
    return result as unknown as SubmittedAttempt;
  }

  wait(options: WaitOptions<AttemptState> = {}): Promise<AttemptState> {
    return waitFor(this.id, (signal) => this.get({ signal }), () => this.cancel(), options);
  }

  assessment(options: PageOptions = {}): Promise<StudyAssessment> {
    return readAssessment(this.client, this.studyId, this.id, options);
  }

  async samples(arm: DevelopmentArm, page: StudySampleOptions = {}, options: RequestOptions = {}): Promise<StudySamplePage> {
    pageBounds(page, 5);
    integer("output_limit", page.output_limit ?? 512, 1, 512);
    integer("output_offset", page.output_offset ?? 0, 0);
    const result = await this.client.request("get_study_samples", {
      study_id: this.studyId, attempt_id: this.id, arm, ...page,
    }, options);
    validateIdentity("get_study_samples", result, "study_id", this.studyId);
    validateIdentity("get_study_samples", result, "attempt_id", this.id);
    validateIdentity("get_study_samples", result, "arm", arm);
    validatePage("get_study_samples", result, "samples", page.offset ?? 0, page.limit ?? 5);
    return result as unknown as StudySamplePage;
  }

  samplePages(arm: DevelopmentArm, page: StudySampleOptions = {}, options: PageOptions = {}): AsyncGenerator<StudySamplePage> {
    return pages(this.id, (offset) => this.samples(arm, { ...page, offset }, options), page, options);
  }

  async *sampleOutputPages(arm: DevelopmentArm, caseOffset: number, page: Pick<StudySampleOptions, "output_offset" | "output_limit"> = {}, options: PageOptions = {}): AsyncGenerator<StudySamplePage> {
    integer("caseOffset", caseOffset, 0);
    let offset = page.output_offset ?? 0;
    const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
    for (let count = 0; count < maximum; count++) {
      const result = await this.samples(arm, { ...page, offset: caseOffset, limit: 1, output_offset: offset }, options);
      const sample = result.samples[0];
      if (sample && (sample.output_offset !== offset || typeof sample.output !== "string"
        || typeof sample.more_output !== "boolean" || (sample.more_output && !sample.output.length))) {
        throw protocolError("get_study_samples", "invalid development output pagination");
      }
      yield result;
      if (!sample?.more_output) return;
      offset += Array.from(sample.output).length;
    }
    throw new NrlyClientError("protocol", "page_limit", `Development output exceeded ${maximum} pages; resume with output_offset`, { resourceId: this.id });
  }
}

export async function listStudies(client: NrlyClient, limit: number, options: RequestOptions): Promise<StudySummary[]> {
  integer("limit", limit, 1, 20);
  const result = await client.request("list_studies", { limit }, options);
  if (!Array.isArray(result.studies) || result.studies.length > limit) throw protocolError("list_studies", "invalid study list");
  for (const item of result.studies) {
    const study = object("list_studies", item);
    stringField("list_studies", study, "id");
    studyStatus("list_studies", study);
  }
  return result.studies as StudySummary[];
}

async function readAssessment(client: NrlyClient, studyId: string, attemptId: string | null, options: PageOptions): Promise<StudyAssessment> {
  const parsed = await readJsonRecord(client, "get_study_assessment", { study_id: studyId, attempt_id: attemptId }, options);
  if (!Array.isArray(parsed.arms) || !Array.isArray(parsed.comparisons) || typeof parsed.verdict !== "string") throw protocolError("get_study_assessment", "invalid assessment record");
  return parsed as unknown as StudyAssessment;
}

function pageBounds(page: OffsetPage, maximum: number): void {
  integer("offset", page.offset ?? 0, 0);
  integer("limit", page.limit ?? maximum, 1, maximum);
}

function studyStatus(tool: string, value: Record<string, unknown>): void {
  if (!["open", "finalizing", "completed", "interrupted"].includes(String(value.status))) throw protocolError(tool, "unknown study status");
}

function transition(tool: string, result: Record<string, unknown>, studyId: string): StudyTransition {
  validateIdentity(tool, result, "study_id", studyId);
  studyStatus(tool, result);
  return result as unknown as StudyTransition;
}

function attemptIdentity(tool: string, result: Record<string, unknown>, studyId: string, attemptId: string): void {
  validateIdentity(tool, result, "study_id", studyId);
  validateIdentity(tool, result, "attempt_id", attemptId);
  validateStatus(tool, result);
}

async function* pages<T extends { next_offset: number | null }>(id: string, read: (offset: number) => Promise<T>, page: OffsetPage, options: PageOptions): AsyncGenerator<T> {
  let offset = page.offset ?? 0;
  const maximum = integer("maxPages", options.maxPages ?? 100, 1, 10_000);
  for (let count = 0; count < maximum; count++) {
    const result = await read(offset);
    yield result;
    if (result.next_offset === null) return;
    offset = result.next_offset;
  }
  throw new NrlyClientError("protocol", "page_limit", `Exceeded ${maximum} pages; resume with the last next_offset`, { resourceId: id });
}
