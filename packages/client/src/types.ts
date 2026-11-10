import type { ExperimentSpec, ModelSpec, TensorCapture, TensorInfo, WeightChange } from "./contracts.generated.js";

export type RunStatus = "queued" | "running" | "cancelling" | "succeeded" | "failed" | "cancelled" | "interrupted";
export interface ErrorInfo { code: string; message: string }
export interface ModelIdentity {
  backend: string; model_id: string; revision: string; fingerprint: string;
  details: Record<string, unknown>;
}
export interface RunState {
  id: string;
  status: RunStatus;
  error: ErrorInfo | null;
  created_at: string;
  finished_at: string | null;
  elapsed_ms: number;
  truncated?: boolean;
  artifact?: string;
  spec?: ExperimentSpec;
  model_identity?: ModelIdentity | null;
  samples?: SampleResult[];
  inspection?: { tensors: TensorInfo[]; weights: TensorCapture[] } | null;
}
export interface RunSummary {
  id: string; status: RunStatus; name: string; purpose: "experiment" | "inspection";
  model_id: string; created_at: string; finished_at: string | null;
  samples_completed: number; case_count: number; elapsed_ms: number; error: ErrorInfo | null;
}
export interface SubmittedRun { run_id: string; status: RunStatus }
export interface WeightReadPage extends TensorCapture {
  value_offset: number; captured_elements: number; more_values: boolean;
}
export interface InspectionPage {
  run_id: string; status: RunStatus; model_identity: ModelIdentity | null;
  tensors: TensorInfo[]; total: number; offset: number; limit: number; next_offset: number | null;
  weight_reads: WeightReadPage[];
}
export interface SampleResult {
  case_id: string; status: "completed" | "failed";
  provenance: "torch_fixture" | "huggingface" | "unavailable";
  output: string | null; elapsed_ms: number;
  input_tokens: number | null; output_tokens: number | null;
  finish_reason: string | null; captures: TensorCapture[];
  error: ErrorInfo | null; weight_changes: WeightChange[];
  diagnostics?: Omit<import("./contracts.generated.js").ContinuationScore, "tokens"> & { tokens?: import("./contracts.generated.js").TokenObservation[] };
  diagnostic_token_count?: number; diagnostic_tokens_omitted?: boolean;
  truncated?: boolean;
}
export interface SamplePage extends SampleResult {
  output_offset: number; output_characters: number; more_output: boolean;
  change_offset: number; weight_change_count: number; next_change_offset: number | null;
}
export interface SampleSummary {
  case_id: string; status: "completed" | "failed";
  provenance: "torch_fixture" | "huggingface" | "unavailable";
  elapsed_ms: number; output_characters: number; output_preview: string; error_code: string | null;
}
export interface SamplesPage {
  run_id: string; status: RunStatus; samples: SampleSummary[];
  total: number; expected_cases: number; offset: number; next_offset: number | null;
}
export interface OffsetPage { offset?: number; limit?: number }
export interface Capabilities {
  name: string; version: string; schema_version: string; backends: string[];
  weight_operations: string[]; diagnostic_operations: string[]; control_operations: string[];
  arbitrary_code: boolean; limits: { response_characters: number; [key: string]: unknown };
  [key: string]: unknown;
}
export interface Evaluation {
  complete: boolean; scope: "engineering_fixture" | "real_model";
  summary: { total: number; scored: number; passed: number; failed: number; unscored: number; accuracy: number | null };
  issues: string[]; [key: string]: unknown;
}
export interface Comparison {
  comparable: boolean;
  verdict: "observed_improvement" | "observed_regression" | "mixed" | "no_change" | "inconclusive";
  baseline: Evaluation; candidate: Evaluation; accuracy_delta: number | null;
  improved_cases: string[]; regressed_cases: string[]; unchanged_cases: string[];
  issues: string[]; truncated?: boolean;
}
export interface InspectionOptions {
  offset?: number; limit?: number; weight_target?: string;
  value_offset?: number; value_limit?: number;
}
export interface SampleOptions { output_offset?: number; output_limit?: number; change_offset?: number; change_limit?: number }
export interface DiagnosticOptions extends OffsetPage { top_k_offset?: number; top_k_limit?: number }
export interface TokenAlternativePage { token_id: number; text: string; logprob: number; text_characters: number; text_truncated: boolean }
export interface TokenObservationPage extends TokenAlternativePage {
  position: number; rank: number; top_k: TokenAlternativePage[];
  top_k_offset: number; top_k_total: number; next_top_k_offset: number | null;
}
export interface DiagnosticsPage {
  run_id: string; case_id: string;
  diagnostics: Omit<import("./contracts.generated.js").ContinuationScore, "tokens">;
  offset: number; total: number; next_offset: number | null; tokens: TokenObservationPage[];
  evidence_kind: "teacher_forced_diagnostic_not_task_accuracy";
}
export interface CapturePage {
  run_id: string; case_id: string; capture: Omit<TensorCapture, "values">;
  offset: number; total: number; next_offset: number | null; values: number[];
}
export interface RequestOptions { signal?: AbortSignal; timeoutMs?: number }
export interface PageOptions extends RequestOptions { maxPages?: number }
export interface WaitOptions<T> extends RequestOptions {
  pollIntervalMs?: number;
  requireSuccess?: boolean;
  cancelOnTimeout?: boolean;
  cancelOnAbort?: boolean;
  onProgress?: (state: T) => void | Promise<void>;
}
export interface ClientOptions { requestTimeoutMs?: number; responseCharacterLimit?: number }
export interface InspectionSubmission {
  model: ModelSpec;
  weight_reads?: import("./contracts.generated.js").WeightReadSpec[];
  name?: string;
  request_id?: string;
}
