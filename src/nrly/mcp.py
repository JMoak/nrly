from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from nrly import __version__
from nrly.api import Application, bounded
from nrly.api import capabilities as get_capabilities
from nrly.contracts import ExperimentSpec, ModelSpec, NrlyError, WeightReadSpec
from nrly.study_contracts import CandidateSpec


def public_errors[**P](function: Callable[P, dict[str, Any]]) -> Callable[P, dict[str, Any]]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        try:
            return bounded(function(*args, **kwargs))
        except NrlyError as error:
            raise ToolError(f"{error.code}: {error}") from error

    return wrapped


def create_server(app: Application) -> MCPServer[Any]:
    server: MCPServer[Any] = MCPServer(
        "nrly",
        version=__version__,
        log_level="WARNING",
        instructions=(
            "Weight-space programming workbench. Inspect exact parameters before editing. "
            "Submit a pinned baseline, candidate, and independent regression cases. "
            "Submission returns immediately: poll get_run until terminal. "
            "Failed/unscored runs are not evidence of improvement. "
            "Tiny results are engineering fixtures. Activation interventions are diagnostic."
        ),
    )

    @server.tool()
    @public_errors
    def capabilities() -> dict[str, Any]:
        """Discover supported backends, operations and limits without loading a model."""
        return bounded(get_capabilities())

    @server.tool()
    @public_errors
    def submit_experiment(spec: ExperimentSpec, request_id: str | None = None) -> dict[str, Any]:
        """Queue a bounded experiment; identical request IDs are idempotent."""
        return bounded(app.submit(spec, request_id))

    @server.tool()
    @public_errors
    def submit_inspection(
        model: ModelSpec,
        weight_reads: list[WeightReadSpec] | None = None,
        name: str = "Model inspection",
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Queue a model catalog and bounded parameter reads without forward passes."""
        return bounded(app.submit_inspection(model, weight_reads or [], name, request_id))

    @server.tool()
    @public_errors
    def get_run(run_id: str) -> dict[str, Any]:
        """Read status/results; large records return an explicit summary and artifact handle."""
        return bounded(app.get_run(run_id))

    @server.tool()
    @public_errors
    def get_inspection(
        run_id: str,
        offset: int = 0,
        limit: int = 40,
        weight_target: str | None = None,
        value_offset: int = 0,
        value_limit: int = 32,
    ) -> dict[str, Any]:
        """Page a completed run's tensor catalog and captured pre-edit weights."""
        return bounded(
            app.get_inspection(run_id, offset, limit, weight_target, value_offset, value_limit)
        )

    @server.tool()
    @public_errors
    def get_samples(run_id: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Enumerate bounded sample summaries, including when the complete run is truncated."""
        return app.get_samples(run_id, offset, limit)

    @server.tool()
    @public_errors
    def get_sample(
        run_id: str,
        case_id: str,
        output_offset: int = 0,
        output_limit: int = 2000,
        change_offset: int = 0,
        change_limit: int = 4,
    ) -> dict[str, Any]:
        """Read one sample, bounded output text, captures and exact weight-change evidence."""
        return app.get_sample(
            run_id, case_id, output_offset, output_limit, change_offset, change_limit
        )

    @server.tool()
    @public_errors
    def list_runs(limit: int = 20) -> dict[str, Any]:
        """List the latest local run summaries."""
        return bounded(app.list_runs(limit))

    @server.tool()
    @public_errors
    def get_diagnostics(
        run_id: str,
        case_id: str,
        offset: int = 0,
        limit: int = 8,
        top_k_offset: int = 0,
        top_k_limit: int = 3,
    ) -> dict[str, Any]:
        """Page teacher-forced token probabilities; these are not generated task outcomes."""
        return app.get_diagnostics(run_id, case_id, offset, limit, top_k_offset, top_k_limit)

    @server.tool()
    @public_errors
    def get_capture(
        run_id: str, case_id: str, target: str, offset: int = 0, limit: int = 64
    ) -> dict[str, Any]:
        """Page stored activation values with their capture position and invocation metadata."""
        return app.get_capture(run_id, case_id, target, offset, limit)

    @server.tool()
    @public_errors
    def get_study_case_messages(
        study_id: str,
        case_id: str,
        offset: int = 0,
        limit: int = 4,
        content_offset: int = 0,
        content_limit: int = 1000,
    ) -> dict[str, Any]:
        """Read bounded chat messages for development cases only; never sealed holdout text."""
        return app.get_study_case_messages(
            study_id, case_id, offset, limit, content_offset, content_limit
        )

    @server.tool()
    @public_errors
    def cancel_run(run_id: str) -> dict[str, Any]:
        """Cancel an owned queued/running job; completed records remain immutable."""
        return bounded(app.cancel(run_id))

    @server.tool()
    @public_errors
    def compare_runs(baseline_id: str, candidate_id: str) -> dict[str, Any]:
        """Compare compatible scored runs, exposing improvements and regressions separately."""
        return bounded(app.compare(baseline_id, candidate_id))

    @server.tool()
    @public_errors
    def get_study(study_id: str, target_offset: int = 0, target_limit: int = 16) -> dict[str, Any]:
        """Read study design summary, cumulative reservations and aggregate final evidence."""
        return app.get_study(study_id, target_offset, target_limit)

    @server.tool()
    @public_errors
    def get_study_assessment(
        study_id: str,
        attempt_id: str | None = None,
        offset: int = 0,
        limit: int = 4000,
    ) -> dict[str, Any]:
        """Page complete assessment JSON; omit attempt_id for public aggregate final evidence."""
        return app.get_study_assessment(study_id, attempt_id, offset, limit)

    @server.tool()
    @public_errors
    def get_study_case_spec(
        study_id: str,
        case_id: str,
        offset: int = 0,
        limit: int = 4000,
    ) -> dict[str, Any]:
        """Page complete development case JSON including chat and scoring constraints."""
        return app.get_study_case_spec(study_id, case_id, offset, limit)

    @server.tool()
    @public_errors
    def get_study_comparator(
        study_id: str,
        comparator: str,
        offset: int = 0,
        limit: int = 4000,
    ) -> dict[str, Any]:
        """Page registered comparator configuration and its measured model identity."""
        return app.get_study_comparator(study_id, comparator, offset, limit)

    @server.tool()
    @public_errors
    def list_studies(limit: int = 20) -> dict[str, Any]:
        """Find operator-provisioned studies without accessing sealed cases."""
        return app.list_studies(limit)

    @server.tool()
    @public_errors
    def submit_study_candidate(
        study_id: str,
        candidate: CandidateSpec,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Admit a frozen candidate and queue baseline/control/candidate development runs."""
        return app.submit_study_candidate(study_id, candidate, request_id)

    @server.tool()
    @public_errors
    def get_study_attempt(study_id: str, attempt_id: str) -> dict[str, Any]:
        """Read a candidate attempt's status and aggregate development assessment."""
        return app.get_study_attempt(study_id, attempt_id)

    @server.tool()
    @public_errors
    def list_study_attempts(study_id: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Page all candidate attempts, preserving unsuccessful and cancelled attempts."""
        return app.list_study_attempts(study_id, offset, limit)

    @server.tool()
    @public_errors
    def finalize_study(study_id: str, attempt_id: str) -> dict[str, Any]:
        """Freeze one completed nominee and queue its one-shot held-out evaluation."""
        return app.finalize_study(study_id, attempt_id)

    @server.tool()
    @public_errors
    def cancel_study_attempt(study_id: str, attempt_id: str) -> dict[str, Any]:
        """Request candidate cancellation; consumed study reservations are not refunded."""
        return app.cancel_study_attempt(study_id, attempt_id)

    @server.tool()
    @public_errors
    def cancel_study_final(study_id: str) -> dict[str, Any]:
        """Stop final evaluation; the nominee remains frozen and the result is inconclusive."""
        return app.cancel_study_final(study_id)

    @server.tool()
    @public_errors
    def get_study_cases(
        study_id: str,
        offset: int = 0,
        limit: int = 5,
        input_offset: int = 0,
        input_limit: int = 1000,
    ) -> dict[str, Any]:
        """Page public development cases and expectations; never returns held-out cases."""
        return app.get_study_cases(study_id, offset, limit, input_offset, input_limit)

    @server.tool()
    @public_errors
    def get_study_samples(
        study_id: str,
        attempt_id: str,
        arm: str,
        offset: int = 0,
        limit: int = 5,
        output_limit: int = 512,
        output_offset: int = 0,
    ) -> dict[str, Any]:
        """Read bounded development outputs for baseline/no_op/control:NAME/candidate arms."""
        return app.get_study_samples(
            study_id, attempt_id, arm, offset, limit, output_limit, output_offset
        )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="nrly weight-space programming MCP server")
    parser.add_argument("--root", type=Path, default=Path(".nrly"))
    parser.add_argument("--model-cache", type=Path)
    args = parser.parse_args()
    app = Application(args.root, args.model_cache)
    try:
        create_server(app).run(transport="stdio")
    finally:
        app.close()


if __name__ == "__main__":
    main()
