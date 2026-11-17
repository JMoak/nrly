"""Trusted local binary adapters. Never load callback code from a JSON request."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from torch import nn

from nrly.backends.local import LocalExecution, LocalTorchBackend
from nrly.contracts import (
    CaseSpec,
    DiagnosticSpec,
    GenerationSpec,
    ModelIdentity,
    NrlyError,
    ProgramSpec,
    digest,
)
from nrly.decision_contracts import DecisionExecution, DecisionRequest, Semantics

if TYPE_CHECKING:
    from nrly.backends.huggingface import HuggingFaceBackend


class DecisionBackend(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    @property
    def semantics(self) -> Semantics: ...

    @property
    def execution_identity(self) -> str: ...

    def observe(self, request: DecisionRequest, program: ProgramSpec) -> DecisionExecution: ...


class CausalDecisionBackend:
    """Normalize P(Yes) and P(No); these are NOT native calibrated probabilities.

    Uses the existing teacher-forcing kernel, exact model/tokenizer identity, bounded
    continuations and full state restoration. Two forwards, no generated answer parsing.
    """

    semantics: Semantics = "causal_label_likelihood"

    def __init__(self, backend: HuggingFaceBackend) -> None:
        self.backend = backend
        self.identity = backend.identity
        self.execution_identity = digest(
            {
                "adapter": "nrly.causal-binary/1",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "labels": ["Yes", "No"],
                "generation": GenerationSpec(max_new_tokens=1).model_dump(mode="json"),
                "diagnostics": DiagnosticSpec(max_tokens=8).model_dump(mode="json"),
            }
        )

    def observe(self, request: DecisionRequest, program: ProgramSpec) -> DecisionExecution:
        prompt = (
            "Judge the supplied state using the question. Answer Yes or No.\n"
            f"Question: {request.question}\nState:\n{request.state}\nAnswer:"
        )
        results = [
            self.backend.run_case(
                CaseSpec(id="decision", input=prompt, continuation=label),
                program,
                [],
                GenerationSpec(max_new_tokens=1),
                DiagnosticSpec(max_tokens=8),
            )
            for label in ("Yes", "No")
        ]
        scores = [r.diagnostics for r in results]
        if any(s is None for s in scores):
            raise NrlyError("invalid_backend_result", "missing label likelihood")
        logs = [s.sum_logprob for s in scores if s is not None]
        maximum = max(logs)
        likelihoods = [math.exp(value - maximum) for value in logs]
        return DecisionExecution(
            probability=likelihoods[0] / sum(likelihoods),
            elapsed_ms=sum(r.elapsed_ms for r in results),
            input_tokens=sum(r.input_tokens or 0 for r in results),
            weight_changes=tuple(c for r in results for c in r.weight_changes),
        )


class LocalBinaryDecisionBackend:
    """Caller-owned native probability head; callback identity is caller-declared.

    As with LocalTorchBackend, this is trusted in-process execution with exclusive
    model ownership. A callback has no sandbox or hard wall-clock timeout.
    """

    semantics: Semantics = "native_binary_probability"

    def __init__(
        self,
        model: nn.Module,
        predict: Callable[[nn.Module, DecisionRequest], float],
        *,
        model_id: str,
        revision: str,
        adapter_id: str,
    ) -> None:
        def execute(module: nn.Module, case: CaseSpec, _: GenerationSpec) -> LocalExecution:
            request = DecisionRequest.model_validate_json(case.input or "")
            probability = DecisionExecution(probability=predict(module, request), elapsed_ms=0)
            return LocalExecution(output=probability.model_dump_json())

        self.backend = LocalTorchBackend(
            model,
            execute,
            model_id=model_id,
            revision=revision,
            adapter_id=adapter_id,
        )
        self.identity = self.backend.identity
        self.execution_identity = digest(
            {
                "adapter": "nrly.native-binary/1",
                "caller_declared_id": adapter_id,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
        )

    def observe(self, request: DecisionRequest, program: ProgramSpec) -> DecisionExecution:
        result = self.backend.run_case(
            CaseSpec(id="decision", input=request.model_dump_json()),
            program,
            [],
            GenerationSpec(),
        )
        probability = DecisionExecution.model_validate_json(result.output).probability
        return DecisionExecution(
            probability=probability,
            elapsed_ms=result.elapsed_ms,
            weight_changes=tuple(result.weight_changes),
            input_tokens=result.input_tokens,
        )
