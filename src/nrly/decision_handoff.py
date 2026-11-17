"""Ephemeral authenticated loopback SystemOne bridge. No registration or promotion."""

from __future__ import annotations

import hmac
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from nrly.contracts import Contract, NrlyError, digest
from nrly.decision_contracts import DecisionRequest, DecisionRun, FrozenRule, Hash, Semantics
from nrly.decision_qualification import assess_decisions, verify_run

if TYPE_CHECKING:
    from nrly.backends.decision import DecisionBackend


class ShadowHandoff(Contract):
    schema_version: Literal["nrly-shadow/1"] = "nrly-shadow/1"
    mode: Literal["shadow"] = "shadow"
    promotion_authorized: Literal[False] = False
    protocol: Literal["systemone-binary/1"] = "systemone-binary/1"
    integration_status: Literal["prepared_unverified"] = "prepared_unverified"
    run_digest: Hash
    freeze_digest: Hash
    served_model: str
    semantics: Semantics
    question: str
    rule: FrozenRule
    threshold_owner: Literal["consumer; server returns raw probabilities"] = (
        "consumer; server returns raw probabilities"
    )
    api_price_per_million_input: Literal[0] = 0
    cost_scope: Literal["no API billing; hardware and energy costs unmeasured"] = (
        "no API billing; hardware and energy costs unmeasured"
    )


class NoulQuestion(Contract):
    type: Literal["noul"]
    instructions: str = Field(min_length=1, max_length=4000)


class SystemOneRequest(Contract):
    model: str = Field(min_length=1, max_length=200)
    state: str = Field(min_length=1, max_length=16000)
    questions: dict[str, NoulQuestion] = Field(min_length=1, max_length=16)


def prepare_shadow_handoff(run: DecisionRun) -> ShadowHandoff:
    assessment = assess_decisions(run)
    if assessment.status != "complete":
        raise NrlyError("incomplete_evidence", "incomplete runs cannot prepare shadow handoffs")
    return ShadowHandoff(
        run_digest=digest(run),
        freeze_digest=digest(run.freeze),
        served_model=f"nrly/{digest(run.freeze)}/patch",
        semantics=run.freeze.semantics,
        question=run.protocol.question,
        rule=run.freeze.patch_rule,
    )


class ShadowDecisionService:
    """Binds trusted loaded execution to the exact reviewed candidate and rubric."""

    def __init__(self, backend: DecisionBackend, run: DecisionRun) -> None:
        verify_run(run)
        if (
            backend.identity != run.identity
            or backend.semantics != run.freeze.semantics
            or backend.execution_identity != run.freeze.execution_identity
        ):
            raise NrlyError("base_mismatch", "shadow backend differs from reviewed identity")
        self.backend = backend
        self.program = run.program.model_copy(deep=True)
        self.handoff = prepare_shadow_handoff(run)

    def answer(self, request: SystemOneRequest) -> dict[str, object]:
        request = SystemOneRequest.model_validate(request.model_dump())
        if request.model != self.handoff.served_model:
            raise NrlyError("base_mismatch", "request must name the exact served model")
        if any(q.instructions != self.handoff.question for q in request.questions.values()):
            raise NrlyError("task_mismatch", "questions must match the frozen rubric exactly")
        # All questions have identical instructions: score once, preserve caller IDs.
        result = self.backend.observe(
            DecisionRequest(state=request.state, question=self.handoff.question),
            self.program,
        )
        return {
            "model": self.handoff.served_model,
            "answers": {
                key: {"type": "noul", "noul": result.probability} for key in request.questions
            },
            "usage": {"input_tokens": result.input_tokens, "cost": 0},
        }


@contextmanager
def shadow_server(service: ShadowDecisionService, token: str) -> Iterator[str]:
    """Serial inference on 127.0.0.1, OS-selected port; lifetime is this context only.

    Caller supplies an ephemeral secret (at least 32 characters). No CORS, no prompt
    logging, no service registration. Do not expose this research bridge to a network.
    """
    if len(token) < 32 or not token.isascii():
        raise NrlyError("invalid_request", "a 32+ character ASCII bearer token is required")

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(10)

        def do_POST(self) -> None:
            if self.path != "/v1/systemone":
                self.send_error(404)
                return
            supplied = self.headers.get("Authorization", "").encode("utf-8")
            if not hmac.compare_digest(supplied, f"Bearer {token}".encode()):
                self.send_error(401)
                return
            if self.headers.get("Origin") or self.headers.get("Transfer-Encoding"):
                self.send_error(400)
                return
            try:
                count = int(self.headers.get("Content-Length", "0"))
                if not 0 < count <= 65536:
                    self.send_error(413)
                    return
                body = self.rfile.read(count)
                if len(body) != count:
                    self.send_error(400)
                    return
                request = SystemOneRequest.model_validate_json(body)
            except (ValueError, OSError):
                self.send_error(400)
                return
            try:
                answer = service.answer(request)
            except NrlyError:
                self.send_error(422)
                return
            except Exception:
                self.send_error(500)
                return
            raw = json.dumps(answer, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=15)
