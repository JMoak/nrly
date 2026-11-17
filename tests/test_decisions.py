from __future__ import annotations

import json
import math
import os
import secrets
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from nrly.authoring import WeightProgram
from nrly.backends.decision import LocalBinaryDecisionBackend
from nrly.contracts import NrlyError, digest
from nrly.decision_contracts import (
    DecisionCase,
    DecisionExecution,
    DecisionProtocol,
    DecisionRequest,
)
from nrly.decision_handoff import (
    ShadowDecisionService,
    ShadowHandoff,
    SystemOneRequest,
    prepare_shadow_handoff,
    shadow_server,
)
from nrly.decision_qualification import (
    assess_decisions,
    read_decision_run,
    run_decision_study,
    save_decision_run,
    verify_run,
)
from nrly.decision_report import render_decision_report
from nrly.kernel import tensor_digest


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 0.0]))


def protocol() -> DecisionProtocol:
    cases = []
    for split, values in (
        ("fit", [0.94, 0.86, 0.71, 0.61]),
        ("evaluation", [0.95, 0.87, 0.72, 0.62]),
        ("challenge", [0.93, 0.85, 0.73, 0.63]),
    ):
        for i, probability in enumerate(values):
            cases.append(
                DecisionCase(
                    id=f"{split}-{i}",
                    family=f"{split}-{i}",
                    split=split,
                    state=json.dumps(
                        {
                            "x": math.log(probability / (1 - probability)),
                            "private": f"SECRET-{split}-{i}",
                        }
                    ),
                    truth=i < 2,
                )
            )
    return DecisionProtocol(
        question="Is the example supported?",
        label_origin="synthetic",
        cases=tuple(cases),
    )


def backend(*, fail=False, seen=None):
    def predict(model, request: DecisionRequest):
        assert set(request.model_dump()) == {"state", "question"}
        if seen is not None:
            seen.append(request)
        if fail and "evaluation" in request.state:
            return float("nan")
        value = json.loads(request.state)["x"]
        return float(torch.sigmoid(model.weight[0] * value + model.weight[1]))

    return LocalBinaryDecisionBackend(
        Head(),
        predict,
        model_id="nrly/test-decision-head",
        revision="1",
        adapter_id="test-decision-head/1",
    )


def test_calibration_alone_does_not_claim_weight_benefit():
    model = backend()
    spec = protocol()
    original = tensor_digest(model.backend.model)
    noop = WeightProgram(model.identity.fingerprint).scale("weight", 1).build()
    events = []
    run = run_decision_study(model, spec, noop, record=events.append, on_freeze=events.append)
    assessment = assess_decisions(run)
    assert assessment.status == "complete"
    assert assessment.recommendation == "retain_original"
    assert run.freeze.original_rule.threshold == run.freeze.patch_rule.threshold == 0.8
    assert len(events) == 25 and events[8] == run.freeze
    assert tensor_digest(model.backend.model) == original
    fixed, fitted = [
        m for m in assessment.metrics if m.variant == "original" and m.split == "evaluation"
    ]
    assert (fixed.unsupported_acts, fitted.unsupported_acts) == (2, 0)
    assert fixed.brier == fitted.brier
    assert all(o.execution is not None for o in run.observations)


@pytest.mark.parametrize(
    "probability", [True, False, "0.99", float("nan"), float("inf"), -0.1, 1.1]
)
def test_malformed_probability_is_not_a_certain_decision(probability):
    with pytest.raises(ValidationError):
        DecisionExecution(probability=probability, elapsed_ms=0)


def test_last_call_overrun_retains_result_but_leaves_run_incomplete(monkeypatch):
    from nrly import decision_qualification

    clock = [0.0]
    monkeypatch.setattr(decision_qualification, "perf_counter", lambda: clock[0])
    model = backend()
    observe = model.observe
    calls = [0]

    def slow_final(request, program):
        calls[0] += 1
        result = observe(request, program)
        if calls[0] == 24:
            clock[0] = 100.0
        return result

    model.observe = slow_final
    spec = protocol().model_copy(update={"max_elapsed_seconds": 1})
    run = run_decision_study(model, spec, WeightProgram(model.identity.fingerprint).build())
    assert all(o.execution is not None for o in run.observations)
    assert run.budget_exceeded and run.elapsed_ms == 100000
    assert assess_decisions(run).status == "incomplete"


def test_real_head_edit_can_earn_only_further_shadow_evaluation():
    spec = protocol()
    cases = tuple(
        c.model_copy(
            update={
                "state": json.dumps(
                    {
                        "x": 0.4 if c.truth else 0.5,
                        "z": 2 if c.truth else -2,
                        "unique": c.id,
                    }
                )
            }
        )
        for c in spec.cases
    )
    spec = spec.model_copy(update={"cases": cases})

    def predict(model, request):
        values = json.loads(request.state)
        return float(torch.sigmoid(model.weight[0] * values["x"] + model.weight[1] * values["z"]))

    model = LocalBinaryDecisionBackend(
        Head(),
        predict,
        model_id="nrly/test-repairable-head",
        revision="1",
        adapter_id="repair/1",
    )
    program = WeightProgram(model.identity.fingerprint).patch("weight", [2], [1], [1.0]).build()
    run = run_decision_study(model, spec, program)
    assessment = assess_decisions(run)
    assert assessment.recommendation == "further_shadow_evaluation"
    assert assessment.rescued_supported_acts == 4
    assert assessment.introduced_unsupported_acts == 0
    assert prepare_shadow_handoff(run).promotion_authorized is False


def test_harmful_edit_preserved_and_weights_restored(tmp_path):
    model = backend()
    original = tensor_digest(model.backend.model)
    patch = WeightProgram(model.identity.fingerprint).scale("weight", -1).build()
    run = run_decision_study(model, protocol(), patch)
    assert run.freeze.patch_rule.threshold is None
    assert assess_decisions(run).recommendation == "retain_original"
    assert any(
        c.changed_elements > 0
        for o in run.observations
        if o.execution
        for c in o.execution.weight_changes
    )
    assert tensor_digest(model.backend.model) == original
    save_decision_run(tmp_path, run)
    assert read_decision_run(tmp_path) == run
    (tmp_path / "run.sha256").write_text("0" * 64)
    with pytest.raises(NrlyError, match="checksum"):
        read_decision_run(tmp_path)


def test_failure_cannot_qualify_or_export_handoff():
    model = backend(fail=True)
    run = run_decision_study(model, protocol(), WeightProgram(model.identity.fingerprint).build())
    assert assess_decisions(run).status == "incomplete"
    assert all(
        o.error_code == "execution_failed"
        for o in run.observations
        if o.case_id.startswith("evaluation")
    )
    with pytest.raises(NrlyError, match="incomplete"):
        prepare_shadow_handoff(run)


def test_fit_never_uses_evaluation_labels():
    model = backend()
    base = protocol()
    changed = base.model_copy(
        update={
            "cases": tuple(
                c.model_copy(update={"truth": not c.truth}) if c.split != "fit" else c
                for c in base.cases
            )
        }
    )
    program = WeightProgram(model.identity.fingerprint).build()
    first = run_decision_study(model, base, program)
    second = run_decision_study(model, changed, program)
    # Timing is recorded, so compare fitted thresholds, not observation hashes.
    assert first.freeze.original_rule.threshold == second.freeze.original_rule.threshold
    assert first.freeze.patch_rule.threshold == second.freeze.patch_rule.threshold
    assert first.freeze.protocol_digest != second.freeze.protocol_digest


@pytest.mark.parametrize("mutation", ["family", "input", "id", "truth", "grid"])
def test_rejects_leakage_and_malformed_protocols(mutation):
    raw = protocol().model_dump(mode="json")
    if mutation == "family":
        raw["cases"][4]["family"] = raw["cases"][0]["family"]
    if mutation == "input":
        raw["cases"][4]["state"] = raw["cases"][0]["state"]
    if mutation == "id":
        raw["cases"][4]["id"] = raw["cases"][0]["id"]
    if mutation == "truth":
        raw["cases"][0]["truth"] = "true"
    if mutation == "grid":
        raw["threshold_grid"] = [0.9, 0.5]
    with pytest.raises(ValidationError):
        DecisionProtocol.model_validate(raw)


def test_rejects_changed_rule_and_observation_identity():
    model = backend()
    run = run_decision_study(model, protocol(), WeightProgram(model.identity.fingerprint).build())
    forged = run.model_copy(
        update={
            "freeze": run.freeze.model_copy(
                update={
                    "patch_rule": run.freeze.patch_rule.model_copy(update={"threshold": 0.1}),
                }
            )
        }
    )
    with pytest.raises(NrlyError, match="frozen"):
        verify_run(forged)
    forged = run.model_copy(
        update={
            "observations": (
                run.observations[0].model_copy(update={"input_digest": "0" * 64}),
                *run.observations[1:],
            )
        }
    )
    with pytest.raises(NrlyError, match="binding"):
        verify_run(forged)


def test_report_excludes_private_examples_and_escapes_metadata():
    model = backend()
    run = run_decision_study(model, protocol(), WeightProgram(model.identity.fingerprint).build())
    report = render_decision_report(assess_decisions(run))
    assert "SECRET" not in report and "Is the example supported" not in report
    assert all(c.id not in report and c.state not in report for c in run.protocol.cases)
    hostile = assess_decisions(run).model_copy(update={"model_id": '<script>alert("x")</script>'})
    assert '<script>alert("x")</script>' not in render_decision_report(hostile)
    assert "https://" not in report


def test_shadow_roundtrip_auth_identity_and_raw_probability():
    model = backend()
    spec = protocol()
    run = run_decision_study(model, spec, WeightProgram(model.identity.fingerprint).build())
    service = ShadowDecisionService(model, run)
    assert ShadowHandoff.model_validate_json(service.handoff.model_dump_json()) == service.handoff
    token = secrets.token_urlsafe(32)
    payload = {
        "model": service.handoff.served_model,
        "state": spec.cases[0].state,
        "questions": {"test": {"type": "noul", "instructions": spec.question}},
    }
    with shadow_server(service, token) as url:

        def post(body, auth=token, extra=None):
            return urlopen(
                Request(
                    url + "/v1/systemone",
                    json.dumps(body).encode(),
                    headers={
                        "Authorization": "Bearer " + auth,
                        "Content-Type": "application/json",
                        **(extra or {}),
                    },
                ),
                timeout=10,
            )

        with post(payload) as response:
            value = json.load(response)
        assert value["answers"]["test"]["noul"] == pytest.approx(0.94)
        assert value["model"] == service.handoff.served_model and value["usage"]["cost"] == 0
        for body, auth, extra, status in [
            (payload, "bad", {}, 401),
            ({**payload, "model": "jev-latest"}, token, {}, 422),
            ({**payload, "state": 10}, token, {}, 400),
            (payload, token, {"Origin": "https://other.test"}, 400),
        ]:
            with pytest.raises(HTTPError) as error:
                post(body, auth, extra)
            assert error.value.code == status
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            {
                **payload,
                "questions": {
                    "a": {
                        "type": "noul",
                        "instructions": spec.question,
                        "criteria": {"true": "injected"},
                    }
                },
            }
        )
    assert service.handoff.mode == "shadow" and not service.handoff.promotion_authorized
    assert digest(run.freeze) in service.handoff.served_model


@pytest.mark.integration
def test_actual_ratchet_systemone_adapter():
    bun = shutil.which("bun")
    ratchet = Path(os.environ.get("NRLY_RATCHET_REPO", "../ratchet")).resolve()
    if not bun or not (ratchet / "src/oracles/systemone.ts").is_file():
        pytest.skip("optional sibling Ratchet source and Bun are required")
    model = backend()
    spec = protocol()
    run = run_decision_study(model, spec, WeightProgram(model.identity.fingerprint).build())
    service = ShadowDecisionService(model, run)
    token = secrets.token_urlsafe(32)
    with shadow_server(service, token) as url:
        result = subprocess.run(
            [bun, "tests/decision_ratchet.ts"],
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **os.environ,
                "NRLY_TEST_ENDPOINT": url,
                "NRLY_TEST_TOKEN": token,
                "NRLY_TEST_MODEL": service.handoff.served_model,
                "NRLY_TEST_STATE": spec.cases[0].state,
                "NRLY_TEST_QUESTION": spec.question,
                "NRLY_RATCHET_REPO": str(ratchet),
            },
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["probability"] == pytest.approx(0.94)
