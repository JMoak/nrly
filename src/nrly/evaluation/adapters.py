"""Versioned deterministic output checks, with no judge model or benchmark claim."""

from __future__ import annotations

import json
from decimal import Decimal, DecimalException
from typing import Any

from nrly.contracts import CaseSpec


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError("non-finite JSON number: " + value)


def _decimal(value: str) -> Decimal:
    if len(value) > 4096:
        raise ValueError("JSON numeric literal exceeds scorer size ceiling")
    try:
        number = Decimal(value)
    except DecimalException as error:
        raise ValueError("JSON numeric literal is unsupported") from error
    exponent = number.as_tuple().exponent
    if not number.is_finite() or not isinstance(exponent, int) or abs(exponent) > 10000:
        raise ValueError("JSON number exceeds scorer exponent ceiling")
    return number


def _json(value: str) -> Any:
    if len(value) > 1000000:
        raise ValueError("JSON exceeds scorer size ceiling")
    result = json.loads(
        value,
        object_pairs_hook=_unique_object,
        parse_constant=_nonfinite,
        parse_float=_decimal,
        parse_int=_decimal,
    )

    # Decimal parsing preserves exact finite JSON numbers; bound recursive comparison.
    def validate(item: Any, depth: int = 0) -> None:
        if depth > 64:
            raise ValueError("JSON exceeds scorer nesting ceiling")
        if isinstance(item, list):
            for child in item:
                validate(child, depth + 1)
        elif isinstance(item, dict):
            for child in item.values():
                validate(child, depth + 1)

    validate(result)
    return result


def _equal_json(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal_json(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equal_json(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def score_adapter(case: CaseSpec, output: str) -> tuple[bool | None, str]:
    adapter = case.adapter
    if adapter is None:
        raise ValueError("task adapter is missing")
    if adapter.kind == "multiple_choice":
        if case.expected not in adapter.choices:
            return None, "expected answer is not a registered choice label"
        actual = output.strip()
        if actual not in adapter.choices:
            return False, "output is not exactly one registered choice label"
        passed = actual == case.expected
        return passed, "choice match" if passed else "choice mismatch"
    if adapter.kind == "json":
        if case.expected is None:
            return None, "expected JSON answer is missing"
        try:
            expected = _json(case.expected)
        except (ValueError, RecursionError):
            return None, "expected answer is not supported strict JSON"
        try:
            actual_json = _json(output)
        except (ValueError, RecursionError):
            return False, "output is not supported strict JSON"
        passed = _equal_json(actual_json, expected)
        return passed, "JSON structural match" if passed else "JSON structural mismatch"
    failures = []
    if any(text not in output for text in adapter.required):
        failures.append("required literal missing")
    if any(text in output for text in adapter.forbidden):
        failures.append("forbidden literal present")
    if adapter.max_chars is not None and len(output) > adapter.max_chars:
        failures.append("character ceiling exceeded")
    return not failures, "; ".join(failures) or "all registered literal constraints satisfied"
