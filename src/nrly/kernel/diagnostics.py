from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from nrly.contracts import (
    ContinuationScore,
    DiagnosticSpec,
    NrlyError,
    TokenAlternative,
    TokenObservation,
)


def score_continuation(
    logits: Tensor,
    continuation_ids: Tensor,
    prompt_tokens: int,
    spec: DiagnosticSpec,
    decode: Callable[[int], str],
    *,
    logits_start: int = 0,
) -> ContinuationScore:
    """Score causal next-token rows without allocating a full float32 log-probability tensor."""
    count = continuation_ids.numel()
    if not 1 <= count <= spec.max_tokens:
        raise NrlyError("budget_exceeded", "continuation exceeds the diagnostic token budget")
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] < prompt_tokens + count - 1 - logits_start
        or prompt_tokens < 1
        or not 0 <= logits_start <= prompt_tokens - 1
    ):
        raise NrlyError(
            "invalid_backend_result", "causal logits do not cover continuation positions"
        )
    observations = []
    for offset, token_id in enumerate(continuation_ids.tolist()):
        row = logits[0, prompt_tokens + offset - 1 - logits_start].float()
        if not bool(torch.isfinite(row).all()):
            raise NrlyError("nonfinite_tensor", "diagnostic logits contain nonfinite values")
        if not 0 <= token_id < row.numel():
            raise NrlyError("tokenization_failed", "continuation token is outside model vocabulary")
        probabilities = row.log_softmax(dim=-1)
        selected = float(probabilities[token_id].item())
        values, indices = probabilities.topk(min(spec.top_k, row.numel()))
        alternatives = [
            TokenAlternative(token_id=index, text=decode(index), logprob=value)
            for index, value in zip(indices.tolist(), values.tolist(), strict=True)
        ]
        observations.append(
            TokenObservation(
                position=prompt_tokens + offset,
                token_id=token_id,
                text=decode(token_id),
                logprob=selected,
                rank=int(torch.count_nonzero(row > row[token_id]).item()) + 1,
                top_k=alternatives,
            )
        )
    total = math.fsum(item.logprob for item in observations)
    return ContinuationScore(
        prompt_tokens=prompt_tokens,
        continuation_tokens=count,
        sum_logprob=total,
        mean_logprob=total / count,
        tokens=observations,
    )
