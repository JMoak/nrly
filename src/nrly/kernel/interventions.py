from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from nrly.contracts import (
    CaptureSpec,
    NrlyError,
    ProgramSpec,
    TensorCapture,
    TensorSlice,
    WeightChange,
    WeightOperationSpec,
)


def tensor_digest(model: nn.Module) -> str:
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode("utf-8"))
        result.update(str(tuple(value.shape)).encode("ascii"))
        result.update(str(value.dtype).encode("ascii"))
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        result.update(memoryview(raw.numpy()))
    return result.hexdigest()


@contextmanager
def intervene(
    model: nn.Module, program: ProgramSpec, fingerprint: str
) -> Iterator[list[WeightChange]]:
    if program.expected_base is not None and program.expected_base != fingerprint:
        raise NrlyError("base_mismatch", "program expected_base does not match the loaded model")
    if program.kind == "identity":
        yield []
        return
    if program.kind.startswith("weight_") or program.kind == "pipeline":
        operations = (
            program.operations
            if program.kind == "pipeline"
            else [
                WeightOperationSpec.model_validate(
                    program.model_dump(exclude={"expected_base", "operations"})
                )
            ]
        )
        with _weight_intervention(model, operations, program.kind == "pipeline") as changes:
            yield changes
        return
    target = _module(model, program.target)

    def transform(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        if not isinstance(output, Tensor):
            raise NrlyError("unsupported_operation", "activation targets must return a tensor")
        if program.position == "last" and output.ndim != 3:
            raise NrlyError(
                "target_mismatch",
                "last requires a [batch, sequence, hidden] activation",
            )
        selected = output[:, -1:, :] if program.position == "last" else output
        if program.kind == "activation_scale":
            changed = selected * program.factor
        elif program.kind == "activation_add":
            accepted = [list(selected.shape)]
            if selected.ndim:
                accepted.append([selected.shape[-1]])
            if program.shape not in accepted:
                raise NrlyError(
                    "target_mismatch",
                    "add payload must match the activation or hidden-width vector",
                )
            addition = torch.tensor(
                program.values, dtype=output.dtype, device=output.device
            ).reshape(program.shape)
            changed = selected + addition
        else:
            raise NrlyError(
                "unsupported_operation",
                f"unsupported activation operation: {program.kind}",
            )
        _finite(changed)
        if program.position == "all":
            return changed
        result = output.clone()
        result[:, -1:, :] = changed
        return result

    hook = target.register_forward_hook(transform)
    try:
        yield []
    finally:
        hook.remove()


@contextmanager
def capture_activations(
    model: nn.Module, specs: list[CaptureSpec]
) -> Iterator[dict[str, TensorCapture]]:
    targets = [(spec, _module(model, spec.target)) for spec in specs]
    results: dict[str, TensorCapture] = {}
    hooks = []

    def recorder(
        spec: CaptureSpec,
    ) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        invocation_index = -1

        def record(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            nonlocal invocation_index
            invocation_index += 1
            if spec.invocation == "first" and invocation_index > 0:
                return
            if spec.source == "input":
                output = _inputs[0] if _inputs else None
            if not isinstance(output, Tensor):
                raise NrlyError("unsupported_operation", "capture source must be a tensor")
            value = output.detach()
            if spec.token_positions:
                if value.ndim != 3 or value.shape[0] != 1:
                    raise NrlyError(
                        "target_mismatch", "token captures require [1, sequence, hidden] tensors"
                    )
                if max(spec.token_positions) >= value.shape[1]:
                    raise NrlyError(
                        "target_mismatch", "capture token position is outside invocation"
                    )
                value = value[:, spec.token_positions, :]
            selected = value.reshape(-1)[: spec.max_elements].float().cpu()
            _finite(selected)
            extended = bool(
                spec.token_positions or spec.invocation != "latest" or spec.source != "output"
            )
            results[spec.target] = TensorCapture(
                target=spec.target,
                shape=list(output.shape),
                dtype=str(output.dtype),
                values=selected.tolist(),
                truncated=output.numel() > selected.numel(),
                token_positions=spec.token_positions,
                selection_shape=list(value.shape) if spec.token_positions else [],
                invocation_index=invocation_index if extended else None,
                source=spec.source if extended else None,
            )

        return record

    def input_recorder(
        spec: CaptureSpec,
    ) -> Callable[[nn.Module, tuple[object, ...]], None]:
        capture = recorder(spec)

        def record_input(module: nn.Module, inputs: tuple[object, ...]) -> None:
            capture(module, inputs, None)

        return record_input

    try:
        for spec, target in targets:
            if spec.source == "input":
                hooks.append(target.register_forward_pre_hook(input_recorder(spec)))
            else:
                hooks.append(target.register_forward_hook(recorder(spec)))
        yield results
        missing = [spec.target for spec in specs if spec.target not in results]
        if missing:
            raise NrlyError("target_mismatch", f"capture targets were not invoked: {missing}")
    finally:
        for hook in hooks:
            hook.remove()


@dataclass
class _PreparedWeight:
    parameter: nn.Parameter
    original: Tensor
    replacement: Tensor


# Bound the stored-weight footprint before cloning. This is not a promise about
# peak memory: dtype promotion, products, snapshots and digests also need memory.
MAX_EDITED_WEIGHT_BYTES = 512 * 1024 * 1024


@contextmanager
def _weight_intervention(
    model: nn.Module, operations: list[WeightOperationSpec], pipeline: bool
) -> Iterator[list[WeightChange]]:
    # Resolve and compute every operation before the first write. Repeated targets
    # read the preceding candidate, while the rollback snapshot remains the base.
    entries = list(model.named_parameters(remove_duplicate=False))
    parameters = dict(entries)
    targets = {str(operation.target) for operation in operations}
    unknown = targets - parameters.keys()
    if unknown:
        raise NrlyError("target_mismatch", f"unknown parameter: {sorted(unknown)[0]}")
    touched_bytes = sum(
        parameters[name].numel() * parameters[name].element_size() for name in targets
    )
    if touched_bytes > MAX_EDITED_WEIGHT_BYTES:
        raise NrlyError(
            "budget_exceeded", "weight program exceeds 512 MiB of distinct target tensors"
        )
    all_storage = [*entries, *model.named_buffers(remove_duplicate=False)]
    prepared: dict[str, _PreparedWeight] = {}
    changes: list[WeightChange] = []
    last_changes: dict[str, int] = {}
    for operation_index, operation in enumerate(operations):
        name = str(operation.target)
        if name not in parameters:
            raise NrlyError("target_mismatch", f"unknown parameter: {name}")
        parameter = parameters[name]
        if not parameter.is_floating_point() or parameter.layout != torch.strided:
            raise NrlyError(
                "unsupported_operation", "weight edits require dense real floating tensors"
            )
        if name not in prepared:
            storage = parameter.untyped_storage().data_ptr()
            aliases = [
                alias
                for alias, candidate in all_storage
                if candidate.device == parameter.device
                and candidate.untyped_storage().data_ptr() == storage
            ]
            if len(aliases) != 1:
                raise NrlyError(
                    "unsupported_operation", "tied parameter edits require an explicit alias policy"
                )
            original = parameter.detach().clone()
            _finite(original)
            prepared[name] = _PreparedWeight(parameter, original, original)
        staged = prepared[name]
        before = staged.replacement
        replacement = _replacement(before, operation)
        _finite(replacement)
        selected = _evidence_indices(operation, before.shape)
        changes.append(
            WeightChange(
                target=name,
                shape=list(before.shape),
                dtype=str(before.dtype),
                before_sha256=_value_digest(before),
                after_sha256=_value_digest(replacement),
                changed_elements=_changed_elements(before, replacement),
                total_elements=before.numel(),
                before_preview=before.reshape(-1)[:32].double().cpu().tolist(),
                after_preview=replacement.reshape(-1)[:32].double().cpu().tolist(),
                edited_indices=selected,
                edited_before=before.reshape(-1)[selected].double().cpu().tolist(),
                edited_after=replacement.reshape(-1)[selected].double().cpu().tolist(),
                edited_coordinates=[flat_coordinate(i, list(before.shape)) for i in selected],
                operation_index=operation_index if pipeline else None,
                operation_kind=operation.kind
                if pipeline
                or operation.kind not in {"weight_scale", "weight_replace", "weight_patch"}
                else None,
            )
        )
        staged.replacement = replacement
        last_changes[name] = operation_index
    if pipeline:
        # Per-operation hashes describe ordered candidate arithmetic. Only these
        # final per-target records describe the net program actually published.
        for name, index in last_changes.items():
            staged = prepared[name]
            changes[index] = changes[index].model_copy(
                update={
                    "net_changed_elements": _changed_elements(staged.original, staged.replacement),
                    "net_before_sha256": _value_digest(staged.original),
                    "net_after_sha256": _value_digest(staged.replacement),
                }
            )
    # Include the attempted target before copy_: a failing device copy may have
    # partially written it. Always attempt every restoration, even if one fails.
    attempted: list[_PreparedWeight] = []
    try:
        with torch.no_grad():
            for staged in prepared.values():
                attempted.append(staged)
                staged.parameter.copy_(staged.replacement)
        yield changes
    finally:
        failures: list[str] = []
        with torch.no_grad():
            for staged in reversed(attempted):
                try:
                    staged.parameter.copy_(staged.original)
                    if _value_digest(staged.parameter) != _value_digest(staged.original):
                        failures.append("restored bytes differ")
                except Exception as error:
                    failures.append(str(error))
        if failures:
            raise NrlyError("restoration_failed", "original weights could not be restored")


def _replacement(original: Tensor, program: WeightOperationSpec) -> Tensor:
    # Historical scale/replace/patch retain their original dtype semantics.
    if program.kind == "weight_scale":
        return original * program.factor
    if list(original.shape) != program.shape:
        raise NrlyError("target_mismatch", "payload shape must exactly match the parameter")
    if program.kind in {"weight_patch", "weight_replace"}:
        payload = torch.tensor(program.values, device=original.device, dtype=original.dtype)
        if program.kind == "weight_replace":
            return payload.reshape(program.shape)
        replacement = original.contiguous().clone()
        indices = torch.tensor(program.indices, device=original.device, dtype=torch.long)
        replacement.view(-1)[indices] = payload
        return replacement
    # New arithmetic accumulates in at least FP32 and is rounded once into the
    # stored dtype. Evidence is measured after that final conversion.
    compute_dtype = torch.float64 if original.dtype == torch.float64 else torch.float32
    if program.kind == "weight_slice_scale":
        replacement = original.clone()
        selection = tensor_slices(program.slices, list(original.shape))
        replacement[selection] = (original[selection].to(compute_dtype) * program.factor).to(
            original.dtype
        )
        return replacement
    if program.kind in {"weight_add", "weight_sparse_add"}:
        if program.factor == 0:
            return original.clone()
        payload = torch.tensor(program.values, device=original.device, dtype=compute_dtype)
        payload = payload * program.factor
        if program.kind == "weight_sparse_add":
            replacement = original.contiguous().clone()
            indices = torch.tensor(program.indices, device=original.device, dtype=torch.long)
            current = original.reshape(-1)[indices]
            updated = (current.to(compute_dtype) + payload).to(original.dtype)
            # Zero additive controls retain signed-zero bytes as well.
            payload = torch.where(payload == 0, current, updated)
            replacement.view(-1)[indices] = payload
            return replacement
        delta = payload.reshape(program.shape)
    elif program.kind == "weight_low_rank":
        if program.factor == 0:
            return original.clone()
        rank = cast(int, program.rank)
        left = torch.tensor(program.left, device=original.device, dtype=compute_dtype).reshape(
            program.shape[0], rank
        )
        right = torch.tensor(program.right, device=original.device, dtype=compute_dtype).reshape(
            rank, program.shape[1]
        )
        delta = (left @ right) * program.factor
    else:
        raise NrlyError("unsupported_operation", f"unsupported weight operation: {program.kind}")
    updated = (original.to(compute_dtype) + delta).to(original.dtype)
    return torch.where(delta == 0, original, updated)


def flat_coordinate(index: int, shape: list[int]) -> list[int]:
    coordinate: list[int] = []
    for width in reversed(shape):
        coordinate.append(index % width)
        index //= width
    return list(reversed(coordinate))


def tensor_slices(parts: list[TensorSlice], shape: list[int]) -> tuple[slice, ...]:
    if len(parts) != len(shape) or any(
        part.stop > width for part, width in zip(parts, shape, strict=True)
    ):
        raise NrlyError(
            "target_mismatch", "slices must match tensor rank and stay within its shape"
        )
    return tuple(slice(part.start, part.stop, part.step) for part in parts)


def _evidence_indices(program: WeightOperationSpec, shape: torch.Size) -> list[int]:
    if program.kind in {"weight_patch", "weight_sparse_add"}:
        return program.indices[:32]
    if program.kind == "weight_slice_scale":
        result = []
        counts = [len(range(part.start, part.stop, part.step)) for part in program.slices]
        for offset in range(min(32, math.prod(counts))):
            coordinate = [
                part.start + value * part.step
                for value, part in zip(flat_coordinate(offset, counts), program.slices, strict=True)
            ]
            index = 0
            for value, width in zip(coordinate, shape, strict=True):
                index = index * width + value
            result.append(index)
        return result
    return []


def _module(model: nn.Module, target: str | None) -> nn.Module:
    if target is None or not target:
        raise NrlyError("target_mismatch", "an exact non-root module path is required")
    modules = dict(model.named_modules())
    if target not in modules:
        raise NrlyError("target_mismatch", f"unknown module: {target}")
    return cast(nn.Module, modules[target])


def _finite(value: Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise NrlyError("nonfinite_tensor", "the operation produced non-finite tensor values")


def _value_digest(value: Tensor) -> str:
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


def _changed_elements(before: Tensor, after: Tensor) -> int:
    width = before.element_size()
    before_bytes = before.detach().contiguous().reshape(-1).view(torch.uint8).reshape(-1, width)
    after_bytes = after.detach().contiguous().reshape(-1).view(torch.uint8).reshape(-1, width)
    return int(torch.count_nonzero(torch.any(before_bytes != after_bytes, dim=1)).item())
