"""Functional distillation helpers for nonlinear adapter compression."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from fed_adapter.adapters.rolora import nonlinear_lora_delta, validate_rolora_state


def distill_nonlinear_lora_modules(
    activation_inputs: Mapping[str, torch.Tensor],
    teacher_A: Mapping[str, torch.Tensor],
    teacher_B: Mapping[str, torch.Tensor],
    init_A: Mapping[str, torch.Tensor],
    init_B: Mapping[str, torch.Tensor],
    *,
    activation: str = "gelu",
    steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    max_relative_mse: float = 0.25,
    strict: bool = False,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, object]]:
    """Fit rank-bounded nonlinear student adapters to exact teacher deltas."""
    validate_rolora_state(teacher_A, teacher_B, "teacher")
    validate_rolora_state(init_A, init_B, "student initialization")
    if set(activation_inputs) != set(teacher_A):
        raise ValueError("activation input module keys must match teacher keys")

    student_A: dict[str, torch.Tensor] = {}
    student_B: dict[str, torch.Tensor] = {}
    module_metrics: dict[str, dict[str, float | int | bool]] = {}
    total_sse_before = 0.0
    total_sse_after = 0.0
    total_energy = 0.0
    total_elements = 0

    for offset, name in enumerate(sorted(teacher_A)):
        inputs = activation_inputs[name].detach().to(dtype=torch.float32)
        if inputs.ndim < 2:
            raise ValueError(f"activation inputs for {name} must have at least 2 dimensions")
        inputs = inputs.reshape(-1, inputs.shape[-1])
        if inputs.shape[0] == 0:
            raise ValueError(f"activation inputs for {name} are empty")

        A_c, B_c, metrics = _distill_one_module(
            inputs,
            teacher_A[name],
            teacher_B[name],
            init_A[name],
            init_B[name],
            activation=activation,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed + offset,
            device=device,
        )
        student_A[name] = A_c
        student_B[name] = B_c
        module_metrics[name] = metrics
        total_sse_before += metrics["sse_before"]
        total_sse_after += metrics["sse_after"]
        total_energy += metrics["teacher_energy"]
        total_elements += metrics["num_elements"]

    mse_before = total_sse_before / max(total_elements, 1)
    mse_after = total_sse_after / max(total_elements, 1)
    relative_mse_after = _relative_mse(total_sse_after, total_energy)
    failed = not torch.isfinite(torch.tensor(relative_mse_after)).item()
    warned = bool(relative_mse_after > max_relative_mse)
    if failed:
        raise RuntimeError("RoLoRA distillation produced non-finite relative MSE")
    if strict and warned:
        raise RuntimeError(
            "RoLoRA distillation relative MSE "
            f"{relative_mse_after:.6g} exceeds threshold {max_relative_mse:.6g}"
        )

    metrics: dict[str, object] = {
        "mse_before": mse_before,
        "mse_after": mse_after,
        "relative_mse_after": relative_mse_after,
        "max_relative_mse": float(max_relative_mse),
        "warned": warned,
        "strict": bool(strict),
        "num_modules": len(module_metrics),
        "num_tokens": {name: int(activation_inputs[name].reshape(-1, activation_inputs[name].shape[-1]).shape[0]) for name in module_metrics},
        "modules": {
            name: {
                key: value
                for key, value in values.items()
                if key not in {"sse_before", "sse_after", "teacher_energy", "num_elements"}
            }
            for name, values in module_metrics.items()
        },
    }
    return student_A, student_B, metrics


def _distill_one_module(
    inputs: torch.Tensor,
    teacher_A: torch.Tensor,
    teacher_B: torch.Tensor,
    init_A: torch.Tensor,
    init_B: torch.Tensor,
    *,
    activation: str,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | bool]]:
    device = torch.device(device)
    x = inputs.to(device=device)
    tA = teacher_A.detach().to(device=device, dtype=torch.float32)
    tB = teacher_B.detach().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        teacher_delta = nonlinear_lora_delta(x, tA, tB, activation)

    A = torch.nn.Parameter(init_A.detach().clone().to(device=device, dtype=torch.float32))
    B = torch.nn.Parameter(init_B.detach().clone().to(device=device, dtype=torch.float32))

    with torch.no_grad():
        before = F.mse_loss(nonlinear_lora_delta(x, A, B, activation), teacher_delta)

    optimizer = torch.optim.AdamW([A, B], lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    n = x.shape[0]
    batch_size = max(1, min(batch_size, n))
    permutation = torch.randperm(n, generator=generator)
    cursor = 0
    for _ in range(max(0, steps)):
        if cursor + batch_size > n:
            permutation = torch.randperm(n, generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + batch_size].to(device=device)
        cursor += batch_size
        prediction = nonlinear_lora_delta(x.index_select(0, indices), A, B, activation)
        target = teacher_delta.index_select(0, indices)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        after = F.mse_loss(nonlinear_lora_delta(x, A, B, activation), teacher_delta)
        teacher_energy = teacher_delta.pow(2).sum().item()
        sse_before = before.item() * teacher_delta.numel()
        sse_after = after.item() * teacher_delta.numel()
        relative_after = _relative_mse(sse_after, teacher_energy)

    return (
        A.detach().cpu().clone(),
        B.detach().cpu().clone(),
        {
            "mse_before": float(before.item()),
            "mse_after": float(after.item()),
            "relative_mse_after": float(relative_after),
            "sse_before": float(sse_before),
            "sse_after": float(sse_after),
            "teacher_energy": float(teacher_energy),
            "num_elements": int(teacher_delta.numel()),
        },
    )


def _relative_mse(sse: float, energy: float) -> float:
    eps = 1e-12
    if energy <= eps:
        return 0.0 if sse <= eps else float("inf")
    return float(sse / energy)
