"""Nonlinear RoLoRA adapter utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fed_adapter.adapters.ffa import apply_activation


class NonlinearRoLoRALayer(nn.Module):
    """LoRA residual that trains one nonlinear factor at a time."""

    def __init__(
        self,
        linear: nn.Linear,
        A_state: torch.Tensor,
        B_state: torch.Tensor,
        scaling: float,
        dropout: float = 0.0,
        activation: str = "gelu",
        train_factor: str | None = None,
    ) -> None:
        super().__init__()
        if train_factor not in {None, "A", "B"}:
            raise ValueError("train_factor must be None, 'A', or 'B'")
        if A_state.ndim != 2 or B_state.ndim != 2:
            raise ValueError("RoLoRA A and B states must be 2D tensors")
        if A_state.shape[0] != B_state.shape[1]:
            raise ValueError("RoLoRA A/B states have inconsistent rank")

        self.linear = linear
        self.scaling = float(scaling)
        self.activation = (activation or "none").lower()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        A = A_state.detach().clone().to(dtype=torch.float32)
        B = B_state.detach().clone().to(dtype=torch.float32)

        if train_factor == "A":
            self.A = nn.Parameter(A)
            self.register_buffer("B", B)
        elif train_factor == "B":
            self.register_buffer("A", A)
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("A", A)
            self.register_buffer("B", B)

        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        y = self.linear(x)
        update = nonlinear_lora_delta(
            self.dropout(x),
            self.A,
            self.B,
            self.activation,
        )
        return y + (self.scaling * update).to(dtype=y.dtype)


def nonlinear_lora_delta(
    x: torch.Tensor,
    A_state: torch.Tensor,
    B_state: torch.Tensor,
    activation: str = "gelu",
) -> torch.Tensor:
    """Compute B activation(A x) for a nonlinear LoRA residual."""
    hidden = F.linear(x.to(dtype=A_state.dtype), A_state)
    hidden = apply_activation(hidden, activation)
    return F.linear(hidden.to(dtype=B_state.dtype), B_state)


def inject_rolora_adapters(
    model,
    target_modules: Sequence[str],
    A_state: Mapping[str, torch.Tensor],
    B_state: Mapping[str, torch.Tensor],
    scaling: float,
    dropout: float,
    activation: str = "gelu",
    train_factor: str | None = None,
) -> tuple[object, int]:
    """Replace target linear modules with nonlinear RoLoRA layers."""
    validate_rolora_state(A_state, B_state, "RoLoRA state")
    count = 0
    for name, module in list(_target_linear_modules(model, target_modules)):
        replacement = NonlinearRoLoRALayer(
            module,
            A_state=A_state[name],
            B_state=B_state[name],
            scaling=scaling,
            dropout=dropout,
            activation=activation,
            train_factor=train_factor,
        ).to(module.weight.device)
        _set_submodule(model, name, replacement)
        count += 1
    return model, count


def rolora_active_state_dict(model, train_factor: str) -> dict[str, torch.Tensor]:
    """Return the active RoLoRA factor trained by a client."""
    if train_factor not in {"A", "B"}:
        raise ValueError("train_factor must be 'A' or 'B'")
    suffix = f".{train_factor}"
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.endswith(suffix)
    }


def split_rolora_factor_state(
    state: Mapping[str, torch.Tensor],
    factor: str,
) -> dict[str, torch.Tensor]:
    """Convert active-factor state-dict keys into module-keyed tensors."""
    if factor not in {"A", "B"}:
        raise ValueError("factor must be 'A' or 'B'")
    suffix = f".{factor}"
    return {
        key[: -len(suffix)]: value.detach().cpu().clone()
        for key, value in state.items()
        if key.endswith(suffix)
    }


def join_rolora_state(
    A_state: Mapping[str, torch.Tensor],
    B_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert module-keyed RoLoRA tensors into state-dict keys."""
    validate_rolora_state(A_state, B_state, "RoLoRA state")
    state: dict[str, torch.Tensor] = {}
    for name in sorted(A_state):
        state[f"{name}.A"] = A_state[name].detach().cpu().clone()
        state[f"{name}.B"] = B_state[name].detach().cpu().clone()
    return state


def split_rolora_state(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Convert state-dict RoLoRA keys into module-keyed A/B tensors."""
    A_state: dict[str, torch.Tensor] = {}
    B_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith(".A"):
            A_state[key[:-2]] = value.detach().cpu().clone()
        elif key.endswith(".B"):
            B_state[key[:-2]] = value.detach().cpu().clone()
    validate_rolora_state(A_state, B_state, "RoLoRA state")
    return A_state, B_state


def validate_rolora_state(
    A_state: Mapping[str, torch.Tensor],
    B_state: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    """Validate module-keyed nonlinear RoLoRA A/B tensors."""
    if set(A_state) != set(B_state):
        raise ValueError(f"{label} has mismatched A/B module keys")
    for name in A_state:
        A = A_state[name]
        B = B_state[name]
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"{label} module {name} must contain 2D A and B tensors")
        if A.shape[0] != B.shape[1]:
            raise ValueError(f"{label} module {name} has inconsistent rank")


def _target_linear_modules(model, target_modules: Sequence[str]):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            yield name, module


def _set_submodule(model, dotted_name: str, module: nn.Module) -> None:
    parent = model
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)
