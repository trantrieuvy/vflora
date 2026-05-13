"""Cumulative residual LoRA adapter variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLoRALayer(nn.Module):
    """LoRA residual block with optional frozen cumulative adapter state."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        nonlinear: bool = False,
        A_frozen: torch.Tensor | None = None,
        B_frozen: torch.Tensor | None = None,
        frozen_scaling: float | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if (A_frozen is None) != (B_frozen is None):
            raise ValueError("A_frozen and B_frozen must be provided together")
        if A_frozen is not None and frozen_scaling is None:
            raise ValueError("frozen_scaling is required with frozen adapters")

        self.linear = linear
        self.rank = rank
        self.new_scaling = alpha / rank
        self.frozen_scaling = frozen_scaling
        self.nonlinear = nonlinear
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.A_new = nn.Parameter(torch.empty(rank, linear.in_features, dtype=torch.float32))
        nn.init.normal_(self.A_new, std=init_std)
        self.B_new = nn.Parameter(torch.zeros(linear.out_features, rank, dtype=torch.float32))

        if A_frozen is None:
            self.A_frozen = None
            self.B_frozen = None
        else:
            self.register_buffer("A_frozen", A_frozen.detach().clone().to(dtype=torch.float32))
            self.register_buffer("B_frozen", B_frozen.detach().clone().to(dtype=torch.float32))

        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        y = self.linear(x)
        adapter_input = x.to(dtype=torch.float32)

        if self.A_frozen is not None:
            frozen_hidden = F.linear(adapter_input, self.A_frozen)
            if self.nonlinear:
                frozen_hidden = F.gelu(frozen_hidden)
            frozen_update = F.linear(frozen_hidden, self.B_frozen)
            y = y + (self.frozen_scaling * frozen_update).to(dtype=y.dtype)

        new_hidden = F.linear(self.dropout(adapter_input), self.A_new)
        if self.nonlinear:
            new_hidden = F.gelu(new_hidden)
        new_update = F.linear(new_hidden, self.B_new)
        return y + (self.new_scaling * new_update).to(dtype=y.dtype)


def inject_residual_adapters(
    model,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    nonlinear: bool,
    A_frozen: Mapping[str, torch.Tensor] | None = None,
    B_frozen: Mapping[str, torch.Tensor] | None = None,
    frozen_scaling: float | None = None,
) -> tuple[object, int]:
    """Replace target linear modules with residual LoRA layers."""
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(target in name for target in target_modules):
            continue
        replacement = ResidualLoRALayer(
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            nonlinear=nonlinear,
            A_frozen=A_frozen.get(name) if A_frozen else None,
            B_frozen=B_frozen.get(name) if B_frozen else None,
            frozen_scaling=frozen_scaling,
        ).to(module.weight.device)
        _set_submodule(model, name, replacement)
        count += 1
    return model, count


def adapter_state_dict(model) -> dict[str, torch.Tensor]:
    """Return trainable residual adapter weights from an injected model."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.endswith(".A_new") or name.endswith(".B_new")
    }


def split_adapter_state(state: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    A_state: dict[str, torch.Tensor] = {}
    B_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.endswith(".A_new"):
            A_state[key[:-6]] = value.detach().cpu().clone()
        elif key.endswith(".B_new"):
            B_state[key[:-6]] = value.detach().cpu().clone()
    validate_adapter_pair(A_state, B_state, "adapter state")
    return A_state, B_state


def join_adapter_state(A_state: Mapping[str, torch.Tensor], B_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    validate_adapter_pair(A_state, B_state, "adapter state")
    state: dict[str, torch.Tensor] = {}
    for name in sorted(A_state):
        state[f"{name}.A_new"] = A_state[name].detach().cpu().clone()
        state[f"{name}.B_new"] = B_state[name].detach().cpu().clone()
    return state


def accumulate_adapters(
    previous_A: Mapping[str, torch.Tensor] | None,
    previous_B: Mapping[str, torch.Tensor] | None,
    round_A: Mapping[str, torch.Tensor],
    round_B: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Append the current round adapter to the frozen cumulative adapter."""
    validate_adapter_pair(round_A, round_B, "round")
    if (previous_A is None) != (previous_B is None):
        raise ValueError("previous_A and previous_B must be provided together")
    if previous_A is None:
        return (
            {name: value.detach().cpu().clone() for name, value in round_A.items()},
            {name: value.detach().cpu().clone() for name, value in round_B.items()},
        )

    validate_adapter_pair(previous_A, previous_B, "previous")
    if set(previous_A) != set(round_A):
        raise ValueError("previous and round adapters target different modules")

    merged_A: dict[str, torch.Tensor] = {}
    merged_B: dict[str, torch.Tensor] = {}
    for name in sorted(round_A):
        old_A = previous_A[name].detach().cpu()
        old_B = previous_B[name].detach().cpu()
        new_A = round_A[name].detach().cpu()
        new_B = round_B[name].detach().cpu()
        if old_A.shape[1] != new_A.shape[1]:
            raise ValueError(f"A input dimension mismatch for {name}")
        if old_B.shape[0] != new_B.shape[0]:
            raise ValueError(f"B output dimension mismatch for {name}")
        merged_A[name] = torch.cat([old_A, new_A], dim=0).clone()
        merged_B[name] = torch.cat([old_B, new_B], dim=1).clone()
    return merged_A, merged_B


def validate_adapter_pair(
    A_state: Mapping[str, torch.Tensor],
    B_state: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    if set(A_state) != set(B_state):
        raise ValueError(f"{label} has mismatched A/B module keys")
    for name in A_state:
        A = A_state[name]
        B = B_state[name]
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(f"{label} module {name} must contain 2D A and B tensors")
        if A.shape[0] != B.shape[1]:
            raise ValueError(f"{label} module {name} has inconsistent rank")


def _set_submodule(model, dotted_name: str, module: nn.Module) -> None:
    parent = model
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)

