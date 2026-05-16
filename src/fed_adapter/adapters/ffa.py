"""Frozen-A LoRA adapter utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFALoRALayer(nn.Module):
    """LoRA-style residual with frozen A and trainable B."""

    def __init__(
        self,
        linear: nn.Linear,
        A_frozen: torch.Tensor,
        B_initial: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.linear = linear
        self.scaling = float(scaling)
        self.activation = (activation or "none").lower()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.register_buffer("A_frozen", A_frozen.detach().clone().to(dtype=torch.float32))
        if B_initial is None:
            B = torch.zeros(linear.out_features, self.A_frozen.shape[0], dtype=torch.float32)
        else:
            B = B_initial.detach().clone().to(dtype=torch.float32)
        self.B = nn.Parameter(B)

        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        y = self.linear(x)
        hidden = F.linear(self.dropout(x).to(dtype=self.A_frozen.dtype), self.A_frozen)
        hidden = apply_activation(hidden, self.activation)
        update = F.linear(hidden.to(dtype=self.B.dtype), self.B)
        return y + (self.scaling * update).to(dtype=y.dtype)


def apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply an FFA hidden activation."""
    activation = (activation or "none").lower()
    if activation in {"none", "linear", "identity"}:
        return x
    if activation == "gelu":
        return F.gelu(x)
    if activation == "relu":
        return F.relu(x)
    if activation == "silu":
        return F.silu(x)
    if activation == "tanh":
        return torch.tanh(x)
    raise ValueError(f"Unknown FFA activation: {activation!r}")


def init_frozen_A(
    model,
    target_modules: Sequence[str],
    rank: int,
    seed: int,
    init_std: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Create seeded frozen A matrices for every target linear module."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return {
        name: torch.randn(
            rank,
            module.in_features,
            generator=generator,
            dtype=torch.float32,
        )
        * init_std
        for name, module in _target_linear_modules(model, target_modules)
    }


def init_zero_B(model, target_modules: Sequence[str], rank: int) -> dict[str, torch.Tensor]:
    """Create zero-initialized B matrices matching target linear modules."""
    return {
        name: torch.zeros(module.out_features, rank, dtype=torch.float32)
        for name, module in _target_linear_modules(model, target_modules)
    }


def inject_ffa_adapters(
    model,
    target_modules: Sequence[str],
    A_frozen: Mapping[str, torch.Tensor],
    B_state: Mapping[str, torch.Tensor],
    scaling: float,
    dropout: float,
    activation: str = "gelu",
    client_rank: int | None = None,
) -> tuple[object, int]:
    """Replace target linear modules with FFA layers."""
    count = 0
    for name, module in list(_target_linear_modules(model, target_modules)):
        A = A_frozen[name]
        B = B_state[name]
        if client_rank is not None:
            A = A[:client_rank, :]
            B = B[:, :client_rank]
        replacement = FFALoRALayer(
            module,
            A_frozen=A,
            B_initial=B,
            scaling=scaling,
            dropout=dropout,
            activation=activation,
        ).to(module.weight.device)
        _set_submodule(model, name, replacement)
        count += 1
    return model, count


def ffa_B_state_dict(model) -> dict[str, torch.Tensor]:
    """Return trainable FFA B parameters from an injected model."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.endswith(".B")
    }


def split_ffa_B_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert state-dict B keys into module-keyed B tensors."""
    return {
        key[:-2]: value.detach().cpu().clone()
        for key, value in state.items()
        if key.endswith(".B")
    }


def join_ffa_B_state(B_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert module-keyed B tensors into state-dict keys."""
    return {
        f"{name}.B": value.detach().cpu().clone()
        for name, value in sorted(B_state.items())
    }


def join_ffa_A_state(A_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert module-keyed frozen A tensors into state-dict keys."""
    return {
        f"{name}.A_frozen": value.detach().cpu().clone()
        for name, value in sorted(A_state.items())
    }


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
