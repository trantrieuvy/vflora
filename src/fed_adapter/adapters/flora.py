"""Normal linear FLoRA helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from fed_adapter.adapters.residual import validate_adapter_pair


def merge_linear_lora_into_model(
    model: nn.Module,
    A_state: Mapping[str, torch.Tensor],
    B_state: Mapping[str, torch.Tensor],
    scaling: float,
) -> None:
    """Merge a stacked linear LoRA residual into target backbone weights."""
    validate_adapter_pair(A_state, B_state, "round")
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name in sorted(A_state):
            module = modules.get(name)
            if module is None:
                raise KeyError(f"Cannot merge residual: target module not found: {name}")
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Cannot merge residual into non-linear module {name}: {type(module).__name__}")
            A = A_state[name].to(device=module.weight.device, dtype=torch.float32)
            B = B_state[name].to(device=module.weight.device, dtype=torch.float32)
            delta = torch.matmul(B, A) * float(scaling)
            if delta.shape != module.weight.shape:
                raise ValueError(
                    f"Residual delta shape mismatch for {name}: "
                    f"delta{tuple(delta.shape)} vs weight{tuple(module.weight.shape)}"
                )
            module.weight.data.add_(delta.to(dtype=module.weight.dtype))


def join_flora_adapter_state(
    A_state: Mapping[str, torch.Tensor],
    B_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Write GLUE FLoRA adapter checkpoints with FederatedLLM-compatible keys."""
    validate_adapter_pair(A_state, B_state, "adapter state")
    state: dict[str, torch.Tensor] = {}
    for name in sorted(A_state):
        state[f"{name}.A"] = A_state[name].detach().cpu().clone()
        state[f"{name}.B"] = B_state[name].detach().cpu().clone()
    return state
