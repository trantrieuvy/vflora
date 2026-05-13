"""Tensor aggregation utilities for federated adapter experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

TensorState = Mapping[str, torch.Tensor]


def normalized_client_weights(client_sizes: Mapping[int, int]) -> dict[int, float]:
    """Convert client dataset sizes into FedAvg weights."""
    if not client_sizes:
        raise ValueError("client_sizes cannot be empty")

    total = sum(client_sizes.values())
    if total <= 0:
        raise ValueError("sum of client sizes must be positive")

    return {client_id: size / total for client_id, size in client_sizes.items()}


def weighted_average(states: Mapping[int, TensorState], weights: Mapping[int, float]) -> dict[str, torch.Tensor]:
    """Compute a key-wise weighted average of compatible tensor state dicts."""
    _validate_state_inputs(states, weights)
    result: dict[str, torch.Tensor] = {}

    for client_id in sorted(states):
        state = states[client_id]
        weight = weights[client_id]
        if not result:
            result = {key: tensor.detach().clone() * weight for key, tensor in state.items()}
            continue
        for key, tensor in state.items():
            result[key] = result[key] + tensor.detach() * weight

    return result


def zero_pad_by_rank(
    states: Mapping[int, TensorState],
    ranks: Mapping[int, int],
) -> dict[int, dict[str, torch.Tensor]]:
    """Pad every rank-shaped tensor dimension to the maximum client rank."""
    if not states:
        raise ValueError("states cannot be empty")
    max_rank = max(ranks[client_id] for client_id in states)

    padded: dict[int, dict[str, torch.Tensor]] = {}
    for client_id, state in states.items():
        rank = ranks[client_id]
        padded[client_id] = {
            key: _pad_rank_dimensions(tensor.detach(), rank, max_rank)
            for key, tensor in state.items()
        }
    return padded


def stack_linear_lora(
    states: Mapping[int, TensorState],
    weights: Mapping[int, float],
    ranks: Mapping[int, int],
) -> dict[str, torch.Tensor]:
    """Stack linear LoRA adapters.

    For linear LoRA, weighting the A matrix preserves the weighted sum:
    B @ ((p * A) @ x) == p * (B @ A @ x).
    """
    return _stack_lora(states, weights, ranks, weight_side="A")


def stack_nonlinear_lora(
    states: Mapping[int, TensorState],
    weights: Mapping[int, float],
    ranks: Mapping[int, int],
) -> dict[str, torch.Tensor]:
    """Stack nonlinear LoRA adapters.

    For B * sigma(A * x), weights must be applied to B, not A, because
    scaling inside the activation changes the function.
    """
    return _stack_lora(states, weights, ranks, weight_side="B")


def _stack_lora(
    states: Mapping[int, TensorState],
    weights: Mapping[int, float],
    ranks: Mapping[int, int],
    weight_side: str,
) -> dict[str, torch.Tensor]:
    _validate_state_inputs(states, weights)
    if weight_side not in {"A", "B"}:
        raise ValueError("weight_side must be 'A' or 'B'")

    result: dict[str, torch.Tensor] = {}
    for client_id in sorted(states):
        rank = ranks[client_id]
        weight = weights[client_id]
        for key, tensor in states[client_id].items():
            tensor = tensor.detach()
            role = infer_lora_role(tensor, rank)
            part = tensor * weight if role == weight_side else tensor.clone()

            if key not in result:
                result[key] = part.clone()
            elif role == "A":
                result[key] = torch.cat([result[key], part], dim=0)
            elif role == "B":
                result[key] = torch.cat([result[key], part], dim=1)
            else:
                result[key] = result[key] + tensor * weight

    return result


def infer_lora_role(tensor: torch.Tensor, rank: int) -> str | None:
    """Infer whether a tensor is an A or B LoRA matrix from its rank dimension."""
    if tensor.ndim >= 1 and tensor.shape[0] == rank:
        return "A"
    if tensor.ndim >= 2 and tensor.shape[1] == rank:
        return "B"
    return None


def _pad_rank_dimensions(tensor: torch.Tensor, rank: int, target_rank: int) -> torch.Tensor:
    if rank == target_rank:
        return tensor.clone()

    padded = tensor.clone()
    for dim, size in enumerate(list(padded.shape)):
        if size != rank:
            continue
        pad_shape = list(padded.shape)
        pad_shape[dim] = target_rank - rank
        zeros = torch.zeros(pad_shape, dtype=padded.dtype, device=padded.device)
        padded = torch.cat([padded, zeros], dim=dim)
    return padded


def _validate_state_inputs(states: Mapping[int, TensorState], weights: Mapping[int, float]) -> None:
    if not states:
        raise ValueError("states cannot be empty")
    missing = set(states) - set(weights)
    if missing:
        raise ValueError(f"missing weights for client IDs: {sorted(missing)}")

    key_sets = {client_id: set(state) for client_id, state in states.items()}
    first_client = next(iter(key_sets))
    expected = key_sets[first_client]
    for client_id, keys in key_sets.items():
        if keys != expected:
            raise ValueError(
                f"state keys for client {client_id} do not match client {first_client}"
            )

