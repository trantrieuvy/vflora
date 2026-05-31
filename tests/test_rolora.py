from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
import torch.nn.functional as F

from fed_adapter.adapters.rolora import (
    NonlinearRoLoRALayer,
    inject_rolora_adapters,
    join_rolora_state,
    nonlinear_lora_delta,
    rolora_active_state_dict,
    split_rolora_factor_state,
    split_rolora_state,
)
from fed_adapter.distillation import distill_nonlinear_lora_modules


def test_rolora_layer_trains_only_active_factor():
    linear = nn.Linear(2, 1, bias=False)
    linear.weight.data.fill_(0.5)
    A = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    B = torch.tensor([[0.25, -0.75]])
    layer = NonlinearRoLoRALayer(linear, A, B, scaling=0.5, activation="gelu", train_factor="A")
    x = torch.tensor([[2.0, 3.0]])

    result = layer(x)
    expected = linear(x) + 0.5 * F.linear(F.gelu(F.linear(x, A)), B)

    assert torch.allclose(result, expected)
    assert layer.A.requires_grad
    assert not layer.B.requires_grad
    assert all(not parameter.requires_grad for parameter in layer.linear.parameters())


def test_rolora_state_helpers_roundtrip_active_factor_and_global_state():
    model = nn.Sequential(OrderedDict([("q_proj", nn.Linear(3, 2, bias=False))]))
    A = {"q_proj": torch.ones(2, 3)}
    B = {"q_proj": torch.zeros(2, 2)}
    injected, count = inject_rolora_adapters(
        model,
        ["q_proj"],
        A,
        B,
        scaling=1.0,
        dropout=0.0,
        train_factor="B",
    )

    state = rolora_active_state_dict(injected, "B")
    split = split_rolora_factor_state(state, "B")
    joined = join_rolora_state(A, split)
    roundtrip_A, roundtrip_B = split_rolora_state(joined)

    assert count == 1
    assert set(state) == {"q_proj.B"}
    assert set(split) == {"q_proj"}
    assert torch.equal(roundtrip_A["q_proj"], A["q_proj"])
    assert torch.equal(roundtrip_B["q_proj"], B["q_proj"])


def test_distillation_reduces_nonlinear_residual_mse():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    inputs = {"q_proj": torch.randn(16, 3, generator=generator)}
    teacher_A = {"q_proj": torch.randn(2, 3, generator=generator)}
    teacher_B = {"q_proj": torch.randn(4, 2, generator=generator)}
    init_A = {"q_proj": teacher_A["q_proj"] + 0.2 * torch.randn(2, 3, generator=generator)}
    init_B = {"q_proj": teacher_B["q_proj"] + 0.2 * torch.randn(4, 2, generator=generator)}

    with torch.no_grad():
        before = torch.mean(
            (
                nonlinear_lora_delta(inputs["q_proj"], init_A["q_proj"], init_B["q_proj"])
                - nonlinear_lora_delta(inputs["q_proj"], teacher_A["q_proj"], teacher_B["q_proj"])
            )
            ** 2
        )

    student_A, student_B, metrics = distill_nonlinear_lora_modules(
        inputs,
        teacher_A,
        teacher_B,
        init_A,
        init_B,
        steps=80,
        batch_size=8,
        lr=5e-2,
        seed=3,
    )

    with torch.no_grad():
        after = torch.mean(
            (
                nonlinear_lora_delta(inputs["q_proj"], student_A["q_proj"], student_B["q_proj"])
                - nonlinear_lora_delta(inputs["q_proj"], teacher_A["q_proj"], teacher_B["q_proj"])
            )
            ** 2
        )

    assert after < before
    assert metrics["mse_after"] < metrics["mse_before"]
