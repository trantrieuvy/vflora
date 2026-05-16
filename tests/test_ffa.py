from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
import torch.nn.functional as F

from fed_adapter.adapters.ffa import (
    FFALoRALayer,
    ffa_B_state_dict,
    init_frozen_A,
    init_zero_B,
    inject_ffa_adapters,
    join_ffa_A_state,
    join_ffa_B_state,
    split_ffa_B_state,
)


def test_ffa_layer_freezes_a_and_trains_b():
    linear = nn.Linear(2, 1, bias=False)
    linear.weight.data.fill_(0.5)
    A = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    B = torch.tensor([[0.25, -0.75]])
    layer = FFALoRALayer(linear, A, B, scaling=0.5, activation="gelu")
    x = torch.tensor([[2.0, 3.0]])

    result = layer(x)
    expected = linear(x) + 0.5 * F.linear(F.gelu(F.linear(x, A)), B)

    assert torch.allclose(result, expected)
    assert not layer.A_frozen.requires_grad
    assert layer.B.requires_grad
    assert all(not parameter.requires_grad for parameter in layer.linear.parameters())


def test_seeded_frozen_a_initialization_is_reproducible():
    model = nn.Sequential(OrderedDict([("q_proj", nn.Linear(3, 2, bias=False))]))

    first = init_frozen_A(model, ["q_proj"], rank=2, seed=7, init_std=0.02)
    second = init_frozen_A(model, ["q_proj"], rank=2, seed=7, init_std=0.02)
    different = init_frozen_A(model, ["q_proj"], rank=2, seed=8, init_std=0.02)

    assert torch.equal(first["q_proj"], second["q_proj"])
    assert not torch.equal(first["q_proj"], different["q_proj"])


def test_inject_ffa_adapters_slices_heterogeneous_rank_prefix():
    model = nn.Sequential(OrderedDict([("q_proj", nn.Linear(3, 2, bias=False))]))
    A = {"q_proj": torch.ones(4, 3)}
    B = {"q_proj": torch.ones(2, 4)}

    injected, count = inject_ffa_adapters(
        model,
        target_modules=["q_proj"],
        A_frozen=A,
        B_state=B,
        scaling=1.0,
        dropout=0.0,
        activation="gelu",
        client_rank=2,
    )

    assert count == 1
    assert injected.q_proj.A_frozen.shape == (2, 3)
    assert injected.q_proj.B.shape == (2, 2)


def test_ffa_state_helpers_roundtrip_module_keys():
    model = nn.Sequential(OrderedDict([("q_proj", nn.Linear(3, 2, bias=False))]))
    A = init_frozen_A(model, ["q_proj"], rank=2, seed=7)
    B = init_zero_B(model, ["q_proj"], rank=2)
    injected, _ = inject_ffa_adapters(model, ["q_proj"], A, B, scaling=1.0, dropout=0.0)

    state = ffa_B_state_dict(injected)
    split = split_ffa_B_state(state)

    assert set(state) == {"q_proj.B"}
    assert set(split) == {"q_proj"}
    assert set(join_ffa_B_state(split)) == {"q_proj.B"}
    assert set(join_ffa_A_state(A)) == {"q_proj.A_frozen"}
