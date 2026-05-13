import pytest

torch = pytest.importorskip("torch")

from fed_adapter.aggregation import (
    normalized_client_weights,
    stack_linear_lora,
    stack_nonlinear_lora,
    weighted_average,
    zero_pad_by_rank,
)


def test_normalized_client_weights():
    assert normalized_client_weights({0: 2, 1: 6}) == {0: 0.25, 1: 0.75}


def test_weighted_average():
    states = {
        0: {"w": torch.tensor([2.0, 4.0])},
        1: {"w": torch.tensor([10.0, 20.0])},
    }
    weights = {0: 0.25, 1: 0.75}

    result = weighted_average(states, weights)

    assert torch.allclose(result["w"], torch.tensor([8.0, 16.0]))


def test_zero_pad_by_rank_pads_rank_dimensions():
    states = {
        0: {
            "A": torch.ones(2, 3),
            "B": torch.ones(4, 2),
        },
        1: {
            "A": torch.full((4, 3), 2.0),
            "B": torch.full((4, 4), 2.0),
        },
    }
    ranks = {0: 2, 1: 4}

    padded = zero_pad_by_rank(states, ranks)

    assert padded[0]["A"].shape == (4, 3)
    assert padded[0]["B"].shape == (4, 4)
    assert torch.allclose(padded[0]["A"][0:2], torch.ones(2, 3))
    assert torch.allclose(padded[0]["A"][2:4], torch.zeros(2, 3))
    assert torch.allclose(padded[1]["A"], states[1]["A"])


def test_stack_linear_lora_weights_a_not_b():
    states = {
        0: {"A": torch.ones(2, 3), "B": torch.ones(4, 2)},
        1: {"A": torch.full((1, 3), 2.0), "B": torch.full((4, 1), 3.0)},
    }
    weights = {0: 0.25, 1: 0.75}
    ranks = {0: 2, 1: 1}

    result = stack_linear_lora(states, weights, ranks)

    assert result["A"].shape == (3, 3)
    assert result["B"].shape == (4, 3)
    assert torch.allclose(result["A"][0:2], torch.full((2, 3), 0.25))
    assert torch.allclose(result["A"][2:3], torch.full((1, 3), 1.5))
    assert torch.allclose(result["B"][:, 0:2], torch.ones(4, 2))
    assert torch.allclose(result["B"][:, 2:3], torch.full((4, 1), 3.0))


def test_stack_nonlinear_lora_weights_b_not_a():
    states = {
        0: {"A": torch.ones(2, 3), "B": torch.ones(4, 2)},
        1: {"A": torch.full((1, 3), 2.0), "B": torch.full((4, 1), 3.0)},
    }
    weights = {0: 0.25, 1: 0.75}
    ranks = {0: 2, 1: 1}

    result = stack_nonlinear_lora(states, weights, ranks)

    assert torch.allclose(result["A"][0:2], torch.ones(2, 3))
    assert torch.allclose(result["A"][2:3], torch.full((1, 3), 2.0))
    assert torch.allclose(result["B"][:, 0:2], torch.full((4, 2), 0.25))
    assert torch.allclose(result["B"][:, 2:3], torch.full((4, 1), 2.25))
