from fed_adapter.selection import select_clients


def test_select_clients_is_deterministic_and_sorted():
    selected = select_clients(num_clients=10, fraction=0.3, seed=7)

    assert selected == sorted(selected)
    assert selected == select_clients(num_clients=10, fraction=0.3, seed=7)
    assert len(selected) == 3


def test_select_clients_selects_at_least_one():
    assert len(select_clients(num_clients=10, fraction=0.01, seed=1)) == 1

