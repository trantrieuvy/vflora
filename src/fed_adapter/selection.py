"""Client participation policies for federated rounds."""

from __future__ import annotations

import random


def select_clients(
    num_clients: int,
    fraction: float,
    seed: int | None = None,
) -> list[int]:
    """Select a deterministic random subset of clients.

    At least one client is selected whenever ``num_clients`` is positive.
    The returned client IDs are sorted so downstream aggregation order is stable.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1]")

    count = max(1, int(num_clients * fraction))
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_clients), count))

