"""Federated client abstraction for the rewritten training loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fed_adapter.adapters import AdapterBackend


@dataclass
class ClientRoundResult:
    client_id: int
    dataset_size: int
    checkpoint_path: Path


class FederatedClient:
    """Small adapter-aware client wrapper.

    This class is intentionally framework-light. The heavy objects are injected
    by the training entrypoint, which keeps this layer testable and keeps PEFT or
    LayerCraft details behind ``AdapterBackend``.
    """

    def __init__(
        self,
        client_id: int,
        model: Any,
        backend: AdapterBackend,
        data_path: Path,
        output_dir: Path,
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.backend = backend
        self.data_path = data_path
        self.output_dir = output_dir

    @property
    def local_training_path(self) -> Path:
        return self.data_path / f"local_training_{self.client_id}.json"

    def train_one_round(
        self,
        round_id: int,
        train_fn: Callable[[Any, Path], int],
        save_fn: Callable[[Path, Any], None],
    ) -> ClientRoundResult:
        """Train locally, save adapter weights, and restore the starting state."""
        initial_weights = {
            key: value.detach().clone() if hasattr(value, "detach") else value
            for key, value in self.backend.state_dict(self.model).items()
        }

        dataset_size = train_fn(self.model, self.local_training_path)
        checkpoint_dir = self.output_dir / str(round_id) / f"local_output_{self.client_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "adapter_model.bin"
        save_fn(checkpoint_path, self.backend.state_dict(self.model))

        self.backend.load_state_dict(self.model, initial_weights)
        return ClientRoundResult(self.client_id, dataset_size, checkpoint_path)

