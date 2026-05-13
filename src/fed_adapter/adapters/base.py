"""Common interface for adapter implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class AdapterBackend(ABC):
    """Minimal surface the federated loop needs from an adapter library."""

    name: str

    @abstractmethod
    def inject(self, model: Any, config: Any) -> Any:
        """Return ``model`` with trainable adapters attached."""

    @abstractmethod
    def state_dict(self, model: Any) -> Mapping[str, Any]:
        """Return only adapter weights from ``model``."""

    @abstractmethod
    def load_state_dict(self, model: Any, weights: Mapping[str, Any]) -> None:
        """Load adapter weights into ``model`` in-place."""

    def merge_if_supported(self, model: Any) -> Any:
        """Merge adapters into base weights when the backend supports it."""
        return model

