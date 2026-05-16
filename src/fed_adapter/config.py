"""Configuration objects for federated adapter training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    tokenizer_name: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass(frozen=True)
class DataConfig:
    train_root: Path
    eval_path: Path | None = None
    prompt_template: str = "alpaca"
    max_length: int = 512


@dataclass(frozen=True)
class FederatedConfig:
    num_clients: int = 10
    client_fraction: float = 1.0
    rounds: int = 3
    seed: int = 42


@dataclass(frozen=True)
class AdapterConfig:
    method: str = "linear-cumulative-flora"
    backend: str = "direct"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    activation: str = "gelu"
    a_init_std: float = 0.02
    heterogeneous: bool = False
    local_ranks: tuple[int, ...] = field(default_factory=tuple)
    aggregation: str = "stack_linear"


@dataclass(frozen=True)
class LocalTrainingConfig:
    epochs: int = 1
    batch_size: int = 128
    micro_batch_size: int = 16
    learning_rate: float = 3e-4
    val_size: int = 0
    group_by_length: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    local_training: LocalTrainingConfig = field(default_factory=LocalTrainingConfig)
    output_dir: Path = Path("runs/default")
