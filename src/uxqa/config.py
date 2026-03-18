from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    image_size: int = 224
    in_channels: int = 3
    patch_size: int = 4
    visual_embed_dim: int = 128
    transformer_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 512
    dropout: float = 0.1
    max_ui_elements: int = 16
    gnn_layers: int = 2
    output_dim: int = 128


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 5


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig
    train: TrainConfig


def _apply_overrides(obj: Any, values: dict[str, Any]) -> Any:
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> AppConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    model_cfg = _apply_overrides(ModelConfig(), raw.get("model", {}))
    train_cfg = _apply_overrides(TrainConfig(), raw.get("train", {}))
    return AppConfig(model=model_cfg, train=train_cfg)
