"""Project-wide configuration values for data pipelines."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 64
    image_height: int = 28
    image_width: int = 28
    channels: int = 1
    num_classes: int = 10
    validation_split: float = 0.1
    random_seed: int = 42
    shuffle_buffer: int = 10_000


DEFAULT_DATA_CONFIG = DataConfig()
