"""Model architectures."""

from src.models.ann import build_ann_model
from src.models.cnn import build_cnn_model

MODEL_REGISTRY = {
    "ann": build_ann_model,
    "cnn": build_cnn_model,
}


def build_model(name: str):
    """Build a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    return MODEL_REGISTRY[name]()


__all__ = ["MODEL_REGISTRY", "build_ann_model", "build_cnn_model", "build_model"]
