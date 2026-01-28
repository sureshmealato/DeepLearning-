"""Training utilities and callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import tensorflow as tf

from src.utils.config import DataConfig, DEFAULT_DATA_CONFIG


def train_model(
    model: tf.keras.Model,
    datasets: Dict[str, tf.data.Dataset],
    config: DataConfig = DEFAULT_DATA_CONFIG,
    model_dir: Path | str = "saved_models",
    monitor: str = "val_accuracy",
    log_dir: Path | str = "logs",
    experiment_name: str = "ann",
) -> tuple[tf.keras.callbacks.History, Path]:
    """Train a model and return the history and best model path."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "ann_best.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor=monitor,
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]

    history = model.fit(
        datasets["train"],
        validation_data=datasets["validation"],
        epochs=20,
        callbacks=callbacks,
    )

    return history, best_model_path
