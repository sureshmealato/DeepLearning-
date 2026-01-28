"""Evaluation utilities for trained models."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def _collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict(dataset, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.concatenate([np.argmax(batch_labels, axis=1) for _, batch_labels in dataset])
    return predictions, labels


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Dict[str, object]:
    """Evaluate a model and return metrics, confusion matrix, and report."""
    loss, accuracy = model.evaluate(dataset, verbose=0)
    predictions, labels = _collect_predictions(model, dataset)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(labels, predictions, digits=4, output_dict=True),
    }


def evaluate_models(
    models: Dict[str, tf.keras.Model],
    dataset: tf.data.Dataset,
) -> Dict[str, Dict[str, object]]:
    """Evaluate multiple models and return a mapping of name to metrics."""
    return {name: evaluate_model(model, dataset) for name, model in models.items()}


def save_evaluation(results: Dict[str, object], output_path: Path | str) -> Path:
    """Save evaluation results to disk as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return output_path
