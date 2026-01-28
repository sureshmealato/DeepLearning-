"""Evaluation utilities for trained models."""

from __future__ import annotations

from typing import Dict, Tuple

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
        "confusion_matrix": confusion_matrix(labels, predictions),
        "classification_report": classification_report(labels, predictions, digits=4),
    }
