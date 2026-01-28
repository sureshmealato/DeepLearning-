"""Dataset loading and preprocessing utilities."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np
import tensorflow as tf

from src.utils.config import DataConfig, DEFAULT_DATA_CONFIG


def _validate_split_ratio(validation_split: float) -> None:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1 (exclusive).")


def _normalize_images(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32") / 255.0
    return images


def _reshape_images(images: np.ndarray, config: DataConfig) -> np.ndarray:
    return images.reshape((-1, config.image_height, config.image_width, config.channels))


def _one_hot_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
    return tf.keras.utils.to_categorical(labels, num_classes)


def _train_validation_split(
    images: np.ndarray,
    labels: np.ndarray,
    validation_split: float,
    seed: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    _validate_split_ratio(validation_split)
    rng = np.random.default_rng(seed)
    indices = np.arange(images.shape[0])
    rng.shuffle(indices)
    split_index = int(images.shape[0] * (1 - validation_split))
    train_idx = indices[:split_index]
    val_idx = indices[split_index:]
    return (images[train_idx], labels[train_idx]), (images[val_idx], labels[val_idx])


def _validate_data(images: np.ndarray, labels: np.ndarray, config: DataConfig) -> None:
    if images.ndim != 4:
        raise ValueError(f"Expected images to be 4D, got {images.ndim}D.")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels to be 2D one-hot, got {labels.ndim}D.")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Images and labels must have the same number of samples.")
    if images.shape[1:] != (config.image_height, config.image_width, config.channels):
        raise ValueError("Images have incorrect shape for configured dimensions.")
    if labels.shape[1] != config.num_classes:
        raise ValueError("Labels have incorrect number of classes.")
    if np.isnan(images).any() or np.isnan(labels).any():
        raise ValueError("NaNs found in images or labels.")


def load_mnist(config: DataConfig = DEFAULT_DATA_CONFIG) -> dict:
    """Load and preprocess the MNIST dataset.

    Returns a dictionary with train, validation, and test splits as NumPy arrays.
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = _reshape_images(_normalize_images(train_images), config)
    test_images = _reshape_images(_normalize_images(test_images), config)

    train_labels = _one_hot_labels(train_labels, config.num_classes)
    test_labels = _one_hot_labels(test_labels, config.num_classes)

    (train_images, train_labels), (val_images, val_labels) = _train_validation_split(
        train_images,
        train_labels,
        config.validation_split,
        config.random_seed,
    )

    _validate_data(train_images, train_labels, config)
    _validate_data(val_images, val_labels, config)
    _validate_data(test_images, test_labels, config)

    return {
        "train": (train_images, train_labels),
        "validation": (val_images, val_labels),
        "test": (test_images, test_labels),
    }


def build_tf_datasets(config: DataConfig = DEFAULT_DATA_CONFIG) -> dict:
    """Create tf.data.Dataset objects for training, validation, and testing."""
    splits = load_mnist(config)

    def to_dataset(images: np.ndarray, labels: np.ndarray, shuffle: bool) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        if shuffle:
            dataset = dataset.shuffle(config.shuffle_buffer, seed=config.random_seed)
        return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return {
        "train": to_dataset(*splits["train"], shuffle=True),
        "validation": to_dataset(*splits["validation"], shuffle=False),
        "test": to_dataset(*splits["test"], shuffle=False),
    }


def with_overrides(**overrides: int | float) -> DataConfig:
    """Return a DataConfig with provided overrides."""
    return replace(DEFAULT_DATA_CONFIG, **overrides)
