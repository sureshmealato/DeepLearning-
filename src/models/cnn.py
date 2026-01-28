"""Baseline CNN model for MNIST."""

from __future__ import annotations

import tensorflow as tf

from src.utils.config import DataConfig, DEFAULT_DATA_CONFIG


def build_cnn_model(config: DataConfig = DEFAULT_DATA_CONFIG) -> tf.keras.Model:
    """Build and compile a baseline CNN model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(config.image_height, config.image_width, config.channels)
            ),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(config.num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
