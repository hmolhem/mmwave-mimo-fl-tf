"""
CNN model architectures for mmWave MIMO human-robot position classification.

Models:
- build_baseline_cnn: compact 2-conv CNN
- build_improved_cnn: deeper CNN with batch norm
"""

from __future__ import annotations

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_baseline_cnn(input_shape=(256, 63, 1), num_classes=10, dropout_rate=0.3) -> keras.Model:
    """
    Baseline compact CNN for 10-class ROI classification.
    
    Architecture:
    - Conv Block 1: Conv2D(16, 3x3) + ReLU + MaxPool(2x2)
    - Conv Block 2: Conv2D(32, 3x3) + ReLU + MaxPool(2x2)
    - Flatten
    - Dense(128, ReLU) + Dropout
    - Dense(num_classes, Softmax)
    """
    inputs = keras.Input(shape=input_shape, name="range_azimuth_input")
    
    # Conv block 1
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    
    # Conv block 2
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    # Dense head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    return model


def build_improved_cnn(input_shape=(256, 63, 1), num_classes=10, dropout_rate=0.3) -> keras.Model:
    """
    Improved CNN with an extra conv block and batch normalization.
    
    Architecture:
    - Conv Block 1: Conv2D(16, 3x3) + BN + ReLU + MaxPool(2x2)
    - Conv Block 2: Conv2D(32, 3x3) + BN + ReLU + MaxPool(2x2)
    - Conv Block 3: Conv2D(64, 3x3) + BN + ReLU + MaxPool(2x2)
    - Flatten
    - Dense(256, ReLU) + Dropout
    - Dense(num_classes, Softmax)
    """
    inputs = keras.Input(shape=input_shape, name="range_azimuth_input")
    
    # Conv block 1
    x = layers.Conv2D(16, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    
    # Conv block 2
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    # Conv block 3
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    
    # Dense head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="improved_cnn")
    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3, weight_decay: float = 0.0):
    """
    Compile a CNN model with Adam optimizer and categorical crossentropy.
    """
    if weight_decay > 0:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_model_callbacks(checkpoint_dir: str, model_name: str, patience: int = 10):
    """
    Standard training callbacks:
    - ModelCheckpoint (save best val_loss)
    - EarlyStopping
    - ReduceLROnPlateau
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.keras")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, patience // 3),
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    return callbacks
