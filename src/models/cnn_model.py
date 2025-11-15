"""
CNN model for mmWave MIMO radar range-azimuth ROI classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional


def build_cnn_model(input_shape: Tuple[int, int, int] = (64, 64, 1),
                   num_classes: int = 10,
                   architecture: str = 'standard') -> keras.Model:
    """
    Build a CNN model for range-azimuth ROI classification.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        architecture: Model architecture ('standard', 'deep', or 'lightweight')
        
    Returns:
        Keras Model
    """
    if architecture == 'standard':
        return build_standard_cnn(input_shape, num_classes)
    elif architecture == 'deep':
        return build_deep_cnn(input_shape, num_classes)
    elif architecture == 'lightweight':
        return build_lightweight_cnn(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def build_standard_cnn(input_shape: Tuple[int, int, int],
                      num_classes: int) -> keras.Model:
    """
    Build a standard CNN architecture.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Keras Model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='standard_cnn')
    
    return model


def build_deep_cnn(input_shape: Tuple[int, int, int],
                  num_classes: int) -> keras.Model:
    """
    Build a deeper CNN architecture for better feature extraction.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Keras Model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='deep_cnn')
    
    return model


def build_lightweight_cnn(input_shape: Tuple[int, int, int],
                         num_classes: int) -> keras.Model:
    """
    Build a lightweight CNN architecture for faster training (suitable for FL).
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Keras Model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First block
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Third block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Fully connected layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='lightweight_cnn')
    
    return model


def compile_model(model: keras.Model,
                 learning_rate: float = 0.001,
                 loss: str = 'sparse_categorical_crossentropy',
                 metrics: list = None) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        loss: Loss function
        metrics: List of metrics to track
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy']
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """
    Get model summary as a string.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary string
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()
