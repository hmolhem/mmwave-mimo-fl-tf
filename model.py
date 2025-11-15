"""
CNN model for 10-class ROI classification on range-azimuth data.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional


def create_cnn_model(input_shape: Tuple[int, int, int] = (128, 128, 1),
                     num_classes: int = 10,
                     model_name: str = 'mmwave_cnn') -> keras.Model:
    """
    Create a CNN model for range-azimuth ROI classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        model_name: Name for the model
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25, name='dropout3'),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4'),
        layers.BatchNormalization(name='bn4'),
        layers.MaxPooling2D((2, 2), name='pool4'),
        layers.Dropout(0.25, name='dropout4'),
        
        # Flatten and dense layers
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='fc1'),
        layers.BatchNormalization(name='bn5'),
        layers.Dropout(0.5, name='dropout5'),
        layers.Dense(256, activation='relu', name='fc2'),
        layers.Dropout(0.5, name='dropout6'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name=model_name)
    
    return model


def create_lightweight_cnn(input_shape: Tuple[int, int, int] = (128, 128, 1),
                           num_classes: int = 10) -> keras.Model:
    """
    Create a lightweight CNN model suitable for federated learning.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Keras model
    """
    model = models.Sequential([
        # First block
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Third block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='lightweight_cnn')
    
    return model


def compile_model(model: keras.Model,
                 learning_rate: float = 0.001,
                 optimizer_name: str = 'adam') -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop')
        
    Returns:
        Compiled model
    """
    # Choose optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
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


def save_model(model: keras.Model, filepath: str):
    """
    Save model to file.
    
    Args:
        model: Keras model to save
        filepath: Path to save the model
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> keras.Model:
    """
    Load model from file.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    model = keras.models.load_model(filepath)
    print(f"Model loaded from {filepath}")
    return model
