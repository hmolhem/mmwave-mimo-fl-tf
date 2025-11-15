"""
Centralized training script for mmWave MIMO CNN classification.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import argparse
from datetime import datetime

from data_loader import create_synthetic_data, split_data
from model import create_cnn_model, create_lightweight_cnn, compile_model
from data_pipeline import prepare_datasets_for_training
from evaluation import evaluate_model, plot_confusion_matrix, plot_training_history


def train_centralized(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     num_classes: int = 10,
                     input_shape: tuple = (128, 128, 1),
                     epochs: int = 50,
                     batch_size: int = 32,
                     learning_rate: float = 0.001,
                     model_type: str = 'standard',
                     save_path: str = 'checkpoints',
                     verbose: int = 1) -> tuple:
    """
    Train a model using centralized learning.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        num_classes: Number of classes
        input_shape: Input shape for the model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        model_type: Type of model ('standard' or 'lightweight')
        save_path: Path to save checkpoints
        verbose: Verbosity level
        
    Returns:
        Tuple of (model, history)
    """
    print("\n" + "="*60)
    print("CENTRALIZED TRAINING")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Num classes: {num_classes}")
    print(f"Input shape: {input_shape}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*60 + "\n")
    
    # Create model
    if model_type == 'lightweight':
        model = create_lightweight_cnn(input_shape=input_shape, num_classes=num_classes)
    else:
        model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)
    
    # Compile model
    model = compile_model(model, learning_rate=learning_rate)
    
    if verbose:
        model.summary()
    
    # Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets_for_training(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=batch_size, augment_train=True
    )
    
    # Create checkpoint directory
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(save_path, f'centralized_{timestamp}')
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_path, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        ),
        keras.callbacks.CSVLogger(
            os.path.join(checkpoint_path, 'training_log.csv')
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Generate predictions and confusion matrix
    y_pred = model.predict(test_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Save results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'training_history': history.history,
        'model_path': checkpoint_path
    }
    
    # Plot and save confusion matrix
    cm_path = os.path.join(checkpoint_path, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred_classes, num_classes=num_classes, 
                         save_path=cm_path)
    
    # Plot and save training history
    history_path = os.path.join(checkpoint_path, 'training_history.png')
    plot_training_history(history.history, save_path=history_path)
    
    print(f"\nModel and results saved to: {checkpoint_path}")
    
    return model, history


def main():
    """Main function for centralized training."""
    parser = argparse.ArgumentParser(description='Centralized training for mmWave MIMO CNN')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing .mat files (if None, uses synthetic data)')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--image_size', type=int, default=128,
                       help='Size of range-azimuth images')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'lightweight'],
                       help='Model type')
    parser.add_argument('--save_path', type=str, default='checkpoints',
                       help='Path to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Load or generate data
    if args.data_dir is not None:
        # TODO: Implement loading from .mat files
        print(f"Loading data from {args.data_dir}...")
        raise NotImplementedError("Loading from .mat files not yet implemented in main()")
    else:
        print(f"Generating {args.num_samples} synthetic samples...")
        X, y = create_synthetic_data(
            num_samples=args.num_samples,
            num_classes=args.num_classes,
            image_shape=(args.image_size, args.image_size)
        )
    
    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=args.seed
    )
    
    input_shape = (args.image_size, args.image_size, 1)
    
    # Train model
    model, history = train_centralized(
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_classes=args.num_classes,
        input_shape=input_shape,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        save_path=args.save_path
    )
    
    print("\nCentralized training completed!")


if __name__ == '__main__':
    main()
