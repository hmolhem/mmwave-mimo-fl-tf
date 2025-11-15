"""
Example script showing basic usage of the mmWave MIMO classification system.
"""

import numpy as np
import tensorflow as tf
from data_loader import create_synthetic_data, split_data, load_mat_file
from model import create_cnn_model, compile_model
from data_pipeline import prepare_datasets_for_training
from evaluation import evaluate_model, print_evaluation_summary


def basic_example():
    """
    Basic example of loading data and training a model.
    """
    print("="*60)
    print("Basic Example: mmWave MIMO CNN Classification")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Generate synthetic data (in practice, load from .mat files)
    print("\n1. Loading data...")
    X, y = create_synthetic_data(
        num_samples=500,
        num_classes=10,
        image_shape=(128, 128)
    )
    print(f"   Data shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Split into train/val/test
    print("\n2. Splitting data...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create and compile model
    print("\n3. Creating model...")
    model = create_cnn_model(input_shape=(128, 128, 1), num_classes=10)
    model = compile_model(model, learning_rate=0.001)
    print(f"   Model created with {model.count_params():,} parameters")
    
    # Prepare datasets
    print("\n4. Preparing data pipelines...")
    train_ds, val_ds, test_ds = prepare_datasets_for_training(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=32
    )
    
    # Train model
    print("\n5. Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=1
    )
    
    # Evaluate model
    print("\n6. Evaluating model...")
    results = evaluate_model(model, X_test, y_test, verbose=1)
    print_evaluation_summary(results, num_classes=10)
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
    
    return model, results


def load_real_data_example(data_dir: str):
    """
    Example of loading real .mat files.
    
    Args:
        data_dir: Directory containing .mat files
    """
    from data_loader import load_dataset_from_directory
    
    print("="*60)
    print("Loading Real Data Example")
    print("="*60)
    
    # Load data from .mat files
    print(f"\nLoading data from: {data_dir}")
    X, y = load_dataset_from_directory(
        data_dir=data_dir,
        pattern='*.mat',
        num_classes=10,
        normalize=True
    )
    
    print(f"Loaded data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    
    return X, y


def custom_model_example():
    """
    Example of creating a custom model architecture.
    """
    from tensorflow.keras import layers, models
    
    print("="*60)
    print("Custom Model Example")
    print("="*60)
    
    # Create custom model
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),
        layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ], name='custom_mmwave_cnn')
    
    # Compile
    model = compile_model(model, learning_rate=0.001)
    
    print("\nCustom model created:")
    model.summary()
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Example usage of mmWave MIMO classification')
    parser.add_argument('--example', type=str, default='basic',
                       choices=['basic', 'custom_model'],
                       help='Which example to run')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing .mat files (for real data example)')
    
    args = parser.parse_args()
    
    if args.example == 'basic':
        basic_example()
    elif args.example == 'custom_model':
        custom_model_example()
    elif args.example == 'load_real' and args.data_dir:
        load_real_data_example(args.data_dir)
    else:
        print("Running basic example by default...")
        basic_example()
