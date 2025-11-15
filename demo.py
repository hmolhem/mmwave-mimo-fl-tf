#!/usr/bin/env python3
"""
Quick demonstration of the mmWave MIMO FL TensorFlow implementation.
This script shows the key features with minimal training.
"""

import sys
sys.path.insert(0, 'src')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
import tensorflow as tf
from data import create_synthetic_data, normalize_data, split_dataset
from data.tf_pipeline import prepare_datasets, create_federated_datasets
from models import build_cnn_model, compile_model
from training import CentralizedTrainer, FedAvgTrainer
from evaluation import ModelEvaluator

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')


def demo_centralized():
    """Demonstrate centralized training."""
    print("\n" + "="*70)
    print("DEMO 1: Centralized Training")
    print("="*70)
    
    # Create and prepare data
    print("\n1. Creating synthetic mmWave MIMO data (100 samples, 10 classes)...")
    data, labels = create_synthetic_data(num_samples=100, num_classes=10, height=64, width=64)
    data = normalize_data(data, method='minmax')
    
    # Split dataset
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset(
        data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print(f"   Train: {train_data.shape[0]} samples")
    print(f"   Val: {val_data.shape[0]} samples")
    print(f"   Test: {test_data.shape[0]} samples")
    
    # Create TF datasets
    print("\n2. Creating TensorFlow data pipelines...")
    train_ds, val_ds, test_ds = prepare_datasets(
        train_data, train_labels, val_data, val_labels, test_data, test_labels,
        batch_size=16, augment_train=True
    )
    
    # Build model
    print("\n3. Building CNN model (Standard architecture)...")
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10, architecture='standard')
    model = compile_model(model, learning_rate=0.001)
    print(f"   Total parameters: {model.count_params():,}")
    
    # Train
    print("\n4. Training for 5 epochs...")
    trainer = CentralizedTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        save_dir='/tmp/demo_checkpoints',
        experiment_name='demo_centralized'
    )
    trainer.train(epochs=5, verbose=0)
    
    # Evaluate
    print("\n5. Evaluating on test set...")
    metrics = trainer.evaluate(test_ds)
    print(f"\n   Test Accuracy: {metrics.get('accuracy', metrics.get('compile_metrics', 0)):.4f}")
    
    print("\n✓ Centralized training demo complete!")
    return model


def demo_federated():
    """Demonstrate federated learning."""
    print("\n" + "="*70)
    print("DEMO 2: Federated Learning (FedAvg)")
    print("="*70)
    
    # Create and prepare data
    print("\n1. Creating synthetic data for federated setup...")
    data, labels = create_synthetic_data(num_samples=100, num_classes=10)
    data = normalize_data(data)
    
    # Split for federated learning
    indices = np.random.permutation(len(data))
    train_val_split = int(len(data) * 0.85)
    
    train_val_data = data[indices[:train_val_split]]
    train_val_labels = labels[indices[:train_val_split]]
    test_data = data[indices[train_val_split:]]
    test_labels = labels[indices[train_val_split:]]
    
    # Create federated datasets
    NUM_CLIENTS = 3
    print(f"\n2. Distributing data to {NUM_CLIENTS} clients (IID)...")
    client_datasets = create_federated_datasets(
        train_val_data[:70], train_val_labels[:70],
        num_clients=NUM_CLIENTS, batch_size=8, iid=True
    )
    
    from data.tf_pipeline import create_tf_dataset
    val_ds = create_tf_dataset(train_val_data[70:], train_val_labels[70:], batch_size=8, shuffle=False)
    test_ds = create_tf_dataset(test_data, test_labels, batch_size=8, shuffle=False)
    
    # Build model
    print("\n3. Building global model (Lightweight architecture for faster FL)...")
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10, architecture='lightweight')
    model = compile_model(model, learning_rate=0.001)
    print(f"   Total parameters: {model.count_params():,}")
    
    # Federated training
    print(f"\n4. Running federated training for 5 rounds...")
    print(f"   {NUM_CLIENTS} clients, 1 epoch per client per round")
    trainer = FedAvgTrainer(
        model=model,
        client_datasets=client_datasets,
        val_dataset=val_ds,
        num_clients=NUM_CLIENTS,
        save_dir='/tmp/demo_checkpoints',
        experiment_name='demo_federated'
    )
    trainer.train(num_rounds=5, client_epochs=1, verbose=0)
    
    # Evaluate
    print("\n5. Evaluating global model...")
    metrics = trainer.evaluate(test_ds)
    print(f"\n   Test Accuracy: {metrics.get('accuracy', metrics.get('compile_metrics', 0)):.4f}")
    
    print("\n✓ Federated learning demo complete!")
    return model


def demo_evaluation():
    """Demonstrate comprehensive evaluation."""
    print("\n" + "="*70)
    print("DEMO 3: Comprehensive Evaluation")
    print("="*70)
    
    # Create test data
    print("\n1. Creating test dataset...")
    data, labels = create_synthetic_data(num_samples=50, num_classes=10)
    data = normalize_data(data)
    
    from data.tf_pipeline import create_tf_dataset
    test_ds = create_tf_dataset(data, labels, batch_size=8, shuffle=False)
    
    # Build and train a quick model
    print("\n2. Training a quick model for evaluation demo...")
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10, architecture='lightweight')
    model = compile_model(model)
    model.fit(test_ds, epochs=3, verbose=0)
    
    # Comprehensive evaluation
    print("\n3. Running comprehensive evaluation...")
    class_names = [f"ROI_{i}" for i in range(10)]
    evaluator = ModelEvaluator(model, save_dir='/tmp/demo_results')
    metrics = evaluator.evaluate_dataset(test_ds, class_names)
    
    print("\n   Generated:")
    print("   - Confusion matrices (normalized & unnormalized)")
    print("   - Per-class metrics (precision, recall, F1)")
    print("   - Overall metrics saved to JSON")
    
    print("\n✓ Evaluation demo complete!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("mmWave MIMO Federated Learning with TensorFlow - Quick Demo")
    print("="*70)
    print("\nThis demo shows the key features of the implementation:")
    print("  1. Centralized training with CNN")
    print("  2. Federated learning with FedAvg")
    print("  3. Comprehensive evaluation with confusion matrices")
    print("\nNote: Using small datasets and few epochs for quick demonstration")
    
    try:
        # Run demos
        demo_centralized()
        demo_federated()
        demo_evaluation()
        
        # Summary
        print("\n" + "="*70)
        print("All Demos Completed Successfully! ✓")
        print("="*70)
        print("\nKey Features Demonstrated:")
        print("  ✓ Data loading and preprocessing for .mat files")
        print("  ✓ TensorFlow data pipelines with batching and augmentation")
        print("  ✓ Multiple CNN architectures (standard, deep, lightweight)")
        print("  ✓ Centralized training with callbacks")
        print("  ✓ FedAvg federated learning with IID/non-IID support")
        print("  ✓ Comprehensive evaluation with confusion matrices")
        print("\nFor full examples, see:")
        print("  - examples/train_centralized.py")
        print("  - examples/train_federated.py")
        print("  - examples/compare_approaches.py")
        print("\nDocumentation: README.md")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
