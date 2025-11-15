#!/usr/bin/env python3
"""
Demo script showing both centralized and federated learning approaches.
"""

import tensorflow as tf
import numpy as np
import os
import sys

from data_loader import create_synthetic_data, split_data
from model import create_cnn_model, create_lightweight_cnn, compile_model
from data_pipeline import prepare_datasets_for_training
from train_centralized import train_centralized
from train_federated import train_federated
from evaluation import evaluate_model, print_evaluation_summary, compare_models


def run_demo(num_samples: int = 1000,
            num_classes: int = 10,
            image_size: int = 128,
            centralized_epochs: int = 20,
            federated_rounds: int = 20,
            seed: int = 42):
    """
    Run a demonstration of both centralized and federated learning.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
        image_size: Size of range-azimuth images
        centralized_epochs: Epochs for centralized training
        federated_rounds: Rounds for federated training
        seed: Random seed
    """
    print("\n" + "="*70)
    print(" mmWave MIMO CNN Classification - Centralized vs Federated Learning")
    print("="*70)
    
    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Generate synthetic data
    print(f"\nGenerating {num_samples} synthetic range-azimuth samples...")
    X, y = create_synthetic_data(
        num_samples=num_samples,
        num_classes=num_classes,
        image_shape=(image_size, image_size)
    )
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Classes: {np.unique(y)}")
    
    # Split data
    print("\nSplitting data (70% train, 15% val, 15% test)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=seed
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    input_shape = (image_size, image_size, 1)
    
    # Create directories for saving results
    os.makedirs('demo_results', exist_ok=True)
    
    # ========================================================================
    # CENTRALIZED TRAINING
    # ========================================================================
    print("\n" + "="*70)
    print(" PART 1: CENTRALIZED TRAINING")
    print("="*70)
    
    centralized_model, centralized_history = train_centralized(
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_classes=num_classes,
        input_shape=input_shape,
        epochs=centralized_epochs,
        batch_size=32,
        learning_rate=0.001,
        model_type='lightweight',
        save_path='demo_results/centralized',
        verbose=1
    )
    
    # Evaluate centralized model
    centralized_results = evaluate_model(
        centralized_model, X_test, y_test, verbose=1
    )
    
    print("\nCentralized Training Summary:")
    print_evaluation_summary(centralized_results, num_classes)
    
    # ========================================================================
    # FEDERATED TRAINING
    # ========================================================================
    print("\n" + "="*70)
    print(" PART 2: FEDERATED LEARNING (FedAvg)")
    print("="*70)
    
    fedavg_trainer, federated_history = train_federated(
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_classes=num_classes,
        input_shape=input_shape,
        num_rounds=federated_rounds,
        num_clients=10,
        clients_per_round=5,
        local_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        iid=True,
        save_path='demo_results/federated',
        verbose=1
    )
    
    # Evaluate federated model
    federated_model = fedavg_trainer.get_global_model()
    federated_results = evaluate_model(
        federated_model, X_test, y_test, verbose=1
    )
    
    print("\nFederated Training Summary:")
    print_evaluation_summary(federated_results, num_classes)
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print(" COMPARISON: CENTRALIZED vs FEDERATED")
    print("="*70)
    
    # Compare results
    comparison_results = [centralized_results, federated_results]
    model_names = ['Centralized', 'Federated (FedAvg)']
    
    print(f"\n{'Model':<25} {'Test Accuracy':<15} {'Test Loss':<15}")
    print("-" * 55)
    print(f"{'Centralized':<25} {centralized_results['accuracy']:<15.4f} "
          f"{centralized_model.evaluate(X_test, y_test, verbose=0)[0]:<15.4f}")
    print(f"{'Federated (FedAvg)':<25} {federated_results['accuracy']:<15.4f} "
          f"{federated_model.evaluate(X_test, y_test, verbose=0)[0]:<15.4f}")
    print("-" * 55)
    
    # Plot comparison
    compare_models(
        comparison_results,
        model_names,
        save_path='demo_results/model_comparison.png'
    )
    
    print("\n" + "="*70)
    print(" DEMO COMPLETED!")
    print("="*70)
    print("\nResults saved to 'demo_results/' directory:")
    print("  - Centralized model and metrics: demo_results/centralized/")
    print("  - Federated model and metrics: demo_results/federated/")
    print("  - Model comparison plot: demo_results/model_comparison.png")
    print("="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo: Centralized vs Federated Learning')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of synthetic samples (default: 1000)')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes (default: 10)')
    parser.add_argument('--image_size', type=int, default=128,
                       help='Image size (default: 128)')
    parser.add_argument('--centralized_epochs', type=int, default=20,
                       help='Epochs for centralized training (default: 20)')
    parser.add_argument('--federated_rounds', type=int, default=20,
                       help='Rounds for federated training (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    run_demo(
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        image_size=args.image_size,
        centralized_epochs=args.centralized_epochs,
        federated_rounds=args.federated_rounds,
        seed=args.seed
    )
