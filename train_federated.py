"""
Federated training script for mmWave MIMO CNN classification.
"""

import tensorflow as tf
import numpy as np
import os
import argparse
from datetime import datetime

from data_loader import create_synthetic_data, split_data
from model import create_lightweight_cnn
from data_pipeline import create_client_data_from_splits
from federated_learning import FederatedAveraging
from evaluation import evaluate_model, plot_confusion_matrix, plot_training_history


def train_federated(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   num_classes: int = 10,
                   input_shape: tuple = (128, 128, 1),
                   num_rounds: int = 50,
                   num_clients: int = 10,
                   clients_per_round: int = 5,
                   local_epochs: int = 5,
                   batch_size: int = 32,
                   learning_rate: float = 0.001,
                   iid: bool = True,
                   save_path: str = 'checkpoints',
                   verbose: int = 1) -> tuple:
    """
    Train a model using federated learning with FedAvg.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        num_classes: Number of classes
        input_shape: Input shape for the model
        num_rounds: Number of federated rounds
        num_clients: Total number of clients
        clients_per_round: Number of clients selected per round
        local_epochs: Number of local epochs per client
        batch_size: Batch size
        learning_rate: Learning rate
        iid: Whether to use IID data distribution
        save_path: Path to save checkpoints
        verbose: Verbosity level
        
    Returns:
        Tuple of (fedavg_trainer, history)
    """
    print("\n" + "="*60)
    print("FEDERATED LEARNING SETUP")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Num classes: {num_classes}")
    print(f"Input shape: {input_shape}")
    print(f"Data distribution: {'IID' if iid else 'Non-IID'}")
    print("="*60 + "\n")
    
    # Split training data among clients
    client_X, client_y = create_client_data_from_splits(
        X_train, y_train,
        num_clients=num_clients,
        shuffle=True,
        random_seed=42
    )
    
    # Create list of client datasets
    client_datasets = list(zip(client_X, client_y))
    
    print(f"Data distributed to {len(client_datasets)} clients")
    for i, (X_c, y_c) in enumerate(client_datasets):
        print(f"  Client {i}: {len(X_c)} samples, "
              f"classes: {np.unique(y_c).tolist()}")
    
    # Create FedAvg trainer
    fedavg = FederatedAveraging(
        model_fn=create_lightweight_cnn,
        num_classes=num_classes,
        input_shape=input_shape,
        learning_rate=learning_rate,
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        local_epochs=local_epochs,
        batch_size=batch_size
    )
    
    if verbose:
        print("\nGlobal Model Architecture:")
        fedavg.global_model.summary()
    
    # Create checkpoint directory
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(save_path, f'federated_{timestamp}')
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Train using FedAvg
    history = fedavg.train(
        client_datasets=client_datasets,
        num_rounds=num_rounds,
        X_val=X_val,
        y_val=y_val,
        verbose=verbose
    )
    
    # Get global model
    global_model = fedavg.get_global_model()
    
    # Evaluate on test set
    print("\nEvaluating global model on test set...")
    test_loss, test_acc = global_model.evaluate(X_test, y_test, 
                                                batch_size=batch_size, 
                                                verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Generate predictions and confusion matrix
    y_pred = global_model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Save global model
    model_path = os.path.join(checkpoint_path, 'global_model.h5')
    fedavg.save_global_model(model_path)
    
    # Plot and save confusion matrix
    cm_path = os.path.join(checkpoint_path, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred_classes, num_classes=num_classes,
                         save_path=cm_path)
    
    # Plot and save training history
    history_path = os.path.join(checkpoint_path, 'training_history.png')
    plot_training_history(history, metrics=['accuracy', 'loss'],
                         save_path=history_path)
    
    # Save training history
    import json
    history_json_path = os.path.join(checkpoint_path, 'history.json')
    with open(history_json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    
    print(f"\nModel and results saved to: {checkpoint_path}")
    
    return fedavg, history


def main():
    """Main function for federated training."""
    parser = argparse.ArgumentParser(description='Federated training for mmWave MIMO CNN')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing .mat files (if None, uses synthetic data)')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of synthetic samples to generate')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--image_size', type=int, default=128,
                       help='Size of range-azimuth images')
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of federated rounds')
    parser.add_argument('--num_clients', type=int, default=10,
                       help='Total number of clients')
    parser.add_argument('--clients_per_round', type=int, default=5,
                       help='Number of clients selected per round')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Number of local epochs per client')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--iid', action='store_true',
                       help='Use IID data distribution (default: True)')
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
    fedavg, history = train_federated(
        X_train, y_train, X_val, y_val, X_test, y_test,
        num_classes=args.num_classes,
        input_shape=input_shape,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        iid=args.iid,
        save_path=args.save_path
    )
    
    print("\nFederated training completed!")


if __name__ == '__main__':
    main()
