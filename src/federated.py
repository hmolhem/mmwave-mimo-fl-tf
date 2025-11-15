"""
Federated Learning implementation using FedAvg for mmWave MIMO classification.

Core components:
- FederatedServer: manages global model, aggregates client updates
- train_federated_round: simulates one FL round with client local training
- Weighted averaging based on client dataset sizes
"""

from __future__ import annotations

import os
import json
import copy
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_loading import load_day_train_as_clients, load_day_test, make_tf_dataset
from models import set_seeds, build_baseline_cnn, build_improved_cnn, compile_model
from evaluation import generate_full_report


class FederatedServer:
    """
    Federated learning server implementing FedAvg aggregation.
    """
    
    def __init__(self, global_model: keras.Model):
        """
        Args:
            global_model: Initial global model (compiled)
        """
        self.global_model = global_model
        self.round_history = {
            "round": [],
            "train_loss": [],
            "train_accuracy": [],
        }
    
    def get_global_weights(self) -> List[np.ndarray]:
        """Return current global model weights."""
        return self.global_model.get_weights()
    
    def set_global_weights(self, weights: List[np.ndarray]):
        """Update global model with new weights."""
        self.global_model.set_weights(weights)
    
    def aggregate_weights(
        self,
        client_weights: List[List[np.ndarray]],
        client_sizes: List[int],
    ) -> List[np.ndarray]:
        """
        FedAvg: weighted average of client model weights by dataset size.
        
        Args:
            client_weights: List of weight lists from each client
            client_sizes: Number of samples per client
        
        Returns:
            Aggregated weights
        """
        total_size = sum(client_sizes)
        
        # Initialize with zeros matching shape of first client
        num_layers = len(client_weights[0])
        aggregated = [np.zeros_like(w) for w in client_weights[0]]
        
        # Weighted sum
        for client_w, size in zip(client_weights, client_sizes):
            weight = size / total_size
            for i in range(num_layers):
                aggregated[i] += weight * client_w[i]
        
        return aggregated
    
    def federated_round(
        self,
        clients_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        local_epochs: int = 1,
        batch_size: int = 32,
        normalize: str = "zscore",
        verbose: int = 0,
    ) -> Dict:
        """
        Execute one federated learning round.
        
        Steps:
        1. Broadcast global weights to all clients
        2. Each client trains locally for local_epochs
        3. Collect client weights and sizes
        4. Aggregate using FedAvg
        5. Update global model
        
        Args:
            clients_data: Dict of client_id -> (X, y)
            local_epochs: Local training epochs per client
            batch_size: Batch size for client training
            normalize: Normalization method
            verbose: Verbosity level (0=silent, 1=progress)
        
        Returns:
            Dict with round metrics (avg loss, avg accuracy across clients)
        """
        global_weights = self.get_global_weights()
        
        client_weights_list = []
        client_sizes = []
        client_losses = []
        client_accs = []
        
        for client_id, (X, y) in clients_data.items():
            # Create local model with global weights
            local_model = keras.models.clone_model(self.global_model)
            local_model.set_weights(global_weights)
            compile_model(local_model, learning_rate=1e-3)
            
            # Train locally
            train_ds = make_tf_dataset(X, y, batch_size=batch_size, shuffle=True, normalize=normalize)
            history = local_model.fit(
                train_ds,
                epochs=local_epochs,
                verbose=verbose,
            )
            
            # Collect results
            client_weights_list.append(local_model.get_weights())
            client_sizes.append(X.shape[0])
            client_losses.append(history.history["loss"][-1])
            client_accs.append(history.history["accuracy"][-1])
            
            if verbose > 0:
                print(f"  Client {client_id}: {X.shape[0]} samples, "
                      f"loss={history.history['loss'][-1]:.4f}, "
                      f"acc={history.history['accuracy'][-1]:.4f}")
        
        # Aggregate
        aggregated_weights = self.aggregate_weights(client_weights_list, client_sizes)
        self.set_global_weights(aggregated_weights)
        
        # Compute round metrics
        round_metrics = {
            "num_clients": len(clients_data),
            "total_samples": sum(client_sizes),
            "avg_loss": float(np.mean(client_losses)),
            "avg_accuracy": float(np.mean(client_accs)),
            "client_sizes": client_sizes,
        }
        
        return round_metrics


def train_federated(
    day_dir: str,
    model_builder,
    num_rounds: int = 20,
    local_epochs: int = 1,
    batch_size: int = 32,
    normalize: str = "zscore",
    output_dir: str = "outputs/federated",
    run_name: str = "federated_experiment",
    seed: int = 42,
    verbose: int = 1,
) -> Tuple[keras.Model, Dict]:
    """
    Complete federated training pipeline.
    
    Args:
        day_dir: Path to day data folder (e.g., data/day0)
        model_builder: Function that returns a compiled model
        num_rounds: Number of FL rounds
        local_epochs: Local epochs per client per round
        batch_size: Batch size for local training
        normalize: Normalization method
        output_dir: Output directory for checkpoints and logs
        run_name: Experiment name
        seed: Random seed
        verbose: Verbosity (0=silent, 1=progress, 2=detailed)
    
    Returns:
        (final_global_model, training_history)
    """
    set_seeds(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load clients and test data
    print(f"Loading federated data from {day_dir}...")
    clients_data = load_day_train_as_clients(day_dir)
    X_test, y_test = load_day_test(day_dir)
    
    num_clients = len(clients_data)
    total_train = sum(x.shape[0] for x, _ in clients_data.values())
    print(f"Clients: {num_clients}, Total train samples: {total_train}, Test samples: {X_test.shape[0]}")
    
    # Initialize global model
    print("Initializing global model...")
    global_model = model_builder()
    compile_model(global_model)
    
    # Create server
    server = FederatedServer(global_model)
    
    # Training history
    history = {
        "round": [],
        "avg_train_loss": [],
        "avg_train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }
    
    # Federated training loop
    print(f"\nStarting federated training for {num_rounds} rounds...")
    test_ds = make_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False, normalize=normalize)
    
    for rnd in range(1, num_rounds + 1):
        if verbose > 0:
            print(f"\n--- Round {rnd}/{num_rounds} ---")
        
        # Execute FL round
        round_metrics = server.federated_round(
            clients_data=clients_data,
            local_epochs=local_epochs,
            batch_size=batch_size,
            normalize=normalize,
            verbose=max(0, verbose - 1),
        )
        
        # Evaluate global model on test set
        test_loss, test_acc = server.global_model.evaluate(test_ds, verbose=0)
        
        # Log metrics
        history["round"].append(rnd)
        history["avg_train_loss"].append(round_metrics["avg_loss"])
        history["avg_train_accuracy"].append(round_metrics["avg_accuracy"])
        history["test_loss"].append(float(test_loss))
        history["test_accuracy"].append(float(test_acc))
        
        if verbose > 0:
            print(f"Round {rnd}: Avg Train Loss={round_metrics['avg_loss']:.4f}, "
                  f"Avg Train Acc={round_metrics['avg_accuracy']:.4f}, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
    
    # Save final model
    model_path = os.path.join(output_dir, f"{run_name}_final.keras")
    server.global_model.save(model_path)
    print(f"\nFinal model saved to {model_path}")
    
    # Save history
    history_path = os.path.join(output_dir, "fl_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")
    
    return server.global_model, history
