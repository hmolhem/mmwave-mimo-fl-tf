"""
Federated Averaging (FedAvg) implementation for mmWave MIMO radar ROI classification.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path
import copy


class FedAvgTrainer:
    """Federated Averaging (FedAvg) trainer."""
    
    def __init__(self,
                 model: keras.Model,
                 client_datasets: List[tf.data.Dataset],
                 val_dataset: tf.data.Dataset,
                 num_clients: int,
                 save_dir: str = 'checkpoints',
                 experiment_name: str = 'fedavg'):
        """
        Initialize FedAvg trainer.
        
        Args:
            model: Global Keras model
            client_datasets: List of datasets for each client
            val_dataset: Validation dataset
            num_clients: Number of federated clients
            save_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
        """
        self.global_model = model
        self.client_datasets = client_datasets
        self.val_dataset = val_dataset
        self.num_clients = num_clients
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        
        # Create save directory
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def federated_averaging(self, client_weights: List[np.ndarray],
                           client_sizes: List[int]) -> np.ndarray:
        """
        Perform federated averaging of client model weights.
        
        Args:
            client_weights: List of weight arrays from each client
            client_sizes: Number of samples for each client
            
        Returns:
            Averaged weights
        """
        # Calculate weights based on dataset sizes
        total_samples = sum(client_sizes)
        scaling_factors = [size / total_samples for size in client_sizes]
        
        # Initialize averaged weights
        avg_weights = []
        
        # Average each layer's weights
        num_layers = len(client_weights[0])
        for layer_idx in range(num_layers):
            # Get weights from all clients for this layer
            layer_weights = [client_weights[i][layer_idx] for i in range(len(client_weights))]
            
            # Weighted average
            avg_layer = np.zeros_like(layer_weights[0])
            for client_idx, weight in enumerate(layer_weights):
                avg_layer += scaling_factors[client_idx] * weight
            
            avg_weights.append(avg_layer)
        
        return avg_weights
    
    def train_client(self,
                    client_id: int,
                    client_dataset: tf.data.Dataset,
                    epochs: int = 1,
                    verbose: int = 0) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train a single client model.
        
        Args:
            client_id: Client identifier
            client_dataset: Client's dataset
            epochs: Number of local epochs
            verbose: Verbosity level
            
        Returns:
            Tuple of (client_weights, num_samples, metrics)
        """
        # Create a new model with the same architecture
        client_model = keras.models.clone_model(self.global_model)
        client_model.set_weights(self.global_model.get_weights())
        
        # Compile the client model with the same configuration
        client_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=self.global_model.loss,
            metrics=['accuracy']
        )
        
        # Train on client data
        history = client_model.fit(
            client_dataset,
            epochs=epochs,
            verbose=verbose
        )
        
        # Get number of samples
        num_samples = sum([batch[0].shape[0] for batch in client_dataset])
        
        # Get final metrics
        metrics = {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1]
        }
        
        return client_model.get_weights(), num_samples, metrics
    
    def train_round(self,
                   round_num: int,
                   client_epochs: int = 1,
                   clients_per_round: Optional[int] = None,
                   verbose: int = 1) -> Dict:
        """
        Execute one round of federated training.
        
        Args:
            round_num: Current round number
            client_epochs: Number of local epochs per client
            clients_per_round: Number of clients to sample (None = all clients)
            verbose: Verbosity level
            
        Returns:
            Round metrics
        """
        if clients_per_round is None:
            clients_per_round = self.num_clients
        
        # Sample clients for this round
        selected_clients = np.random.choice(
            self.num_clients,
            size=min(clients_per_round, self.num_clients),
            replace=False
        )
        
        if verbose:
            print(f"\n--- Round {round_num} ---")
            print(f"Selected clients: {selected_clients}")
        
        # Train each selected client
        client_weights_list = []
        client_sizes = []
        client_metrics = []
        
        for client_id in selected_clients:
            if verbose:
                print(f"Training client {client_id}...")
            
            weights, num_samples, metrics = self.train_client(
                client_id=client_id,
                client_dataset=self.client_datasets[client_id],
                epochs=client_epochs,
                verbose=0
            )
            
            client_weights_list.append(weights)
            client_sizes.append(num_samples)
            client_metrics.append(metrics)
            
            if verbose:
                print(f"  Client {client_id} - Loss: {metrics['loss']:.4f}, "
                      f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Aggregate client weights
        averaged_weights = self.federated_averaging(client_weights_list, client_sizes)
        
        # Update global model
        self.global_model.set_weights(averaged_weights)
        
        # Evaluate global model on validation set
        val_results = self.global_model.evaluate(self.val_dataset, verbose=0)
        val_loss = val_results[0]
        val_accuracy = val_results[1]
        
        # Calculate average training metrics
        avg_train_loss = np.mean([m['loss'] for m in client_metrics])
        avg_train_accuracy = np.mean([m['accuracy'] for m in client_metrics])
        
        if verbose:
            print(f"Round {round_num} - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Store metrics
        round_metrics = {
            'round': round_num,
            'train_loss': float(avg_train_loss),
            'train_accuracy': float(avg_train_accuracy),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy)
        }
        
        return round_metrics
    
    def train(self,
             num_rounds: int = 50,
             client_epochs: int = 1,
             clients_per_round: Optional[int] = None,
             save_frequency: int = 10,
             verbose: int = 1) -> Dict:
        """
        Execute federated training.
        
        Args:
            num_rounds: Number of federated rounds
            client_epochs: Number of local epochs per client
            clients_per_round: Number of clients per round
            save_frequency: Save model every N rounds
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        print(f"\nStarting FedAvg training for {num_rounds} rounds...")
        print(f"Clients: {self.num_clients}, Epochs per client: {client_epochs}")
        print(f"Experiment: {self.experiment_name}")
        
        best_val_accuracy = 0.0
        
        for round_num in range(1, num_rounds + 1):
            # Train one round
            round_metrics = self.train_round(
                round_num=round_num,
                client_epochs=client_epochs,
                clients_per_round=clients_per_round,
                verbose=verbose
            )
            
            # Update history
            for key, value in round_metrics.items():
                self.history[key].append(value)
            
            # Save best model
            if round_metrics['val_accuracy'] > best_val_accuracy:
                best_val_accuracy = round_metrics['val_accuracy']
                self.save_model('best_model.h5')
            
            # Periodic save
            if round_num % save_frequency == 0:
                self.save_model(f'model_round_{round_num}.h5')
                self.save_history()
        
        # Save final model
        self.save_model('final_model.h5')
        self.save_history()
        
        return self.history
    
    def save_model(self, filename: str):
        """Save the global model."""
        model_path = self.experiment_dir / filename
        self.global_model.save(model_path)
        
    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.experiment_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict:
        """
        Evaluate the global model on test data.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating global model...")
        results = self.global_model.evaluate(test_dataset, verbose=1)
        
        metrics_dict = {}
        for name, value in zip(self.global_model.metrics_names, results):
            metrics_dict[name] = float(value)
            print(f"{name}: {value:.4f}")
        
        # Save evaluation results
        eval_path = self.experiment_dir / 'evaluation.json'
        with open(eval_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        return metrics_dict


def train_federated(client_datasets: List[tf.data.Dataset],
                   val_dataset: tf.data.Dataset,
                   test_dataset: tf.data.Dataset,
                   model: keras.Model,
                   num_rounds: int = 50,
                   client_epochs: int = 1,
                   save_dir: str = 'checkpoints',
                   experiment_name: str = 'fedavg') -> Tuple[keras.Model, Dict]:
    """
    Convenience function for federated training.
    
    Args:
        client_datasets: List of client datasets
        val_dataset: Validation dataset
        test_dataset: Test dataset
        model: Keras model
        num_rounds: Number of federated rounds
        client_epochs: Number of local epochs
        save_dir: Save directory
        experiment_name: Experiment name
        
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    trainer = FedAvgTrainer(
        model=model,
        client_datasets=client_datasets,
        val_dataset=val_dataset,
        num_clients=len(client_datasets),
        save_dir=save_dir,
        experiment_name=experiment_name
    )
    
    # Train
    trainer.train(num_rounds=num_rounds, client_epochs=client_epochs)
    
    # Evaluate
    eval_metrics = trainer.evaluate(test_dataset)
    
    return trainer.global_model, eval_metrics
