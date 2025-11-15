"""
Federated learning implementation using FedAvg algorithm.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import copy

from model import create_cnn_model, create_lightweight_cnn, compile_model
from data_pipeline import create_federated_datasets, create_client_data_from_splits


class FederatedAveraging:
    """
    Implements the Federated Averaging (FedAvg) algorithm.
    """
    
    def __init__(self,
                 model_fn,
                 num_classes: int = 10,
                 input_shape: tuple = (128, 128, 1),
                 learning_rate: float = 0.001,
                 num_clients: int = 10,
                 clients_per_round: int = 5,
                 local_epochs: int = 5,
                 batch_size: int = 32):
        """
        Initialize FedAvg trainer.
        
        Args:
            model_fn: Function that creates and returns a Keras model
            num_classes: Number of output classes
            input_shape: Input shape for the model
            learning_rate: Learning rate for client training
            num_clients: Total number of clients
            clients_per_round: Number of clients to sample per round
            local_epochs: Number of local epochs per client
            batch_size: Batch size for client training
        """
        self.model_fn = model_fn
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.num_clients = num_clients
        self.clients_per_round = min(clients_per_round, num_clients)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        
        # Create global model
        self.global_model = model_fn(input_shape=input_shape, num_classes=num_classes)
        self.global_model = compile_model(self.global_model, learning_rate=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def get_model_weights(self, model: keras.Model) -> List[np.ndarray]:
        """Get model weights as list of numpy arrays."""
        return model.get_weights()
    
    def set_model_weights(self, model: keras.Model, weights: List[np.ndarray]):
        """Set model weights from list of numpy arrays."""
        model.set_weights(weights)
    
    def average_weights(self, weights_list: List[List[np.ndarray]],
                       sample_sizes: List[int]) -> List[np.ndarray]:
        """
        Average model weights weighted by number of samples.
        
        Args:
            weights_list: List of weight lists from different clients
            sample_sizes: Number of samples for each client
            
        Returns:
            Averaged weights
        """
        total_samples = sum(sample_sizes)
        
        # Initialize averaged weights with zeros
        avg_weights = [np.zeros_like(w) for w in weights_list[0]]
        
        # Weighted average
        for client_weights, n_samples in zip(weights_list, sample_sizes):
            weight = n_samples / total_samples
            for i, w in enumerate(client_weights):
                avg_weights[i] += weight * w
        
        return avg_weights
    
    def train_client(self,
                     client_data: Tuple[np.ndarray, np.ndarray],
                     global_weights: List[np.ndarray],
                     verbose: int = 0) -> Tuple[List[np.ndarray], int, float, float]:
        """
        Train a client model on local data.
        
        Args:
            client_data: Tuple of (X, y) for client
            global_weights: Current global model weights
            verbose: Verbosity level
            
        Returns:
            Tuple of (updated_weights, num_samples, loss, accuracy)
        """
        X_client, y_client = client_data
        n_samples = len(X_client)
        
        # Create client model and set to global weights
        client_model = self.model_fn(input_shape=self.input_shape, 
                                    num_classes=self.num_classes)
        client_model = compile_model(client_model, learning_rate=self.learning_rate)
        self.set_model_weights(client_model, global_weights)
        
        # Train on client data
        history = client_model.fit(
            X_client, y_client,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=verbose,
            validation_split=0.1
        )
        
        # Get updated weights and metrics
        updated_weights = self.get_model_weights(client_model)
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        return updated_weights, n_samples, final_loss, final_accuracy
    
    def train_round(self,
                   client_datasets: List[Tuple[np.ndarray, np.ndarray]],
                   round_num: int,
                   verbose: int = 1) -> Dict:
        """
        Perform one round of federated training.
        
        Args:
            client_datasets: List of (X, y) tuples for each client
            round_num: Current round number
            verbose: Verbosity level
            
        Returns:
            Dictionary with round metrics
        """
        # Sample clients for this round
        selected_clients = np.random.choice(
            self.num_clients,
            size=self.clients_per_round,
            replace=False
        )
        
        if verbose:
            print(f"\nRound {round_num}: Selected clients {selected_clients}")
        
        # Get current global weights
        global_weights = self.get_model_weights(self.global_model)
        
        # Train selected clients
        client_weights_list = []
        client_sample_sizes = []
        client_losses = []
        client_accuracies = []
        
        for client_idx in selected_clients:
            client_data = client_datasets[client_idx]
            
            # Train client
            weights, n_samples, loss, acc = self.train_client(
                client_data, global_weights, verbose=0
            )
            
            client_weights_list.append(weights)
            client_sample_sizes.append(n_samples)
            client_losses.append(loss)
            client_accuracies.append(acc)
            
            if verbose > 1:
                print(f"  Client {client_idx}: loss={loss:.4f}, acc={acc:.4f}, "
                      f"samples={n_samples}")
        
        # Average weights
        avg_weights = self.average_weights(client_weights_list, client_sample_sizes)
        self.set_model_weights(self.global_model, avg_weights)
        
        # Compute round metrics
        round_metrics = {
            'round': round_num,
            'avg_train_loss': np.mean(client_losses),
            'avg_train_accuracy': np.mean(client_accuracies),
            'num_clients': len(selected_clients)
        }
        
        if verbose:
            print(f"Round {round_num} - Avg Loss: {round_metrics['avg_train_loss']:.4f}, "
                  f"Avg Accuracy: {round_metrics['avg_train_accuracy']:.4f}")
        
        return round_metrics
    
    def evaluate_global_model(self,
                             X_val: np.ndarray,
                             y_val: np.ndarray,
                             verbose: int = 0) -> Tuple[float, float]:
        """
        Evaluate the global model on validation data.
        
        Args:
            X_val: Validation data
            y_val: Validation labels
            verbose: Verbosity level
            
        Returns:
            Tuple of (loss, accuracy)
        """
        loss, accuracy = self.global_model.evaluate(X_val, y_val, 
                                                     batch_size=self.batch_size,
                                                     verbose=verbose)
        return loss, accuracy
    
    def train(self,
             client_datasets: List[Tuple[np.ndarray, np.ndarray]],
             num_rounds: int,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             verbose: int = 1) -> Dict:
        """
        Train using federated averaging for multiple rounds.
        
        Args:
            client_datasets: List of (X, y) tuples for each client
            num_rounds: Number of federated rounds
            X_val: Optional validation data
            y_val: Optional validation labels
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        print("\n" + "="*60)
        print("FEDERATED TRAINING (FedAvg)")
        print("="*60)
        print(f"Total clients: {self.num_clients}")
        print(f"Clients per round: {self.clients_per_round}")
        print(f"Local epochs: {self.local_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total rounds: {num_rounds}")
        print("="*60 + "\n")
        
        for round_num in range(1, num_rounds + 1):
            # Train one round
            round_metrics = self.train_round(client_datasets, round_num, verbose=verbose)
            
            # Record training metrics
            self.history['train_loss'].append(round_metrics['avg_train_loss'])
            self.history['train_accuracy'].append(round_metrics['avg_train_accuracy'])
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate_global_model(X_val, y_val, verbose=0)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                if verbose:
                    print(f"Round {round_num} - Val Loss: {val_loss:.4f}, "
                          f"Val Accuracy: {val_acc:.4f}")
        
        print("\nFederated training completed!")
        return self.history
    
    def get_global_model(self) -> keras.Model:
        """Return the global model."""
        return self.global_model
    
    def save_global_model(self, filepath: str):
        """Save the global model."""
        self.global_model.save(filepath)
        print(f"Global model saved to {filepath}")
