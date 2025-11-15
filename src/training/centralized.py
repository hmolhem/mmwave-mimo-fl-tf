"""
Centralized training implementation for mmWave MIMO radar ROI classification.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Optional, Tuple
import json
from pathlib import Path


class CentralizedTrainer:
    """Centralized training for CNN models."""
    
    def __init__(self,
                 model: keras.Model,
                 train_dataset: tf.data.Dataset,
                 val_dataset: tf.data.Dataset,
                 save_dir: str = 'checkpoints',
                 experiment_name: str = 'centralized'):
        """
        Initialize centralized trainer.
        
        Args:
            model: Compiled Keras model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            save_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        
        # Create save directory
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = None
        
    def train(self,
             epochs: int = 50,
             callbacks: Optional[list] = None,
             verbose: int = 1) -> Dict:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            callbacks: List of Keras callbacks
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        print(f"\nStarting centralized training for {epochs} epochs...")
        print(f"Experiment: {self.experiment_name}")
        print(f"Save directory: {self.experiment_dir}")
        
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Save final model
        self.save_model('final_model.h5')
        
        # Save training history
        self.save_history()
        
        return self.history.history
    
    def _get_default_callbacks(self) -> list:
        """
        Get default callbacks for training.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            # Model checkpoint - save best model
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.experiment_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=str(self.experiment_dir / 'logs'),
                histogram_freq=1,
                write_graph=True
            ),
            
            # CSV logger
            keras.callbacks.CSVLogger(
                filename=str(self.experiment_dir / 'training_log.csv'),
                append=True
            )
        ]
        
        return callbacks
    
    def save_model(self, filename: str):
        """
        Save the model.
        
        Args:
            filename: Filename for the model
        """
        model_path = self.experiment_dir / filename
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def save_history(self):
        """Save training history to JSON file."""
        if self.history is not None:
            history_path = self.experiment_dir / 'history.json'
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_dict = {}
                for key, value in self.history.history.items():
                    history_dict[key] = [float(v) for v in value]
                json.dump(history_dict, f, indent=2)
            print(f"Training history saved to {history_path}")
    
    def load_model(self, filename: str):
        """
        Load a saved model.
        
        Args:
            filename: Filename of the model
        """
        model_path = self.experiment_dir / filename
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return self.model
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model...")
        results = self.model.evaluate(test_dataset, verbose=1)
        
        metrics_dict = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics_dict[name] = float(value)
            print(f"{name}: {value:.4f}")
        
        # Save evaluation results
        eval_path = self.experiment_dir / 'evaluation.json'
        with open(eval_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        return metrics_dict
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: Input data
            
        Returns:
            Predictions
        """
        return self.model.predict(data)


def train_centralized(train_dataset: tf.data.Dataset,
                     val_dataset: tf.data.Dataset,
                     test_dataset: tf.data.Dataset,
                     model: keras.Model,
                     epochs: int = 50,
                     save_dir: str = 'checkpoints',
                     experiment_name: str = 'centralized') -> Tuple[keras.Model, Dict]:
    """
    Convenience function for centralized training.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        model: Keras model
        epochs: Number of epochs
        save_dir: Save directory
        experiment_name: Experiment name
        
    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    trainer = CentralizedTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_dir=save_dir,
        experiment_name=experiment_name
    )
    
    # Train
    trainer.train(epochs=epochs)
    
    # Evaluate
    eval_metrics = trainer.evaluate(test_dataset)
    
    return trainer.model, eval_metrics
