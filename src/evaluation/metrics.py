"""
Evaluation utilities for mmWave MIMO radar ROI classification.
Includes accuracy metrics, confusion matrix, and visualization.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from typing import Dict, Optional, Tuple, List
import json
from pathlib import Path


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, model: keras.Model, save_dir: str = 'results'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
            save_dir: Directory to save evaluation results
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_dataset(self,
                        dataset: tf.data.Dataset,
                        class_names: Optional[List[str]] = None) -> Dict:
        """
        Comprehensive evaluation of a dataset.
        
        Args:
            dataset: Test dataset
            class_names: Optional list of class names
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions and true labels
        y_true = []
        y_pred = []
        
        print("Generating predictions...")
        for batch_x, batch_y in dataset:
            predictions = self.model.predict(batch_x, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(batch_y.numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, class_names)
        
        # Generate confusion matrix
        cm = self.plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Save results
        self.save_results(metrics, cm)
        
        return metrics
    
    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None) -> Dict:
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names
            
        Returns:
            Dictionary of metrics
        """
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'per_class_metrics': {}
        }
        
        # Per-class metrics
        num_classes = len(np.unique(y_true))
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f"class_{i}"
            metrics['per_class_metrics'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        # Print metrics
        print("\n=== Evaluation Metrics ===")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"\nMacro Averages:")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall: {recall_macro:.4f}")
        print(f"  F1-Score: {f1_macro:.4f}")
        print(f"\nWeighted Averages:")
        print(f"  Precision: {precision_weighted:.4f}")
        print(f"  Recall: {recall_weighted:.4f}")
        print(f"  F1-Score: {f1_weighted:.4f}")
        
        # Classification report
        if class_names:
            target_names = class_names
        else:
            target_names = [f"class_{i}" for i in range(num_classes)]
        
        print("\n=== Classification Report ===")
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        return metrics
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             normalize: bool = False) -> np.ndarray:
        """
        Plot and save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names
            normalize: Whether to normalize the confusion matrix
            
        Returns:
            Confusion matrix array
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        num_classes = cm.shape[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names if class_names else range(num_classes),
            yticklabels=class_names if class_names else range(num_classes),
            ax=ax
        )
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        ax.set_title(title)
        
        plt.tight_layout()
        
        # Save figure
        filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
        plt.close()
        
        # Also plot normalized version if we plotted unnormalized
        if not normalize:
            self.plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
        
        return cm
    
    def plot_training_history(self, history: Dict, save_name: str = 'training_history.png'):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            save_name: Filename to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Check if history has validation data
        has_val = 'val_loss' in history or 'val_accuracy' in history
        
        # Plot loss
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss')
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss')
            
        if has_val:
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Val Loss')
                
        axes[0].set_xlabel('Epoch/Round')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Train Accuracy')
        if 'train_accuracy' in history:
            axes[1].plot(history['train_accuracy'], label='Train Accuracy')
            
        if has_val:
            if 'val_accuracy' in history:
                axes[1].plot(history['val_accuracy'], label='Val Accuracy')
                
        axes[1].set_xlabel('Epoch/Round')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()
    
    def save_results(self, metrics: Dict, confusion_matrix: np.ndarray):
        """
        Save evaluation results to JSON.
        
        Args:
            metrics: Metrics dictionary
            confusion_matrix: Confusion matrix array
        """
        results = {
            'metrics': metrics,
            'confusion_matrix': confusion_matrix.tolist()
        }
        
        results_path = self.save_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {results_path}")


def evaluate_model(model: keras.Model,
                  test_dataset: tf.data.Dataset,
                  class_names: Optional[List[str]] = None,
                  save_dir: str = 'results') -> Dict:
    """
    Convenience function for model evaluation.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        class_names: Optional list of class names
        save_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = ModelEvaluator(model, save_dir)
    metrics = evaluator.evaluate_dataset(test_dataset, class_names)
    return metrics


def compare_models(models: Dict[str, keras.Model],
                  test_dataset: tf.data.Dataset,
                  class_names: Optional[List[str]] = None,
                  save_dir: str = 'results') -> Dict:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of model_name -> model
        test_dataset: Test dataset
        class_names: Optional list of class names
        save_dir: Directory to save results
        
    Returns:
        Dictionary of comparison results
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    comparison = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        model_save_dir = save_path / model_name
        evaluator = ModelEvaluator(model, str(model_save_dir))
        metrics = evaluator.evaluate_dataset(test_dataset, class_names)
        
        comparison[model_name] = metrics
    
    # Save comparison results
    comparison_path = save_path / 'model_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\nModel comparison saved to {comparison_path}")
    
    # Print comparison summary
    print("\n=== Model Comparison Summary ===")
    print(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 40)
    for model_name, metrics in comparison.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['f1_weighted']:<10.4f}")
    
    return comparison
