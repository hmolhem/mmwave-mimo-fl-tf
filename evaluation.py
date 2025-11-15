"""
Evaluation utilities for model performance assessment.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from typing import Optional, Dict, List
import seaborn as sns


def evaluate_model(model: tf.keras.Model,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  batch_size: int = 32,
                  verbose: int = 1) -> Dict:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels
        batch_size: Batch size for evaluation
        verbose: Verbosity level
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=verbose)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Generate classification report
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred_classes,
        'probabilities': y_pred
    }
    
    if verbose:
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))
    
    return results


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         num_classes: int = 10,
                         normalize: bool = False,
                         title: str = 'Confusion Matrix',
                         cmap: str = 'Blues',
                         save_path: Optional[str] = None,
                         figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        normalize: Whether to normalize the confusion matrix
        title: Plot title
        cmap: Colormap
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                square=True, cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set tick labels
    ax.set_xticks(np.arange(num_classes) + 0.5)
    ax.set_yticks(np.arange(num_classes) + 0.5)
    ax.set_xticklabels(range(num_classes))
    ax.set_yticklabels(range(num_classes))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_training_history(history: Dict,
                          metrics: List[str] = ['accuracy', 'loss'],
                          save_path: Optional[str] = None,
                          figsize: tuple = (12, 5)) -> plt.Figure:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary from model.fit()
        metrics: List of metrics to plot
        save_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training metric
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric.capitalize()}', 
                   linewidth=2, marker='o', markersize=4)
        
        # Plot validation metric
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric.capitalize()}',
                   linewidth=2, marker='s', markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f'Training and Validation {metric.capitalize()}', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    return fig


def compute_per_class_accuracy(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               num_classes: int = 10) -> Dict:
    """
    Compute per-class accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class index to accuracy
    """
    per_class_acc = {}
    
    for class_idx in range(num_classes):
        mask = y_true == class_idx
        if mask.sum() > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            per_class_acc[class_idx] = class_acc
        else:
            per_class_acc[class_idx] = 0.0
    
    return per_class_acc


def print_evaluation_summary(results: Dict, num_classes: int = 10):
    """
    Print a comprehensive evaluation summary.
    
    Args:
        results: Results dictionary from evaluate_model()
        num_classes: Number of classes
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    report = results['classification_report']
    for i in range(num_classes):
        class_key = str(i)
        if class_key in report:
            precision = report[class_key]['precision']
            recall = report[class_key]['recall']
            f1 = report[class_key]['f1-score']
            print(f"{i:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    # Macro and weighted averages
    print("-" * 60)
    if 'macro avg' in report:
        macro = report['macro avg']
        print(f"{'Macro Avg':<10} {macro['precision']:<12.4f} "
              f"{macro['recall']:<12.4f} {macro['f1-score']:<12.4f}")
    
    if 'weighted avg' in report:
        weighted = report['weighted avg']
        print(f"{'Weighted Avg':<10} {weighted['precision']:<12.4f} "
              f"{weighted['recall']:<12.4f} {weighted['f1-score']:<12.4f}")
    
    print("="*60 + "\n")


def compare_models(results_list: List[Dict],
                  model_names: List[str],
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare multiple models' performance.
    
    Args:
        results_list: List of result dictionaries
        model_names: List of model names
        save_path: Path to save the comparison plot
        
    Returns:
        Matplotlib figure
    """
    accuracies = [r['accuracy'] for r in results_list]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, accuracies, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    return fig
