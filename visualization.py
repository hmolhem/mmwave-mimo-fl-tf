"""
Visualization utilities for mmWave MIMO data and results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import os


def visualize_range_azimuth_sample(data: np.ndarray,
                                   label: Optional[int] = None,
                                   save_path: Optional[str] = None,
                                   title: Optional[str] = None,
                                   figsize: tuple = (8, 6)):
    """
    Visualize a single range-azimuth sample.
    
    Args:
        data: Range-azimuth data (H, W) or (H, W, 1)
        label: Optional class label
        save_path: Path to save figure
        title: Custom title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Handle different input shapes
    if len(data.shape) == 3 and data.shape[-1] == 1:
        data = data[:, :, 0]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap='viridis', aspect='auto', origin='lower')
    ax.set_xlabel('Azimuth Bins', fontsize=11)
    ax.set_ylabel('Range Bins', fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    elif label is not None:
        ax.set_title(f'Range-Azimuth Map (Class {label})', fontsize=13, fontweight='bold')
    else:
        ax.set_title('Range-Azimuth Map', fontsize=13, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Intensity')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def visualize_multiple_samples(data: np.ndarray,
                               labels: np.ndarray,
                               num_samples: int = 9,
                               save_path: Optional[str] = None,
                               figsize: tuple = (12, 12)):
    """
    Visualize multiple range-azimuth samples in a grid.
    
    Args:
        data: Array of range-azimuth samples (N, H, W, 1)
        labels: Array of labels (N,)
        num_samples: Number of samples to display
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    num_samples = min(num_samples, len(data))
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx in range(num_samples):
        ax = axes[idx]
        sample = data[idx, :, :, 0] if len(data.shape) == 4 else data[idx]
        
        im = ax.imshow(sample, cmap='viridis', aspect='auto', origin='lower')
        ax.set_title(f'Class {labels[idx]}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Range-Azimuth Sample Grid', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_class_distribution(labels: np.ndarray,
                            num_classes: int = 10,
                            save_path: Optional[str] = None,
                            title: str = 'Class Distribution',
                            figsize: tuple = (10, 6)):
    """
    Plot distribution of classes in dataset.
    
    Args:
        labels: Array of labels
        num_classes: Number of classes
        save_path: Path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(range(num_classes), 
                  [counts[list(unique).index(i)] if i in unique else 0 
                   for i in range(num_classes)],
                  color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_classes))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_federated_client_distribution(client_labels: List[np.ndarray],
                                       num_classes: int = 10,
                                       save_path: Optional[str] = None,
                                       figsize: tuple = (14, 6)):
    """
    Plot data distribution across federated clients.
    
    Args:
        client_labels: List of label arrays for each client
        num_classes: Number of classes
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    num_clients = len(client_labels)
    
    # Compute distribution for each client
    client_distributions = []
    for labels in client_labels:
        unique, counts = np.unique(labels, return_counts=True)
        dist = np.zeros(num_classes)
        for u, c in zip(unique, counts):
            dist[int(u)] = c
        client_distributions.append(dist)
    
    client_distributions = np.array(client_distributions)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(num_clients)
    width = 0.8
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    bottom = np.zeros(num_clients)
    for class_idx in range(num_classes):
        values = client_distributions[:, class_idx]
        ax.bar(x, values, width, label=f'Class {class_idx}',
               bottom=bottom, color=colors[class_idx])
        bottom += values
    
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Data Distribution Across Federated Clients', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i}' for i in range(num_clients)])
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_learning_curves_comparison(histories: List[dict],
                                    labels: List[str],
                                    metrics: List[str] = ['accuracy', 'loss'],
                                    save_path: Optional[str] = None,
                                    figsize: tuple = (14, 5)):
    """
    Compare learning curves from multiple training runs.
    
    Args:
        histories: List of training history dictionaries
        labels: List of labels for each history
        metrics: Metrics to plot
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for hist_idx, (history, label, color) in enumerate(zip(histories, labels, colors)):
            # Training metric
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], 
                       label=f'{label} (train)',
                       color=color, linestyle='-', linewidth=2)
            
            # Validation metric
            val_metric = f'val_{metric}'
            if val_metric in history:
                epochs = range(1, len(history[val_metric]) + 1)
                ax.plot(epochs, history[val_metric],
                       label=f'{label} (val)',
                       color=color, linestyle='--', linewidth=2)
        
        ax.set_xlabel('Epoch/Round', fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def create_result_summary_plot(centralized_results: dict,
                               federated_results: dict,
                               save_path: Optional[str] = None,
                               figsize: tuple = (14, 8)):
    """
    Create a comprehensive summary plot comparing centralized and federated results.
    
    Args:
        centralized_results: Results from centralized training
        federated_results: Results from federated training
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Centralized', 'Federated']
    accuracies = [centralized_results.get('accuracy', 0), 
                  federated_results.get('accuracy', 0)]
    bars = ax1.bar(models, accuracies, color=['steelblue', 'orange'], alpha=0.8)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Training curves (accuracy)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'training_history' in centralized_results:
        hist = centralized_results['training_history']
        if 'accuracy' in hist:
            ax2.plot(hist['accuracy'], label='Centralized', linewidth=2, color='steelblue')
    if 'train_accuracy' in federated_results:
        ax2.plot(federated_results['train_accuracy'], label='Federated', 
                linewidth=2, color='orange')
    ax2.set_xlabel('Epoch/Round', fontsize=11)
    ax2.set_ylabel('Training Accuracy', fontsize=11)
    ax2.set_title('Training Progress', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion matrices would go here if we had the data
    ax3 = fig.add_subplot(gs[1, :])
    ax3.text(0.5, 0.5, 'Result Summary\n' + 
             f"Centralized Accuracy: {centralized_results.get('accuracy', 0):.4f}\n" +
             f"Federated Accuracy: {federated_results.get('accuracy', 0):.4f}",
             ha='center', va='center', fontsize=14)
    ax3.axis('off')
    
    plt.suptitle('Centralized vs Federated Learning - Summary', 
                fontsize=15, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig
