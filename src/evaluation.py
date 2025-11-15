"""
Evaluation utilities for mmWave MIMO classification experiments.

Features:
- Confusion matrix heatmaps
- Per-class accuracy and metrics
- Safety-aware grouping (near/mid/far/empty)
- Training history plots
- Classification reports
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)


# Define safety-aware class groupings based on proposal description
# Class 0: empty workspace
# Classes 1-9: different distance/azimuth bins
# For safety analysis, we can group by approximate distance ranges
SAFETY_GROUPS = {
    "empty": [0],
    "near": [2],  # ~0.3-0.5m (closest to robot)
    "mid": [1, 3, 4, 5],  # ~0.5-1.0m range
    "far": [6, 7, 8, 9],  # >1.0m range
}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save figure (optional)
        title: Plot title
        class_names: Custom class names (default: 0-9)
        normalize: If True, normalize by row (true class)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = ".2f"
    else:
        fmt = "d"
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Count" if normalize else "Count"},
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Per-Class Metrics",
    class_names: Optional[List[str]] = None,
):
    """
    Bar plot of per-class precision, recall, F1, and accuracy.
    """
    num_classes = max(y_true.max(), y_pred.max()) + 1
    if class_names is None:
        class_names = [f"C{i}" for i in range(num_classes)]
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    
    # Per-class accuracy
    per_class_acc = []
    for cls in range(num_classes):
        mask = y_true == cls
        if mask.sum() > 0:
            acc = (y_pred[mask] == cls).sum() / mask.sum()
        else:
            acc = 0.0
        per_class_acc.append(acc)
    per_class_acc = np.array(per_class_acc)
    
    x = np.arange(num_classes)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 1.5*width, precision, width, label="Precision", alpha=0.9)
    ax.bar(x - 0.5*width, recall, width, label="Recall", alpha=0.9)
    ax.bar(x + 0.5*width, f1, width, label="F1-Score", alpha=0.9)
    ax.bar(x + 1.5*width, per_class_acc, width, label="Accuracy", alpha=0.9)
    
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved per-class metrics to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_safety_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute safety-aware metrics by grouping classes into near/mid/far/empty.
    
    Returns dict with:
    - group_accuracy: accuracy per safety group
    - critical_errors: count of dangerous misclassifications (e.g., near predicted as empty)
    """
    results = {}
    
    # Map labels to groups
    def label_to_group(label):
        for group_name, classes in SAFETY_GROUPS.items():
            if label in classes:
                return group_name
        return "unknown"
    
    y_true_groups = np.array([label_to_group(y) for y in y_true])
    y_pred_groups = np.array([label_to_group(y) for y in y_pred])
    
    # Group-level accuracy
    group_names = list(SAFETY_GROUPS.keys())
    for group in group_names:
        mask = y_true_groups == group
        if mask.sum() > 0:
            acc = (y_pred_groups[mask] == group).sum() / mask.sum()
            results[f"{group}_accuracy"] = float(acc)
        else:
            results[f"{group}_accuracy"] = 0.0
    
    # Critical errors: near predicted as empty
    near_mask = y_true_groups == "near"
    if near_mask.sum() > 0:
        critical_near_empty = ((y_true_groups == "near") & (y_pred_groups == "empty")).sum()
        results["critical_near_as_empty"] = int(critical_near_empty)
        results["critical_near_as_empty_rate"] = float(critical_near_empty / near_mask.sum())
    else:
        results["critical_near_as_empty"] = 0
        results["critical_near_as_empty_rate"] = 0.0
    
    # Empty predicted as near (false alarm)
    empty_mask = y_true_groups == "empty"
    if empty_mask.sum() > 0:
        false_near = ((y_true_groups == "empty") & (y_pred_groups == "near")).sum()
        results["false_alarm_empty_as_near"] = int(false_near)
        results["false_alarm_empty_as_near_rate"] = float(false_near / empty_mask.sum())
    else:
        results["false_alarm_empty_as_near"] = 0
        results["false_alarm_empty_as_near_rate"] = 0.0
    
    return results


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
    title: str = "Training History",
):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dict with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        save_path: Path to save figure
        title: Plot title
    """
    epochs = range(1, len(history.get("loss", [])) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    if "loss" in history:
        ax1.plot(epochs, history["loss"], "b-", label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Model Loss", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    if "accuracy" in history:
        ax2.plot(epochs, history["accuracy"], "b-", label="Train Acc", linewidth=2)
    if "val_accuracy" in history:
        ax2.plot(epochs, history["val_accuracy"], "r-", label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Model Accuracy", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    history: Optional[Dict] = None,
    run_name: str = "experiment",
):
    """
    Generate a comprehensive evaluation report with all plots and metrics.
    
    Creates:
    - confusion_matrix.png (raw counts)
    - confusion_matrix_normalized.png
    - per_class_metrics.png
    - training_history.png (if history provided)
    - safety_metrics.txt
    - classification_report.txt
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrices
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
        title=f"{run_name} - Confusion Matrix",
    )
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "confusion_matrix_normalized.png"),
        title=f"{run_name} - Confusion Matrix (Normalized)",
        normalize=True,
    )
    
    # Per-class metrics
    plot_per_class_metrics(
        y_true, y_pred,
        save_path=os.path.join(output_dir, "per_class_metrics.png"),
        title=f"{run_name} - Per-Class Metrics",
    )
    
    # Training history
    if history:
        plot_training_history(
            history,
            save_path=os.path.join(output_dir, "training_history.png"),
            title=f"{run_name} - Training History",
        )
    
    # Safety metrics
    safety = compute_safety_metrics(y_true, y_pred)
    safety_path = os.path.join(output_dir, "safety_metrics.txt")
    with open(safety_path, "w") as f:
        f.write(f"Safety-Aware Metrics for {run_name}\n")
        f.write("=" * 60 + "\n\n")
        for key, val in safety.items():
            f.write(f"{key}: {val}\n")
    print(f"Saved safety metrics to {safety_path}")
    
    # Classification report
    report = classification_report(y_true, y_pred, digits=4)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report for {run_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"Saved classification report to {report_path}")
    
    print(f"\nFull evaluation report generated in: {output_dir}")
