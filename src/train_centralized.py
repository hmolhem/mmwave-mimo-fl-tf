"""
Centralized training script for mmWave MIMO position classification.

Usage:
    python src/train_centralized.py --day 0 --model baseline --epochs 50 --batch_size 32
"""

from __future__ import annotations

import argparse
import os
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_loading import make_centralized_tf_datasets, make_tf_dataset
from models import (
    set_seeds,
    build_baseline_cnn,
    build_improved_cnn,
    compile_model,
    get_model_callbacks,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for mmWave MIMO classification (centralized)")
    parser.add_argument("--day", type=int, default=0, choices=[0, 1, 2], help="Training day (0, 1, or 2)")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "improved"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--normalize", type=str, default="zscore", choices=["none", "zscore", "minmax", "log1p"], help="Normalization method")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for checkpoints and metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    
    # Setup paths
    day_dir = os.path.join("data", f"day{args.day}")
    run_name = f"{args.model}_day{args.day}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Load data
    print(f"\nLoading data from {day_dir}...")
    train_ds, val_ds, (X_test, y_test) = make_centralized_tf_datasets(
        day_dir=day_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )
    test_ds = make_tf_dataset(X_test, y_test, batch_size=args.batch_size, shuffle=False, normalize=args.normalize)
    
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build and compile model
    print(f"\nBuilding {args.model} model...")
    if args.model == "baseline":
        model = build_baseline_cnn(dropout_rate=args.dropout)
    else:
        model = build_improved_cnn(dropout_rate=args.dropout)
    
    compile_model(model, learning_rate=args.lr)
    model.summary()
    
    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")
    callbacks = get_model_callbacks(checkpoint_dir=output_dir, model_name=run_name, patience=args.patience)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save history
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    print(f"History saved to {history_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    # Predictions and confusion matrix
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    # Save metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"Training complete: {run_name}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Results: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
