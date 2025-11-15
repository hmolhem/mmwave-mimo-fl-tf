"""
Cross-day robustness evaluation: train on one day, test on others.

This module trains models on a source day and evaluates on target days to
measure domain shift and model generalization across different temporal conditions.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from data_loading import (
    load_day_train_concatenated,
    load_day_test,
    make_tf_dataset,
)
from models import build_baseline_cnn, build_improved_cnn, compile_model, set_seeds
from evaluation import generate_full_report


def train_on_day(
    train_day,
    model_type="baseline",
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    dropout=0.3,
    normalize="zscore",
    val_ratio=0.2,
    patience=10,
    seed=42,
    output_dir="outputs/cross_day",
):
    """
    Train a model on a single day's training data.

    Parameters
    ----------
    train_day : int
        Day to train on (0, 1, or 2).
    model_type : str
        Model architecture: 'baseline' or 'improved'.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size for training.
    learning_rate : float
        Initial learning rate.
    dropout : float
        Dropout rate.
    normalize : str
        Normalization strategy: 'zscore', 'minmax', or 'log1p'.
    val_ratio : float
        Validation split ratio.
    patience : int
        Early stopping patience.
    seed : int
        Random seed.
    output_dir : str
        Directory to save model and metrics.

    Returns
    -------
    model : keras.Model
        Trained model.
    history : keras.callbacks.History
        Training history.
    output_path : Path
        Path to saved outputs.
    """
    set_seeds(seed)

    # Create output directory
    output_path = Path(output_dir) / f"train_day{train_day}_{model_type}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training on day{train_day} with {model_type} model")
    print(f"{'='*60}\n")

    # Construct day directory path
    data_root = Path(__file__).parent.parent / "data"
    day_dir = data_root / f"day{train_day}"
    
    # Load training data
    X_train, y_train, device_ids = load_day_train_concatenated(str(day_dir))
    print(f"Loaded {len(X_train)} training samples from day{train_day}")

    # Split train/validation
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=seed, stratify=y_train
    )

    print(f"Train: {len(X_tr)} samples, Val: {len(X_val)} samples")

    # Create TF datasets
    train_ds = make_tf_dataset(X_tr, y_tr, batch_size, shuffle=True, normalize=normalize)
    val_ds = make_tf_dataset(X_val, y_val, batch_size, shuffle=False, normalize=normalize)

    # Build model
    if model_type == "baseline":
        model = build_baseline_cnn(dropout_rate=dropout)
    elif model_type == "improved":
        model = build_improved_cnn(dropout_rate=dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    compile_model(model, learning_rate=learning_rate)

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(output_path / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=patience // 2, verbose=1, min_lr=1e-7
        ),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training history
    history_dict = {
        "train_loss": [float(x) for x in history.history["loss"]],
        "train_accuracy": [float(x) for x in history.history["accuracy"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
    }
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    # Save training config
    config = {
        "train_day": train_day,
        "model_type": model_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "normalize": normalize,
        "val_ratio": val_ratio,
        "patience": patience,
        "seed": seed,
        "train_samples": len(X_tr),
        "val_samples": len(X_val),
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nModel and config saved to: {output_path}")

    return model, history, output_path


def evaluate_on_day(
    model,
    test_day,
    normalize="zscore",
    batch_size=32,
    output_dir=None,
    train_day=None,
    model_type=None,
):
    """
    Evaluate a trained model on a target day's test data.

    Parameters
    ----------
    model : keras.Model
        Trained model.
    test_day : int
        Day to test on (0, 1, or 2).
    normalize : str
        Normalization strategy (must match training).
    batch_size : int
        Batch size for evaluation.
    output_dir : Path or str, optional
        Directory to save evaluation results.
    train_day : int, optional
        Day model was trained on (for naming).
    model_type : str, optional
        Model type (for naming).

    Returns
    -------
    metrics : dict
        Test metrics including loss, accuracy, and per-class results.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on day{test_day} test data")
    print(f"{'='*60}\n")

    # Construct day directory path
    data_root = Path(__file__).parent.parent / "data"
    day_dir = data_root / f"day{test_day}"
    
    # Load test data
    X_test, y_test = load_day_test(str(day_dir))
    print(f"Loaded {len(X_test)} test samples from day{test_day}")

    # Create TF dataset
    test_ds = make_tf_dataset(X_test, y_test, batch_size, shuffle=False, normalize=normalize)

    # Evaluate
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Predictions
    y_pred = model.predict(test_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Save metrics
    metrics = {
        "test_day": test_day,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "num_samples": len(y_test),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        metrics_file = output_dir / f"test_day{test_day}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Generate full evaluation report
        report_dir = output_dir / f"test_day{test_day}_report"
        report_dir.mkdir(parents=True, exist_ok=True)

        title_suffix = ""
        if train_day is not None and model_type is not None:
            title_suffix = f" (Train day{train_day}, {model_type})"

        generate_full_report(
            y_true=y_test,
            y_pred=y_pred_classes,
            history=None,  # No training history for cross-day evaluation
            output_dir=str(report_dir),
            run_name=f"test_day{test_day}{title_suffix}",
        )

        print(f"Evaluation results saved to: {output_dir}")

    return metrics


def cross_day_experiment(
    train_day,
    test_days=None,
    model_type="baseline",
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    dropout=0.3,
    normalize="zscore",
    val_ratio=0.2,
    patience=10,
    seed=42,
    output_dir="outputs/cross_day",
):
    """
    Run complete cross-day experiment: train on one day, test on all days.

    Parameters
    ----------
    train_day : int
        Day to train on (0, 1, or 2).
    test_days : list of int, optional
        Days to test on. If None, tests on all days (0, 1, 2).
    model_type : str
        Model architecture: 'baseline' or 'improved'.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Initial learning rate.
    dropout : float
        Dropout rate.
    normalize : str
        Normalization strategy: 'zscore', 'minmax', or 'log1p'.
    val_ratio : float
        Validation split ratio.
    patience : int
        Early stopping patience.
    seed : int
        Random seed.
    output_dir : str
        Directory to save all outputs.

    Returns
    -------
    results : dict
        Dictionary mapping test_day to metrics.
    """
    if test_days is None:
        test_days = [0, 1, 2]

    # Train on source day
    model, history, train_output_path = train_on_day(
        train_day=train_day,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout=dropout,
        normalize=normalize,
        val_ratio=val_ratio,
        patience=patience,
        seed=seed,
        output_dir=output_dir,
    )

    # Evaluate on all target days
    results = {}
    for test_day in test_days:
        metrics = evaluate_on_day(
            model=model,
            test_day=test_day,
            normalize=normalize,
            batch_size=batch_size,
            output_dir=train_output_path,
            train_day=train_day,
            model_type=model_type,
        )
        results[test_day] = metrics

    # Save aggregate results
    aggregate = {
        "train_day": train_day,
        "model_type": model_type,
        "test_results": results,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "normalize": normalize,
            "val_ratio": val_ratio,
            "patience": patience,
            "seed": seed,
        },
    }

    aggregate_file = train_output_path / "cross_day_results.json"
    with open(aggregate_file, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n{'='*60}")
    print("Cross-Day Robustness Summary")
    print(f"{'='*60}")
    print(f"Trained on: day{train_day}")
    print(f"Model type: {model_type}\n")
    for test_day, metrics in results.items():
        acc = metrics["test_accuracy"]
        print(f"  Test day{test_day}: {acc:.4f} accuracy")
    print(f"\nAggregate results saved to: {aggregate_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cross-day robustness: train on one day, test on others."
    )
    parser.add_argument(
        "--train_day",
        type=int,
        required=True,
        choices=[0, 1, 2],
        help="Day to train on (0, 1, or 2).",
    )
    parser.add_argument(
        "--test_days",
        type=int,
        nargs="+",
        default=None,
        choices=[0, 1, 2],
        help="Days to test on. If not specified, tests on all days.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "improved"],
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate.")
    parser.add_argument(
        "--normalize",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "log1p"],
        help="Normalization strategy.",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Validation split ratio."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cross_day",
        help="Output directory for results.",
    )

    args = parser.parse_args()

    cross_day_experiment(
        train_day=args.train_day,
        test_days=args.test_days,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        normalize=args.normalize,
        val_ratio=args.val_ratio,
        patience=args.patience,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
