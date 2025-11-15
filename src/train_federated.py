"""
Federated training script for mmWave MIMO position classification.

Usage:
    python src/train_federated.py --day 0 --model baseline --rounds 20 --local_epochs 1
"""

from __future__ import annotations

import argparse
import os
import json
from datetime import datetime

import numpy as np

from data_loading import load_day_test, make_tf_dataset
from models import set_seeds, build_baseline_cnn, build_improved_cnn
from federated import train_federated
from evaluation import generate_full_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN with Federated Learning (FedAvg)")
    parser.add_argument("--day", type=int, default=0, choices=[0, 1, 2], help="Training day")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "improved"], help="Model architecture")
    parser.add_argument("--rounds", type=int, default=20, help="Number of FL rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per client per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--normalize", type=str, default="zscore", choices=["none", "zscore", "minmax", "log1p"], help="Normalization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Verbosity level")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    
    # Setup paths
    day_dir = os.path.join("data", f"day{args.day}")
    run_name = f"federated_{args.model}_day{args.day}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Model builder
    def model_builder():
        if args.model == "baseline":
            return build_baseline_cnn(dropout_rate=args.dropout)
        else:
            return build_improved_cnn(dropout_rate=args.dropout)
    
    # Train federated
    print(f"\n{'='*60}")
    print(f"Federated Learning: {args.model} on day{args.day}")
    print(f"{'='*60}\n")
    
    final_model, history = train_federated(
        day_dir=day_dir,
        model_builder=model_builder,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        normalize=args.normalize,
        output_dir=output_dir,
        run_name=run_name,
        seed=args.seed,
        verbose=args.verbose,
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    
    X_test, y_test = load_day_test(day_dir)
    test_ds = make_tf_dataset(X_test, y_test, batch_size=args.batch_size, shuffle=False, normalize=args.normalize)
    
    test_loss, test_acc = final_model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predictions and evaluation report
    y_pred_probs = final_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\nGenerating evaluation report...")
    # Convert FL history to match centralized format for plotting
    plot_history = {
        "loss": history["avg_train_loss"],
        "accuracy": history["avg_train_accuracy"],
        "val_loss": history["test_loss"],
        "val_accuracy": history["test_accuracy"],
    }
    
    generate_full_report(
        y_true=y_test,
        y_pred=y_pred,
        output_dir=output_dir,
        history=plot_history,
        run_name=run_name,
    )
    
    # Save final metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "num_rounds": args.rounds,
        "num_clients": len(history.get("round", [])),
    }
    
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Summary
    print("\n" + "="*60)
    print(f"Federated Training Complete: {run_name}")
    print(f"Rounds: {args.rounds}, Local Epochs: {args.local_epochs}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Results: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
