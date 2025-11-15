"""
Cross-day federated robustness: train FL on one day, test on all days.

This script runs FedAvg on a source day and evaluates the resulting global
model on test sets from all days (0, 1, 2), generating full reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data_loading import load_day_test, make_tf_dataset
from models import set_seeds, build_baseline_cnn, build_improved_cnn
from federated import train_federated
from evaluation import generate_full_report


def parse_args():
    p = argparse.ArgumentParser(description="Cross-day evaluation for Federated Learning (FedAvg)")
    p.add_argument("--train_day", type=int, choices=[0, 1, 2], required=True, help="Day to train FL on")
    p.add_argument("--test_days", type=int, nargs="+", default=None, choices=[0, 1, 2], help="Days to test on; default: all")
    p.add_argument("--model", type=str, choices=["baseline", "improved"], default="baseline", help="Model architecture")
    p.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    p.add_argument("--local_epochs", type=int, default=1, help="Local epochs per client per round")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    p.add_argument("--normalize", type=str, choices=["none", "zscore", "minmax", "log1p"], default="zscore", help="Normalization")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output_dir", type=str, default="outputs/cross_day_fed", help="Base output directory")
    return p.parse_args()


essential_keys = [
    "train_day", "model", "rounds", "local_epochs", "batch_size", "lr", "dropout", "normalize", "seed",
]


def build_model_factory(model: str, dropout: float):
    def _builder():
        if model == "baseline":
            return build_baseline_cnn(dropout_rate=dropout)
        else:
            return build_improved_cnn(dropout_rate=dropout)
    return _builder


def evaluate_model_on_day(model, test_day: int, batch_size: int, normalize: str, report_dir: Path, run_name: str) -> Dict:
    # Resolve day path relative to repo
    data_root = Path(__file__).parent.parent / "data"
    day_dir = data_root / f"day{test_day}"

    X_test, y_test = load_day_test(str(day_dir))
    test_ds = make_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False, normalize=normalize)

    loss, acc = model.evaluate(test_ds, verbose=0)
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_dir.mkdir(parents=True, exist_ok=True)
    generate_full_report(
        y_true=y_test,
        y_pred=y_pred,
        output_dir=str(report_dir),
        history=None,
        run_name=run_name,
    )

    metrics = {
        "test_day": test_day,
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "num_samples": int(len(y_test)),
    }
    return metrics


def main():
    args = parse_args()
    set_seeds(args.seed)

    # Resolve train day path
    data_root = Path(__file__).parent.parent / "data"
    train_day_dir = str(data_root / f"day{args.train_day}")

    # Prepare output folder
    base_out = Path(args.output_dir)
    exp_dir = base_out / f"train_day{args.train_day}_{args.model}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    cfg = {
        "train_day": args.train_day,
        "model": args.model,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "normalize": args.normalize,
        "seed": args.seed,
        "output_dir": str(exp_dir),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Train FL
    model_builder = build_model_factory(args.model, args.dropout)
    run_name = f"fed_train_day{args.train_day}_{args.model}"

    final_model, fl_history = train_federated(
        day_dir=train_day_dir,
        model_builder=model_builder,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        normalize=args.normalize,
        output_dir=str(exp_dir),
        run_name=run_name,
        seed=args.seed,
        verbose=1,
    )

    # Determine test days
    test_days: List[int] = args.test_days if args.test_days is not None else [0, 1, 2]

    # Evaluate on all test days
    results: Dict[int, Dict] = {}
    for td in test_days:
        report_dir = exp_dir / f"test_day{td}_report"
        rn = f"FL test_day{td} (Train day{args.train_day}, {args.model})"
        metrics = evaluate_model_on_day(
            model=final_model,
            test_day=td,
            batch_size=args.batch_size,
            normalize=args.normalize,
            report_dir=report_dir,
            run_name=rn,
        )
        # Also save raw metrics JSON at exp root
        with open(exp_dir / f"test_day{td}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        results[td] = metrics

    # Save aggregate cross-day results
    aggregate = {
        "train_day": args.train_day,
        "model": args.model,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "normalize": args.normalize,
        "results": results,
    }
    with open(exp_dir / "cross_day_results.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Federated Cross-Day Robustness Summary")
    print("=" * 60)
    print(f"Trained on: day{args.train_day} | Model: {args.model}")
    for td, m in results.items():
        print(f"  Test day{td}: {m['test_accuracy']:.4f} accuracy")
    print(f"\nAggregate results saved to: {exp_dir / 'cross_day_results.json'}")


if __name__ == "__main__":
    main()
