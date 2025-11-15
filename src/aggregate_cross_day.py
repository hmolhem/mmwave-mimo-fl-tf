"""
Aggregate cross-day robustness results from multiple experiments.

This script collects results from multiple cross-day experiments and generates
comparison tables and visualizations to analyze domain shift patterns.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_cross_day_results(output_dir):
    """
    Load all cross-day experiment results from a directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing cross-day experiment subdirectories.

    Returns
    -------
    results : list of dict
        List of experiment result dictionaries.
    """
    output_dir = Path(output_dir)
    results = []

    # Find all cross_day_results.json files
    for results_file in output_dir.rglob("cross_day_results.json"):
        with open(results_file, "r") as f:
            data = json.load(f)
            results.append(data)

    return results


def create_accuracy_matrix(results):
    """
    Create a matrix of test accuracies: rows=train_day, cols=test_day.

    Parameters
    ----------
    results : list of dict
        List of cross-day experiment results.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with train days as rows, test days as columns.
    """
    # Organize by (train_day, model_type)
    data = {}
    for result in results:
        train_day = result["train_day"]
        model_type = result["model_type"]
        key = (train_day, model_type)

        if key not in data:
            data[key] = {}

        for test_day, metrics in result["test_results"].items():
            test_day = int(test_day)
            data[key][test_day] = metrics["test_accuracy"]

    # Create DataFrames for each model type
    dfs = {}
    for (train_day, model_type), test_results in data.items():
        if model_type not in dfs:
            dfs[model_type] = pd.DataFrame(index=[0, 1, 2], columns=[0, 1, 2])

        for test_day, acc in test_results.items():
            dfs[model_type].loc[train_day, test_day] = acc

    return dfs


def plot_accuracy_heatmap(df, model_type, output_path):
    """
    Plot heatmap of cross-day accuracies.

    Parameters
    ----------
    df : pd.DataFrame
        Accuracy matrix (train_day x test_day).
    model_type : str
        Model type for title.
    output_path : Path
        Path to save plot.
    """
    plt.figure(figsize=(8, 6))

    # Convert to float
    df_float = df.astype(float)

    # Plot heatmap
    sns.heatmap(
        df_float,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Test Accuracy"},
        linewidths=0.5,
        linecolor="gray",
    )

    plt.title(f"Cross-Day Robustness: {model_type.capitalize()} Model", fontsize=14, pad=15)
    plt.xlabel("Test Day", fontsize=12)
    plt.ylabel("Train Day", fontsize=12)
    plt.xticks([0.5, 1.5, 2.5], ["Day 0", "Day 1", "Day 2"])
    plt.yticks([0.5, 1.5, 2.5], ["Day 0", "Day 1", "Day 2"], rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved heatmap: {output_path}")


def compute_domain_shift_metrics(df):
    """
    Compute metrics characterizing domain shift.

    Parameters
    ----------
    df : pd.DataFrame
        Accuracy matrix (train_day x test_day).

    Returns
    -------
    metrics : dict
        Domain shift metrics.
    """
    df_float = df.astype(float)

    # In-domain accuracy (diagonal)
    in_domain = np.diag(df_float.values)

    # Out-of-domain accuracy (off-diagonal)
    mask = ~np.eye(3, dtype=bool)
    out_of_domain = df_float.values[mask]

    metrics = {
        "mean_in_domain_acc": float(np.mean(in_domain)),
        "std_in_domain_acc": float(np.std(in_domain)),
        "mean_out_of_domain_acc": float(np.mean(out_of_domain)),
        "std_out_of_domain_acc": float(np.std(out_of_domain)),
        "domain_shift_gap": float(np.mean(in_domain) - np.mean(out_of_domain)),
    }

    return metrics


def generate_summary_report(results, output_dir):
    """
    Generate comprehensive cross-day robustness report.

    Parameters
    ----------
    results : list of dict
        List of cross-day experiment results.
    output_dir : str or Path
        Directory to save report and plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Generating Cross-Day Robustness Summary")
    print(f"{'='*60}\n")

    # Create accuracy matrices
    dfs = create_accuracy_matrix(results)

    # Generate heatmaps and compute metrics for each model type
    all_metrics = {}
    for model_type, df in dfs.items():
        print(f"\n{model_type.capitalize()} Model:")
        print(df.to_string())

        # Plot heatmap
        heatmap_path = output_dir / f"cross_day_heatmap_{model_type}.png"
        plot_accuracy_heatmap(df, model_type, heatmap_path)

        # Compute domain shift metrics
        metrics = compute_domain_shift_metrics(df)
        all_metrics[model_type] = metrics

        print(f"\nDomain Shift Metrics ({model_type}):")
        print(f"  In-domain accuracy:     {metrics['mean_in_domain_acc']:.4f} ± {metrics['std_in_domain_acc']:.4f}")
        print(f"  Out-of-domain accuracy: {metrics['mean_out_of_domain_acc']:.4f} ± {metrics['std_out_of_domain_acc']:.4f}")
        print(f"  Domain shift gap:       {metrics['domain_shift_gap']:.4f}")

        # Save accuracy matrix
        csv_path = output_dir / f"cross_day_accuracy_{model_type}.csv"
        df.to_csv(csv_path)
        print(f"Saved CSV: {csv_path}")

    # Save aggregate metrics
    summary = {
        "num_experiments": len(results),
        "metrics_by_model": all_metrics,
    }

    summary_file = output_dir / "cross_day_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")
    print(f"All outputs in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize cross-day robustness results."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs/cross_day",
        help="Directory containing cross-day experiment results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cross_day_summary",
        help="Directory to save aggregated results and plots.",
    )

    args = parser.parse_args()

    # Load results
    results = load_cross_day_results(args.input_dir)

    if not results:
        print(f"No cross-day results found in: {args.input_dir}")
        print("Run cross_day_robustness.py first to generate experiment results.")
        return

    print(f"Found {len(results)} cross-day experiment(s)")

    # Generate summary report
    generate_summary_report(results, args.output_dir)


if __name__ == "__main__":
    main()
