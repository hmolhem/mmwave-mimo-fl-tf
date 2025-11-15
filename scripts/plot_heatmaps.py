import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data matrices derived from docs/results.md
# Federated Baseline (Train→Test)
FL_BASELINE = np.array([
    [100.0, 91.0, 82.9],
    [94.93, 98.2, 94.5],
    [89.27, 95.1, 97.0],
])

# Federated Improved (Train→Test)
FL_IMPROVED = np.array([
    [99.9, 90.9, 94.0],
    [86.6, 98.8, 94.7],
    [89.6, 94.8, 96.6],
])

# Centralized day0 only (train on day0 → test day0/1/2)
CENTRALIZED_DAY0_BASELINE = [100.0, 92.1, 84.7]
CENTRALIZED_DAY0_IMPROVED = [100.0, 88.8, 91.3]

TRAIN_DAYS = ["Train Day0", "Train Day1", "Train Day2"]
TEST_DAYS = ["Test Day0", "Test Day1", "Test Day2"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fl_heatmap(matrix: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu", vmin=80, vmax=100,
                     xticklabels=TEST_DAYS, yticklabels=TRAIN_DAYS, cbar_kws={"label": "Accuracy (%)"})
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_centralized_bars(base_vals, impr_vals, out_path: str):
    x = np.arange(len(TEST_DAYS))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, base_vals, width, label="Baseline", color="#4C72B0")
    plt.bar(x + width/2, impr_vals, width, label="Improved", color="#55A868")
    plt.ylim(80, 101)
    plt.ylabel("Accuracy (%)")
    plt.title("Centralized (Train Day0) — Cross-Day Accuracy")
    plt.xticks(x, TEST_DAYS)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    out_dir = os.path.join("docs", "figures")
    ensure_dir(out_dir)

    save_fl_heatmap(FL_BASELINE, "Federated — Baseline CNN (Accuracy %)", os.path.join(out_dir, "fl_baseline_cross_day_heatmap.png"))
    save_fl_heatmap(FL_IMPROVED, "Federated — Improved CNN (Accuracy %)", os.path.join(out_dir, "fl_improved_cross_day_heatmap.png"))

    save_centralized_bars(CENTRALIZED_DAY0_BASELINE, CENTRALIZED_DAY0_IMPROVED, os.path.join(out_dir, "centralized_day0_bars.png"))

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
