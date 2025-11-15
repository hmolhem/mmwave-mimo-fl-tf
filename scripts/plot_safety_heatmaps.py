import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXPORTS = os.path.join(ROOT, 'exports')
FIG_DIR = os.path.join(ROOT, 'docs', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

METRICS_FILE = os.path.join(EXPORTS, 'metrics_flat.csv')

# Heatmaps: near_accuracy and empty_accuracy per train_day vs test_day for each (mode, model)

def load_metrics():
    if not os.path.isfile(METRICS_FILE):
        raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}. Run export_metrics.py first.")
    df = pd.read_csv(METRICS_FILE)
    return df


def pivot_metric(df: pd.DataFrame, mode: str, model: str, metric: str) -> pd.DataFrame:
    sub = df[(df['mode'] == mode) & (df['model'] == model)]
    if sub.empty:
        return pd.DataFrame()
    pivot = sub.pivot_table(values=metric, index='train_day', columns='test_day', aggfunc='mean')
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
    return pivot * 100.0  # to percentage


def save_heatmap(data: pd.DataFrame, title: str, out_name: str):
    if data.empty:
        return
    plt.figure(figsize=(5.2, 4.4))
    ax = sns.heatmap(data, annot=True, fmt='.1f', cmap='YlOrRd', vmin=0, vmax=100,
                     cbar_kws={'label': 'Accuracy (%)'})
    ax.set_xlabel('Test Day')
    ax.set_ylabel('Train Day')
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path}")


def main():
    df = load_metrics()
    for metric in ['near_accuracy', 'empty_accuracy']:
        for mode in ['centralized', 'federated']:
            for model in ['baseline', 'improved']:
                pivot = pivot_metric(df, mode, model, metric)
                if pivot.empty:
                    continue
                title = f"{mode.capitalize()} {model.capitalize()} {metric.replace('_',' ').title()}"
                out_name = f"{mode}_{model}_{metric}_heatmap.png"
                save_heatmap(pivot, title, out_name)

if __name__ == '__main__':
    main()
