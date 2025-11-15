# Usage Guide

Comprehensive reference for running each Python script in this repository on Windows PowerShell.

Activate the virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8=1
$env:TF_CPP_MIN_LOG_LEVEL=2
```

## Common Arguments

- `--model {baseline|improved}`: Select CNN architecture.
- `--normalize {none|zscore|minmax|log1p}`: Per-sample normalization strategy.
- `--dropout FLOAT`: Dropout rate in dense layers (and conv blocks for improved).
- `--seed INT`: Reproducibility seed.
- `--output_dir PATH`: Base output directory for artifacts.

---

## Centralized Training (`src/train_centralized.py`)

Train on all devices of a single day with validation split.

```powershell
python src/train_centralized.py `
  --day 0 `
  --model baseline `
  --epochs 50 `
  --batch_size 32 `
  --lr 0.001 `
  --dropout 0.3 `
  --val_ratio 0.2 `
  --normalize zscore `
  --patience 10 `
  --seed 42 `
  --output_dir outputs/centralized
```

Outputs: `config.json`, `history.json`, `test_metrics.json`, evaluation report (confusion matrices, per-class metrics, safety metrics).

### Centralized Key Args

- `--day`: {0,1,2}
- `--val_ratio`: Validation split fraction.
- `--patience`: Early stopping patience (val_loss).

---

## Federated Training (`src/train_federated.py`)

FedAvg across 9 device clients for a chosen day.

```powershell
python src/train_federated.py `
  --day 1 `
  --model improved `
  --rounds 20 `
  --local_epochs 2 `
  --batch_size 32 `
  --lr 0.001 `
  --dropout 0.3 `
  --normalize zscore `
  --seed 42 `
  --output_dir outputs/federated `
  --verbose 1
```

Outputs: `config.json`, `federated_history.json`, `global_model.keras`, `test_metrics.json`, evaluation report.

### Federated Key Args

- `--rounds`: FL global aggregation iterations.
- `--local_epochs`: Per-client epochs each round.
- `--verbose`: 0 (silent), 1 (progress), 2 (detailed per-client).

---

## Cross-Day Robustness (Centralized) (`src/cross_day_robustness.py`)

Train on one day; evaluate on specified or all days.

```powershell
python src/cross_day_robustness.py `
  --train_day 0 `
  --model baseline `
  --epochs 30 `
  --batch_size 32 `
  --lr 0.001 `
  --dropout 0.3 `
  --normalize zscore `
  --val_ratio 0.2 `
  --patience 5 `
  --seed 42 `
  --output_dir outputs/cross_day
```

Restrict test targets:

```powershell
python src/cross_day_robustness.py --train_day 1 --test_days 1 2 --model improved
```

Outputs: `train_dayX_model/` with per test-day metrics + reports and aggregate `cross_day_results.json`.

### Cross-Day Key Args

- `--test_days`: Space-separated list (e.g. `--test_days 0 2`). Default all.

---

## Cross-Day Federated (`src/cross_day_federated.py`)

Run FedAvg on one day; evaluate global model on all/selected days.

```powershell
python src/cross_day_federated.py `
  --train_day 2 `
  --model improved `
  --rounds 10 `
  --local_epochs 1 `
  --batch_size 32 `
  --lr 0.001 `
  --dropout 0.3 `
  --normalize zscore `
  --seed 42 `
  --output_dir outputs/cross_day_fed
```

Limit test days:

```powershell
python src/cross_day_federated.py --train_day 0 --test_days 0 2 --model baseline
```

Outputs: `train_dayX_model/` with `cross_day_results.json` and per-day evaluation folders.

### Cross-Day Federated Key Args

- Mirrors centralized cross-day plus FL-specific: `--rounds`, `--local_epochs`.

---

## Aggregate Cross-Day Summary (`src/aggregate_cross_day.py`)

Collect multiple centralized cross-day experiments and produce heatmaps & summary metrics.

```powershell
python src/aggregate_cross_day.py `
  --input_dir outputs/cross_day `
  --output_dir outputs/cross_day_summary
```

Outputs: Heatmaps, CSV matrices, `cross_day_summary.json`.

### Notes

- Expects subfolders each with `cross_day_results.json` (produced by `cross_day_robustness.py`).

---

## Metrics Export (`scripts/export_metrics.py`)

Aggregate centralized + federated cross-day runs into flat CSV/JSON plus Excel workbook.

```powershell
python scripts/export_metrics.py
```

Outputs (in `exports/`):

- `metrics_flat.csv` / `.json` (includes per-class F1 & safety metrics)
- `*_accuracy_matrix.csv` (centralized & federated baseline/improved)
- `metrics.xlsx` (sheets: flat, pivots, safety_summary)

---

## Plot Federated & Centralized Heatmaps (`scripts/plot_heatmaps.py`)

Generates cross-day accuracy heatmaps and centralized bar chart from hard-coded matrices (use for quick visualization refresh).

```powershell
python scripts/plot_heatmaps.py
```

Outputs: PNG files in `docs/figures/`.

---

## Safety Accuracy Heatmaps (`scripts/plot_safety_heatmaps.py`)

Visualize Near / Empty accuracy across train/test days using exported metrics.

```powershell
python scripts/plot_safety_heatmaps.py
```

Prerequisite: Run `export_metrics.py` first.

Outputs: `*_near_accuracy_heatmap.png`, `*_empty_accuracy_heatmap.png` in `docs/figures/`.

---

## Supporting Modules (Non-CLI)

- `src/data_loading.py`: Dataset loaders, TF dataset builders; referenced internally.
- `src/models.py`: Model builders & compilation helpers.
- `src/federated.py`: FedAvg server logic (`train_federated`).
- `src/evaluation.py`: `generate_full_report` (plots, confusion matrices, safety metrics).

---

## Recommended Workflows

### 1. Centralized Baseline + Cross-Day

```powershell
python src/train_centralized.py --day 0 --model baseline --epochs 30
python src/cross_day_robustness.py --train_day 0 --model baseline --epochs 30
```

### 2. Federated Improved + Cross-Day Evaluation

```powershell
python src/train_federated.py --day 0 --model improved --rounds 15 --local_epochs 1
python src/cross_day_federated.py --train_day 0 --model improved --rounds 10 --local_epochs 1
```

### 3. Aggregate & Export

```powershell
python src/aggregate_cross_day.py --input_dir outputs/cross_day --output_dir outputs/cross_day_summary
python scripts/export_metrics.py
python scripts/plot_heatmaps.py
python scripts/plot_safety_heatmaps.py
```

---

## Tips & Troubleshooting

- Ensure dataset placed under `data/day{0,1,2}/train_data` and `test_data` (day0 also accepts `test_dat`).
- If GPU memory issues occur, reduce `--batch_size` (e.g., 16) or enable mixed precision (future enhancement).
- Re-run `export_metrics.py` after new experiments to refresh Excel and heatmaps.
- Safety metrics rely on `evaluation.generate_full_report`; always generate reports before exporting.

---

## Future Enhancements (See `future_enhancements.md`)

- Multi-day combined training baseline.
- Domain augmentation / mixup across days.
- Mixed precision & XLA acceleration.
- Automated hyperparameter sweeps.
- Packaging as installable module (`pyproject.toml`).

---

## License

MIT. See `LICENSE`.

---

## Citation

If used academically, cite the IEEE DataPort dataset and FedAvg paper referenced in `README.md`.
