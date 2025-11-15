# Reproducible Experiment Scripts

This directory contains PowerShell scripts to run all experiments from the proposal with pinned configurations.

## Scripts

### `run_centralized_experiments.ps1`

Trains baseline and improved CNN models on all 3 days using centralized learning.

**Usage:**

```powershell
.\scripts\run_centralized_experiments.ps1
```

**Optional parameters:**

- `-Seed <int>`: Random seed (default: 42)
- `-Epochs <int>`: Training epochs (default: 50)
- `-BatchSize <int>`: Batch size (default: 32)
- `-LearningRate <double>`: Initial learning rate (default: 0.001)
- `-Dropout <double>`: Dropout rate (default: 0.3)
- `-Normalize <string>`: Normalization method - zscore/minmax/log1p (default: zscore)
- `-ValRatio <double>`: Validation split ratio (default: 0.2)
- `-Patience <int>`: Early stopping patience (default: 10)

**Outputs:** `outputs/centralized/day{0,1,2}_{baseline,improved}/`

---

### `run_federated_experiments.ps1`

Runs federated learning (FedAvg) with 9 clients per day for baseline and improved models.

**Usage:**

```powershell
.\scripts\run_federated_experiments.ps1
```

**Optional parameters:**

- `-Seed <int>`: Random seed (default: 42)
- `-Rounds <int>`: Federated rounds (default: 20)
- `-LocalEpochs <int>`: Local epochs per round (default: 5)
- `-BatchSize <int>`: Batch size (default: 32)
- `-LearningRate <double>`: Learning rate (default: 0.001)
- `-Dropout <double>`: Dropout rate (default: 0.3)
- `-Normalize <string>`: Normalization method (default: zscore)
- `-Patience <int>`: Early stopping patience for local training (default: 5)

**Outputs:** `outputs/federated/day{0,1,2}_{baseline,improved}/`

---

### `run_cross_day_experiments.ps1`

Trains models on each day and tests on all days to measure domain shift robustness.

**Usage:**

```powershell
.\scripts\run_cross_day_experiments.ps1
```

**Optional parameters:** Same as centralized experiments.

**Outputs:**

- Individual experiments: `outputs/cross_day/train_day{0,1,2}_{baseline,improved}/`
- Aggregated analysis: `outputs/cross_day_summary/`

---

### `run_all_experiments.ps1`

Master script that runs all three experiment types sequentially.

**Usage:**

```powershell
.\scripts\run_all_experiments.ps1
```

**Optional parameters:**

- `-Seed <int>`: Random seed for all experiments (default: 42)
- `-SkipCentralized`: Skip centralized training
- `-SkipFederated`: Skip federated learning
- `-SkipCrossDay`: Skip cross-day robustness

**Example:**

```powershell
# Run only federated experiments
.\scripts\run_all_experiments.ps1 -SkipCentralized -SkipCrossDay

# Run with custom seed
.\scripts\run_all_experiments.ps1 -Seed 123
```

---

## Prerequisites

1. **Dataset placement:** Place `.mat` files in `data/day{0,1,2}/train_data/` and `data/day{0,1,2}/test_data/`
2. **Virtual environment:** Ensure `.venv` is created and dependencies installed:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

## Expected Runtime

- **Centralized:** ~30-60 min total (6 runs)
- **Federated:** ~45-90 min total (6 runs, 20 rounds each)
- **Cross-day:** ~30-60 min total (6 runs)
- **Total:** ~2-4 hours (hardware dependent)

## Output Structure

```text
outputs/
├── centralized/
│   ├── day0_baseline/
│   │   ├── best_model.keras
│   │   ├── config.json
│   │   ├── training_history.json
│   │   ├── test_metrics.json
│   │   └── evaluation_report/
│   ├── day0_improved/
│   └── ...
├── federated/
│   ├── day0_baseline/
│   │   ├── global_model.keras
│   │   ├── config.json
│   │   ├── federated_history.json
│   │   ├── test_metrics.json
│   │   └── evaluation_report/
│   └── ...
├── cross_day/
│   ├── train_day0_baseline/
│   │   ├── best_model.keras
│   │   ├── config.json
│   │   ├── cross_day_results.json
│   │   ├── test_day0_metrics.json
│   │   ├── test_day1_metrics.json
│   │   ├── test_day2_metrics.json
│   │   └── test_day{0,1,2}_report/
│   └── ...
└── cross_day_summary/
    ├── cross_day_summary.json
    ├── cross_day_heatmap_baseline.png
    ├── cross_day_heatmap_improved.png
    ├── cross_day_accuracy_baseline.csv
    └── cross_day_accuracy_improved.csv
```

## Notes

- All scripts use the same default hyperparameters for fair comparison
- Random seeds are pinned for reproducibility
- Scripts automatically activate the virtual environment
- Progress is logged to console with color-coded status messages
- Failed runs will abort with error messages and non-zero exit codes
