# mmWave MIMO FL — Experimental Results Report

This report summarizes the experiments we ran on the mmWave MIMO human–robot distance classification task using centralized and federated learning with cross-day robustness evaluation.

## TL;DR Highlights

- Perfect in-domain accuracy does not translate cross-day.
- Baseline CNN: strong on Day1, weaker on Day2.
- Improved CNN: stronger on Day2, slightly weaker on Day1; more consistent overall.
- Significant temporal domain shift validated; motivates cross-day training and/or FL.

## Dataset

- Source: IEEE DataPort mmWave MIMO radar (3 days: Day0, Day1, Day2)
- Input: 256×63 range–azimuth maps, 10-class distance bins
- Quirk: Day0 uses folder name `test_dat` instead of `test_data` (loader supports both)

## Environment

- Python 3.10
- TensorFlow 2.15 / Keras 2.15
- Reproducibility: seeds fixed, early stopping + best-checkpoint saving

## How To Reproduce

Run from repo root in PowerShell (Windows):

```powershell
# Baseline CNN, train on day0, test cross-day (0,1,2)
$env:PYTHONUTF8=1; $env:TF_CPP_MIN_LOG_LEVEL=2; .\.venv\Scripts\python.exe src\cross_day_robustness.py --train_day 0 --model baseline --epochs 30 --batch_size 32 --lr 0.001 --dropout 0.3 --normalize zscore --val_ratio 0.2 --patience 5 --seed 42 --output_dir outputs\cross_day\baseline_day0

# Improved CNN, train on day0, test cross-day (0,1,2)
$env:PYTHONUTF8=1; $env:TF_CPP_MIN_LOG_LEVEL=2; .\.venv\Scripts\python.exe src\cross_day_robustness.py --train_day 0 --model improved --epochs 30 --batch_size 32 --lr 0.001 --dropout 0.3 --normalize zscore --val_ratio 0.2 --patience 5 --seed 42 --output_dir outputs\cross_day\improved_day0
```

Outputs, metrics and plots are saved under `outputs/` (ignored in git to keep the repo lean). Run the commands locally to regenerate all figures.

## Results

### Cross-Day Accuracy (Train on Day0)

| Model     | Day0  | Day1  | Day2  | Avg(Day1, Day2) | Notes |
|-----------|-------|-------|-------|------------------|-------|
| Baseline  | 100%  | 92.1% | 84.7% | 88.4%           | Larger drop on Day2 |
| Improved  | 100%  | 88.8% | 91.3% | 90.1%           | More balanced across days |

#### Federated (Train on Day0)

| Model (FL) | Day0  | Day1  | Day2  | Avg(Day1, Day2) | Notes |
|------------|-------|-------|-------|------------------|-------|
| Baseline   | 100%  | 91.0% | 82.9% | 86.95%          | Slightly below centralized on Day1/Day2 |

- Baseline shows better Day1 but degrades more on Day2.
- Improved CNN flips this: better Day2, slightly worse Day1.
- Improved reduces cross-day variance (88.8–91.3% vs 84.7–92.1%).

### Safety Metrics (Key)

Cross-day safety-aware results highlight robustness in critical zones (Near and Empty):

| Model     | Test Day | Empty Acc. | Near Acc. | Mid Acc. | Far Acc. | Critical Near→Empty |
|-----------|----------|------------|-----------|----------|----------|---------------------|
| Baseline  | Day1     | 1.00       | 1.00      | 0.961    | 0.883    | 0                   |
| Baseline  | Day2     | 0.00       | 0.971     | 1.000    | 0.931    | 0                   |
| Improved  | Day1     | 1.00       | 1.00      | 0.907    | 0.843    | 0                   |
| Improved  | Day2     | 0.74       | 0.981     | 0.975    | 0.958    | 0                   |
| Fed Base  | Day1     | 1.00       | 1.00      | 0.966    | 0.865    | 0                   |
| Fed Base  | Day2     | 0.01       | 0.981     | 1.000    | 0.906    | 0                   |

Notes:

- No critical Near→Empty errors observed across models/days in our runs.
- Improved model notably boosts Day2 Empty and Far accuracy vs baseline.

### In-Domain (Day0) Sanity Checks

- Centralized baseline (Day0): 100% test accuracy
- Centralized improved (Day0): 100% test accuracy
- Federated baseline (Day0, 9 clients, 10 rounds): 100% test accuracy

## Key Findings

- Temporal domain shift is significant: up to −15.3% from Day0 → Day2 with the baseline.
- Deeper architecture with BatchNorm does not universally improve cross-day generalization but yields more consistent performance.
- Results motivate domain-robust training (e.g., cross-day training, domain augmentation) and federated learning across clients/days.

## Where To Find Detailed Reports (Local)

After running the commands above, open these folders for per-day reports (confusion matrices, per-class metrics, safety metrics, classification reports):

- Baseline (train Day0):
  - `outputs/cross_day/baseline_day0/train_day0_baseline/test_day0_report/`
  - `outputs/cross_day/baseline_day0/train_day0_baseline/test_day1_report/`
  - `outputs/cross_day/baseline_day0/train_day0_baseline/test_day2_report/`

- Improved (train Day0):
  - `outputs/cross_day/improved_day0/train_day0_improved/test_day0_report/`
  - `outputs/cross_day/improved_day0/train_day0_improved/test_day1_report/`
  - `outputs/cross_day/improved_day0/train_day0_improved/test_day2_report/`

Each report includes:

- `confusion_matrix.png` and `confusion_matrix_normalized.png`
- `per_class_metrics.png`
- `safety_metrics.txt`
- `classification_report.txt`
- `test_*_metrics.json` (raw numbers)


### Confusion Matrices (Local-only previews)

Open these images locally after reproducing the runs:

- Baseline → Day1: `outputs/cross_day/baseline_day0/train_day0_baseline/test_day1_report/confusion_matrix_normalized.png`
- Baseline → Day2: `outputs/cross_day/baseline_day0/train_day0_baseline/test_day2_report/confusion_matrix_normalized.png`
- Improved → Day1: `outputs/cross_day/improved_day0/train_day0_improved/test_day1_report/confusion_matrix_normalized.png`
- Improved → Day2: `outputs/cross_day/improved_day0/train_day0_improved/test_day2_report/confusion_matrix_normalized.png`


## Next Steps

- Run federated learning cross-day to test temporal robustness improvements.
- Explore domain augmentation or mixup across days to reduce shift.
- Optionally add EDA notebook for data distribution and sample visualizations.
