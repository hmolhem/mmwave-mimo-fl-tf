# Changelog

All notable changes to this project will be documented in this file.

Versioning follows incremental tags (e.g., v0.1, v0.2). Each snapshot corresponds to a stable, reproducible state of experiment code and results.

## [v0.1-baseline-results] - 2025-11-15

Commit: e0c2f52 (tag annotated at e0c2f52)

### Added

- Initial aggregated metrics snapshot: `exports/metrics_flat_baseline.csv` pinned for reproducibility.
- Baseline Metrics Snapshot section in `docs/results.md` describing snapshot contents and regeneration workflow.
- Detailed usage guide (`docs/usage.md`) covering all CLI scripts, safety plotting, aggregation, and recommended workflows.
- Future enhancements roadmap (`docs/future_enhancements.md`).

### Experiments Included

- Centralized training (baseline & improved CNN) across Day0, Day1, Day2.
- Federated learning (FedAvg) with 9 clients per day; baseline & improved CNN.
- Cross-day robustness matrices (train_day × test_day) for centralized and federated setups.
- Safety metrics: Empty / Near / Mid / Far accuracies; critical Near→Empty and false alarm Empty→Near counts (all zeros for Near→Empty).
- Per-class F1 scores exported for all train/test day pairs.

### Documentation

- `README.md` updated with links to usage guide and future enhancements roadmap.
- Results report (`docs/results.md`) includes heatmaps, safety heatmaps, interpretation, and snapshot reference.

### Tooling

- `scripts/export_metrics.py` generates flat metrics CSV/JSON, accuracy matrices, Excel workbook with conditional formatting, and safety summary.
- Plot scripts for cross-day accuracy heatmaps and safety (Near/Empty) accuracy heatmaps.

### Project Status

- Implementation baseline complete (models, loaders, FL logic, evaluation).
- Safety evaluation integrated: No critical Near→Empty errors in recorded snapshot.

### Notes

- Auto-regenerated `exports/metrics_flat.csv` ignored via `.gitignore` to prevent noisy commits.
- Future snapshots should create new versioned files (e.g., `metrics_flat_v2.csv`) and corresponding tag (e.g., `v0.2-*`).

### Recommended Next Steps

- Introduce multi-day combined training baseline.
- Add domain augmentation / mixup for improved cross-day consistency.
- Add mixed precision (TensorFlow AMP) for performance.
- Implement hyperparameter sweep harness.
- Tag subsequent releases with concise notes (e.g., `v0.2-augmentation`).

---
