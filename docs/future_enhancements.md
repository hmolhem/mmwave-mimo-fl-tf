# Future Enhancements

Saved potential future work items and suggestions for extending the project.

## Training & Data
- Multi-day combined training (merge day0+day1+day2 centralized baseline).
- Domain augmentation (range/azimuth jitter, noise injection, mixup/cutmix).
- Curriculum or progressive domain adaptation (sequential day training).
- Client sampling strategies (subset of devices per FL round).

## Modeling & Optimization
- Mixed precision (TensorFlow `tf.keras.mixed_precision.set_global_policy('mixed_float16')`).
- XLA compilation (`TF_XLA_FLAGS=--tf_xla_auto_jit=2`).
- Hyperparameter sweeps (learning rate, rounds, local epochs) with a simple launcher.
- Add lightweight attention or squeeze-excite blocks to improved CNN.

## Evaluation & Metrics
- Calibration metrics (ECE / reliability diagrams).
- Latency & throughput profiling (inference time per sample on CPU/GPU).
- Safety confusion tracking across epochs/rounds (trend analysis of near→empty risk).

## Automation & Packaging
- Convert `src/` to package with `pyproject.toml` (`pip install -e .`).
- CLI entry points (`console_scripts`) for main workflows.
- Dockerfile for reproducible environment.

## Visualization
- Interactive dashboard (e.g., Streamlit) for exploring cross-day & safety metrics.
- Animated training curves comparing centralized vs FL.

## Federated Extensions
- Differential privacy (DP-SGD or noise on updates).
- Secure aggregation simulation (masking client updates).
- Personalized models (fine-tune global per client).

## Data Quality
- Automatic anomaly detection (outlier radar frames removal).
- Class rebalancing or focal loss for imbalanced bins.

## Documentation & Reproducibility
- Full reproduction bundle with fixed commit hash manifest.
- Expanded EDA notebook (range/azimuth intensity distributions per class/day).

## Research Directions
- Cross-day domain adaptation using adversarial feature alignment.
- Semi-supervised learning on partially labeled client data.
- Few-shot adaptation when introducing a new day/device.

---
Feel free to open issues or PRs referencing items here to gradually extend capabilities.
