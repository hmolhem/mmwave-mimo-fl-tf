# Requirements

## Functional Requirements
- Load and preprocess mmWave MIMO datasets locally (not stored in repo).
- Train models (centralized and, later, FL) with configurable hyperparameters.
- Evaluate on per-day test sets.

## Non-Functional Requirements
- Reproducible environment using a pinned Python version and requirements.
- Keep dataset files out of version control.
- Reasonable training time on CPU-only environments.

## Dataset Handling
- Place dataset under `data/` locally (ignored by Git).
- Provide a short `data/README.md` with instructions if sharing.

## Environment
- Python 3.10.x
- TensorFlow 2.15.x + Keras 2.15.x
- JupyterLab/Notebook for experiments
