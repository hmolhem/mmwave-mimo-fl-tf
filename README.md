# mmWave MIMO Federated Learning with TensorFlow

TensorFlow/Keras implementation of CNN and federated learning on the IEEE DataPort mmWave MIMO radar range–azimuth dataset for human–robot workspace monitoring. Includes centralized and FedAvg-style training, evaluation scripts, and comprehensive analysis tools.

## Features

- **Data Loading**: Utilities for loading and preprocessing .mat range-azimuth data from IEEE DataPort mmWave MIMO radar dataset
- **CNN Architecture**: Configurable CNN models for 10-class ROI classification
- **TensorFlow Data Pipelines**: Efficient tf.data pipelines with batching, shuffling, and data augmentation
- **Centralized Training**: Standard centralized learning approach
- **Federated Learning**: FedAvg (Federated Averaging) implementation with configurable clients
- **Comprehensive Evaluation**: Accuracy metrics, confusion matrices, and visualization tools

## Installation

```bash
# Clone the repository
git clone https://github.com/hmolhem/mmwave-mimo-fl-tf.git
cd mmwave-mimo-fl-tf

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the demo to see both centralized and federated learning in action:

```bash
python demo.py
```

This will:
1. Generate synthetic range-azimuth data (similar to mmWave MIMO radar data)
2. Train a CNN using centralized learning
3. Train the same architecture using federated learning (FedAvg)
4. Compare and visualize the results

## Project Structure

```
mmwave-mimo-fl-tf/
├── data_loader.py          # Data loading and preprocessing utilities
├── model.py                # CNN model architectures
├── data_pipeline.py        # TensorFlow data pipeline utilities
├── train_centralized.py    # Centralized training script
├── train_federated.py      # Federated learning (FedAvg) script
├── federated_learning.py   # FedAvg implementation
├── evaluation.py           # Evaluation and visualization tools
├── demo.py                 # Demo script for both approaches
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

### Centralized Training

```bash
python train_centralized.py \
    --num_samples 1000 \
    --num_classes 10 \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --model_type standard
```

### Federated Learning

```bash
python train_federated.py \
    --num_samples 1000 \
    --num_classes 10 \
    --num_rounds 50 \
    --num_clients 10 \
    --clients_per_round 5 \
    --local_epochs 5 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Using Real Data

To use real .mat files from the IEEE DataPort mmWave MIMO dataset:

```python
from data_loader import load_dataset_from_directory

# Load data from directory containing .mat files
X, y = load_dataset_from_directory(
    data_dir='path/to/mat/files',
    pattern='*.mat',
    num_classes=10,
    normalize=True
)
```

## Model Architecture

The project includes two CNN architectures:

1. **Standard CNN**: Deeper network with 4 convolutional blocks, batch normalization, and dropout
2. **Lightweight CNN**: Faster, more efficient model suitable for federated learning

Both models are designed for range-azimuth image classification with configurable input shapes and number of classes.

## Federated Learning (FedAvg)

The FedAvg implementation includes:
- Client sampling: Randomly select a subset of clients each round
- Local training: Each selected client trains on local data
- Weight aggregation: Average client models weighted by dataset size
- Support for IID and non-IID data distributions

### Key Parameters

- `num_clients`: Total number of federated clients
- `clients_per_round`: Number of clients selected for each round
- `local_epochs`: Training epochs per client per round
- `num_rounds`: Total number of federated rounds

## Evaluation

The evaluation module provides:
- Accuracy metrics (overall and per-class)
- Confusion matrices with visualization
- Training history plots
- Classification reports
- Model comparison tools

Example:

```python
from evaluation import evaluate_model, plot_confusion_matrix

results = evaluate_model(model, X_test, y_test)
plot_confusion_matrix(y_test, results['predictions'], 
                     num_classes=10, 
                     save_path='confusion_matrix.png')
```

## Data Pipeline

Efficient TensorFlow data pipelines with:
- Automatic batching and prefetching
- Data augmentation (flip, brightness, contrast)
- Shuffling with configurable buffer size
- Support for federated data partitioning

## Citation

If you use this code, please cite the IEEE DataPort mmWave MIMO radar dataset:

```
[Add appropriate citation for the IEEE DataPort mmWave MIMO dataset]
```

## License

[Add license information]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- IEEE DataPort mmWave MIMO radar dataset
- TensorFlow and Keras teams
- Federated learning research community
