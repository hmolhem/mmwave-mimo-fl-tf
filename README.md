# mmWave MIMO Federated Learning with TensorFlow

TensorFlow/Keras implementation of CNN and federated learning on the IEEE DataPort mmWave MIMO radar range–azimuth dataset for human–robot workspace monitoring. Includes centralized and FedAvg-style training, evaluation scripts, and comprehensive metrics with confusion matrices.

## Features

- 📊 **Data Loading**: Load and process .mat range-azimuth radar data
- 🔄 **TF Data Pipelines**: Efficient TensorFlow data pipelines with batching and augmentation
- 🧠 **CNN Models**: Multiple CNN architectures (standard, deep, lightweight) for 10-class ROI classification
- 🏢 **Centralized Training**: Traditional centralized training with callbacks and checkpointing
- 🌐 **Federated Learning**: FedAvg (Federated Averaging) implementation with IID/non-IID data distribution
- 📈 **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrices
- 📉 **Visualization**: Training history plots and comparison charts

## Installation

```bash
# Clone the repository
git clone https://github.com/hmolhem/mmwave-mimo-fl-tf.git
cd mmwave-mimo-fl-tf

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
mmwave-mimo-fl-tf/
├── src/
│   ├── data/
│   │   ├── data_loader.py      # .mat file loading and preprocessing
│   │   └── tf_pipeline.py      # TensorFlow data pipelines
│   ├── models/
│   │   └── cnn_model.py        # CNN architectures
│   ├── training/
│   │   ├── centralized.py      # Centralized training
│   │   └── federated.py        # FedAvg federated learning
│   └── evaluation/
│       └── metrics.py          # Evaluation and visualization
├── examples/
│   ├── train_centralized.py    # Centralized training example
│   ├── train_federated.py      # Federated training example
│   └── compare_approaches.py   # Compare centralized vs federated
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Centralized Training

```bash
cd examples
python train_centralized.py
```

This script:
- Creates synthetic range-azimuth data (or loads your .mat files)
- Trains a CNN model using centralized learning
- Evaluates with accuracy and confusion matrix
- Saves model checkpoints and visualizations

### 2. Federated Learning (FedAvg)

```bash
cd examples
python train_federated.py
```

This script:
- Distributes data across multiple clients (IID or non-IID)
- Trains using Federated Averaging algorithm
- Performs federated rounds with local client updates
- Evaluates the global model

### 3. Compare Centralized vs Federated

```bash
cd examples
python compare_approaches.py
```

This script trains both approaches and generates comparison metrics.

## Usage Examples

### Loading Your Own Data

```python
from src.data import load_dataset, normalize_data, split_dataset

# Load .mat files from directory
data, labels, file_paths = load_dataset(
    data_dir='path/to/your/data',
    num_classes=10
)

# Normalize
data = normalize_data(data, method='minmax')

# Split into train/val/test
train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset(
    data, labels,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### Building and Training a Model

```python
from src.models import build_cnn_model, compile_model
from src.training import train_centralized
from src.data.tf_pipeline import prepare_datasets

# Create TF datasets
train_ds, val_ds, test_ds = prepare_datasets(
    train_data, train_labels,
    val_data, val_labels,
    test_data, test_labels,
    batch_size=32
)

# Build model
model = build_cnn_model(
    input_shape=(64, 64, 1),
    num_classes=10,
    architecture='standard'
)
model = compile_model(model, learning_rate=0.001)

# Train
trained_model, metrics = train_centralized(
    train_dataset=train_ds,
    val_dataset=val_ds,
    test_dataset=test_ds,
    model=model,
    epochs=50
)
```

### Federated Learning

```python
from src.data.tf_pipeline import create_federated_datasets
from src.training import train_federated

# Create federated datasets for 5 clients
client_datasets = create_federated_datasets(
    train_data,
    train_labels,
    num_clients=5,
    batch_size=32,
    iid=True  # or False for non-IID
)

# Train with FedAvg
global_model, metrics = train_federated(
    client_datasets=client_datasets,
    val_dataset=val_ds,
    test_dataset=test_ds,
    model=model,
    num_rounds=50,
    client_epochs=1
)
```

### Evaluation

```python
from src.evaluation import evaluate_model, ModelEvaluator

# Simple evaluation
metrics = evaluate_model(
    model=trained_model,
    test_dataset=test_ds,
    class_names=['Class_0', 'Class_1', ...],
    save_dir='results'
)

# Advanced evaluation with visualizations
evaluator = ModelEvaluator(trained_model, save_dir='results')
metrics = evaluator.evaluate_dataset(test_ds, class_names)
# Generates: confusion matrix, classification report, metrics JSON
```

## Model Architectures

Three CNN architectures are available:

1. **Standard CNN**: Balanced performance and speed
   - 3 conv blocks (32, 64, 128 filters)
   - Batch normalization and dropout
   - Dense layers (256, 128)

2. **Deep CNN**: Enhanced feature extraction
   - 4 conv blocks (32, 64, 128, 256 filters)
   - More capacity for complex patterns
   - Dense layers (512, 256)

3. **Lightweight CNN**: Fast training for FL
   - 3 conv blocks (16, 32, 64 filters)
   - Reduced parameters for faster federated rounds
   - Dense layer (128)

## Configuration Options

### Data Loading
- `.mat` file support for mmWave MIMO radar data
- Synthetic data generation for testing
- Normalization (min-max, standardization)
- Train/val/test splitting

### TF Data Pipeline
- Batching and prefetching
- Data augmentation (flip, brightness, contrast, noise)
- Federated data partitioning (IID/non-IID)

### Training
- Centralized: Early stopping, learning rate reduction, checkpointing
- Federated: Client sampling, local epochs, weighted averaging

### Evaluation
- Accuracy, precision, recall, F1-score
- Per-class metrics
- Confusion matrices (normalized and unnormalized)
- Training history visualization

## Results

The implementation generates:

1. **Model Checkpoints**: Best and final models saved as .h5 files
2. **Training Logs**: CSV and JSON format with epoch/round metrics
3. **Evaluation Reports**: Comprehensive JSON with all metrics
4. **Visualizations**:
   - Training/validation accuracy and loss curves
   - Confusion matrices (heatmaps)
   - Model comparison plots

## IEEE DataPort mmWave MIMO Dataset

This project is designed for the IEEE DataPort mmWave MIMO radar dataset. The expected data format:

```
data/
├── class_0/
│   ├── sample_001.mat
│   ├── sample_002.mat
│   └── ...
├── class_1/
│   ├── sample_001.mat
│   └── ...
...
└── class_9/
    └── ...
```

Each `.mat` file should contain range-azimuth maps (2D arrays).

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mmwave_mimo_fl_tf,
  title={mmWave MIMO Federated Learning with TensorFlow},
  author={Your Name},
  year={2024},
  url={https://github.com/hmolhem/mmwave-mimo-fl-tf}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
