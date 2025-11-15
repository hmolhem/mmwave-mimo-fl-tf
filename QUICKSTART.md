# Quick Start Guide

This guide will help you get started with mmWave MIMO Federated Learning in TensorFlow.

## Installation

```bash
# Clone the repository
git clone https://github.com/hmolhem/mmwave-mimo-fl-tf.git
cd mmwave-mimo-fl-tf

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Demo

Run the demonstration script to see all features:

```bash
python demo.py
```

This will run three demos:
1. Centralized training
2. Federated learning (FedAvg)
3. Comprehensive evaluation with confusion matrices

## Using Your Own Data

### 1. Organize Your Data

Structure your .mat files like this:

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

### 2. Load and Train

```python
from src.data import load_dataset, normalize_data, split_dataset
from src.data.tf_pipeline import prepare_datasets
from src.models import build_cnn_model, compile_model
from src.training import train_centralized

# Load your data
data, labels, file_paths = load_dataset('path/to/your/data', num_classes=10)
data = normalize_data(data, method='minmax')

# Split dataset
train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset(
    data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

# Create TF datasets
train_ds, val_ds, test_ds = prepare_datasets(
    train_data, train_labels,
    val_data, val_labels,
    test_data, test_labels,
    batch_size=32
)

# Build and compile model
model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10)
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

### 3. Evaluate

```python
from src.evaluation import evaluate_model

class_names = ['Class_0', 'Class_1', ..., 'Class_9']
metrics = evaluate_model(
    model=trained_model,
    test_dataset=test_ds,
    class_names=class_names,
    save_dir='results'
)
```

## Federated Learning

For federated learning with multiple clients:

```python
from src.data.tf_pipeline import create_federated_datasets
from src.training import train_federated

# Create federated datasets (5 clients)
client_datasets = create_federated_datasets(
    train_data,
    train_labels,
    num_clients=5,
    batch_size=32,
    iid=True  # or False for non-IID distribution
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

## Model Architectures

Three architectures are available:

```python
# Standard CNN (balanced)
model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10, architecture='standard')

# Deep CNN (more capacity)
model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10, architecture='deep')

# Lightweight CNN (faster for FL)
model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10, architecture='lightweight')
```

## Configuration Options

### Data Augmentation

```python
train_ds, val_ds, test_ds = prepare_datasets(
    train_data, train_labels,
    val_data, val_labels,
    test_data, test_labels,
    batch_size=32,
    augment_train=True  # Enable augmentation
)
```

### Learning Rate

```python
model = compile_model(model, learning_rate=0.001)  # Default
model = compile_model(model, learning_rate=0.0001)  # Lower LR
```

### Federated Settings

```python
# More clients, more rounds
global_model, metrics = train_federated(
    client_datasets=client_datasets,
    val_dataset=val_ds,
    test_dataset=test_ds,
    model=model,
    num_rounds=100,     # More rounds
    client_epochs=5     # More local epochs
)
```

## Next Steps

1. Check out the full examples in `examples/` directory
2. Read the complete documentation in `README.md`
3. Experiment with different hyperparameters
4. Try non-IID data distribution for federated learning

## Troubleshooting

### Out of Memory

- Reduce batch size: `batch_size=16` or `batch_size=8`
- Use lightweight model: `architecture='lightweight'`

### Slow Training

- Use GPU if available
- Reduce model complexity
- Decrease number of epochs/rounds

### Poor Accuracy

- Increase training epochs/rounds
- Try data augmentation
- Adjust learning rate
- Use deeper model architecture

## Support

For questions or issues, please open an issue on GitHub.
