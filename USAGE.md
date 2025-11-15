# Quick Start Guide

This guide will help you get started with the mmWave MIMO Federated Learning project.

## Installation

```bash
# Clone the repository
git clone https://github.com/hmolhem/mmwave-mimo-fl-tf.git
cd mmwave-mimo-fl-tf

# Install dependencies
pip install -r requirements.txt
```

## Quick Demo

Run the demo to see both centralized and federated learning:

```bash
python demo.py
```

This will:
1. Generate synthetic data
2. Train a CNN using centralized learning
3. Train using federated learning (FedAvg)
4. Compare and visualize results

Results are saved to `demo_results/` directory.

## Using Your Own Data

### Loading .mat Files

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

### Loading a Single .mat File

```python
from data_loader import load_mat_file, extract_range_azimuth_data

# Load single file
mat_data = load_mat_file('path/to/file.mat')
X, y = extract_range_azimuth_data(mat_data)
```

## Training Models

### Centralized Training

```bash
# Basic usage
python train_centralized.py --num_samples 1000 --epochs 50

# With custom parameters
python train_centralized.py \
    --num_samples 2000 \
    --num_classes 10 \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0005 \
    --model_type standard
```

### Federated Learning

```bash
# Basic usage
python train_federated.py --num_samples 1000 --num_rounds 50

# With custom parameters
python train_federated.py \
    --num_samples 2000 \
    --num_clients 20 \
    --clients_per_round 10 \
    --local_epochs 5 \
    --num_rounds 100 \
    --batch_size 32
```

## Using Configuration Files

```python
from config_utils import load_config, setup_directories

# Load configuration
config = load_config('config.json')

# Setup directories
setup_directories(config)

# Get training config
train_config = config['training']['centralized']
epochs = train_config['epochs']
batch_size = train_config['batch_size']
```

## Programmatic Usage

### Basic Training Example

```python
import numpy as np
import tensorflow as tf
from data_loader import create_synthetic_data, split_data
from model import create_cnn_model, compile_model
from evaluation import evaluate_model

# Generate data
X, y = create_synthetic_data(num_samples=1000, num_classes=10)

# Split data
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

# Create and compile model
model = create_cnn_model(input_shape=(128, 128, 1), num_classes=10)
model = compile_model(model, learning_rate=0.001)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

# Evaluate
results = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {results['accuracy']:.4f}")
```

### Federated Learning Example

```python
from federated_learning import FederatedAveraging
from model import create_lightweight_cnn
from data_pipeline import create_client_data_from_splits

# Split data among clients
client_X, client_y = create_client_data_from_splits(
    X_train, y_train, num_clients=10
)
client_datasets = list(zip(client_X, client_y))

# Create FedAvg trainer
fedavg = FederatedAveraging(
    model_fn=create_lightweight_cnn,
    num_classes=10,
    num_clients=10,
    clients_per_round=5,
    local_epochs=5
)

# Train
history = fedavg.train(
    client_datasets=client_datasets,
    num_rounds=50,
    X_val=X_val,
    y_val=y_val
)

# Get global model
global_model = fedavg.get_global_model()
```

## Visualization

```python
from visualization import (
    visualize_range_azimuth_sample,
    visualize_multiple_samples,
    plot_class_distribution,
    plot_federated_client_distribution
)

# Visualize a single sample
visualize_range_azimuth_sample(
    X[0], label=y[0], 
    save_path='sample.png'
)

# Visualize multiple samples
visualize_multiple_samples(
    X[:9], y[:9],
    save_path='samples_grid.png'
)

# Plot class distribution
plot_class_distribution(
    y, num_classes=10,
    save_path='class_dist.png'
)

# Plot federated data distribution
plot_federated_client_distribution(
    client_y, num_classes=10,
    save_path='federated_dist.png'
)
```

## Running Tests

```bash
# Run all tests
python test_suite.py
```

This will test:
- Data loading and preprocessing
- Model creation and compilation
- Data pipeline functionality
- Training processes
- Federated learning
- Evaluation utilities

## Command-line Arguments

### train_centralized.py

- `--num_samples`: Number of synthetic samples (default: 1000)
- `--num_classes`: Number of classes (default: 10)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--model_type`: Model type - 'standard' or 'lightweight' (default: standard)
- `--save_path`: Path to save checkpoints (default: checkpoints)

### train_federated.py

- `--num_samples`: Number of synthetic samples (default: 1000)
- `--num_classes`: Number of classes (default: 10)
- `--num_rounds`: Number of federated rounds (default: 50)
- `--num_clients`: Total number of clients (default: 10)
- `--clients_per_round`: Clients selected per round (default: 5)
- `--local_epochs`: Local training epochs per client (default: 5)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--save_path`: Path to save checkpoints (default: checkpoints)

## Project Structure

```
mmwave-mimo-fl-tf/
├── data_loader.py          # Data loading and preprocessing
├── model.py                # CNN model architectures
├── data_pipeline.py        # TensorFlow data pipelines
├── train_centralized.py    # Centralized training script
├── train_federated.py      # Federated training script
├── federated_learning.py   # FedAvg implementation
├── evaluation.py           # Evaluation and metrics
├── visualization.py        # Visualization tools
├── demo.py                 # Demo script
├── example.py              # Usage examples
├── test_suite.py           # Test suite
├── config_utils.py         # Configuration utilities
├── config.json             # Configuration file
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Tips

1. **Start small**: Test with fewer samples and epochs first
2. **Use GPU**: If available, TensorFlow will automatically use GPU
3. **Monitor training**: Check the saved plots in checkpoint directories
4. **Adjust hyperparameters**: Use config.json for easy parameter management
5. **Save checkpoints**: Models are automatically saved during training

## Troubleshooting

### Out of Memory

Reduce batch size or use the lightweight model:
```bash
python train_centralized.py --batch_size 16 --model_type lightweight
```

### Slow Training

- Reduce number of samples
- Use lightweight model
- Decrease image size in config.json

### Poor Accuracy

- Increase training epochs/rounds
- Adjust learning rate
- Try data augmentation
- Use the standard (deeper) model

## Next Steps

1. **Load your own data**: Replace synthetic data with real .mat files
2. **Customize the model**: Modify architectures in model.py
3. **Tune hyperparameters**: Edit config.json
4. **Add more metrics**: Extend evaluation.py
5. **Implement more FL algorithms**: Extend federated_learning.py

## Support

For issues and questions, please open an issue on GitHub.
