"""
mmWave MIMO Federated Learning with TensorFlow

A comprehensive TensorFlow/Keras implementation for CNN-based ROI classification
on mmWave MIMO radar range-azimuth data, with support for both centralized
and federated learning (FedAvg).
"""

__version__ = '1.0.0'
__author__ = 'mmwave-mimo-fl-tf contributors'

# Make key components easily accessible
from .data import (
    load_mat_file,
    load_dataset,
    normalize_data,
    split_dataset,
    create_synthetic_data,
    create_tf_dataset,
    create_federated_datasets,
    prepare_datasets
)

from .models import (
    build_cnn_model,
    compile_model
)

from .training import (
    CentralizedTrainer,
    train_centralized,
    FedAvgTrainer,
    train_federated
)

from .evaluation import (
    ModelEvaluator,
    evaluate_model,
    compare_models
)

__all__ = [
    # Data
    'load_mat_file',
    'load_dataset',
    'normalize_data',
    'split_dataset',
    'create_synthetic_data',
    'create_tf_dataset',
    'create_federated_datasets',
    'prepare_datasets',
    
    # Models
    'build_cnn_model',
    'compile_model',
    
    # Training
    'CentralizedTrainer',
    'train_centralized',
    'FedAvgTrainer',
    'train_federated',
    
    # Evaluation
    'ModelEvaluator',
    'evaluate_model',
    'compare_models'
]
