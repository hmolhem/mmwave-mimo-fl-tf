"""
Simple test script to validate the implementation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

import numpy as np
import tensorflow as tf
from data import create_synthetic_data, normalize_data, split_dataset
from data.tf_pipeline import prepare_datasets, create_federated_datasets
from models import build_cnn_model, compile_model
from training import CentralizedTrainer, FedAvgTrainer
from evaluation import ModelEvaluator

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def test_data_loading():
    """Test data loading and preprocessing."""
    print("Testing data loading...")
    
    # Create synthetic data
    data, labels = create_synthetic_data(num_samples=100, num_classes=10)
    assert data.shape == (100, 64, 64), f"Wrong data shape: {data.shape}"
    assert labels.shape == (100,), f"Wrong labels shape: {labels.shape}"
    
    # Normalize
    data_norm = normalize_data(data, method='minmax')
    assert data_norm.min() >= 0 and data_norm.max() <= 1, "Normalization failed"
    
    # Split
    splits = split_dataset(data, labels)
    assert len(splits) == 6, "Split should return 6 arrays"
    
    print("  ✓ Data loading tests passed")


def test_tf_pipeline():
    """Test TensorFlow data pipeline."""
    print("Testing TF data pipeline...")
    
    data, labels = create_synthetic_data(num_samples=100, num_classes=10)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset(data, labels)
    
    # Create datasets
    train_ds, val_ds, test_ds = prepare_datasets(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
        batch_size=16
    )
    
    # Check dataset
    for batch_x, batch_y in train_ds.take(1):
        assert batch_x.shape[0] <= 16, "Batch size issue"
        assert len(batch_x.shape) == 4, "Should be (batch, H, W, C)"
    
    print("  ✓ TF pipeline tests passed")


def test_federated_datasets():
    """Test federated dataset creation."""
    print("Testing federated datasets...")
    
    data, labels = create_synthetic_data(num_samples=100, num_classes=10)
    
    # Create federated datasets
    client_datasets = create_federated_datasets(
        data, labels,
        num_clients=5,
        batch_size=8,
        iid=True
    )
    
    assert len(client_datasets) == 5, "Should have 5 clients"
    
    # Check each client has data
    for i, ds in enumerate(client_datasets):
        count = 0
        for _ in ds:
            count += 1
        assert count > 0, f"Client {i} has no data"
    
    print("  ✓ Federated datasets tests passed")


def test_model_building():
    """Test CNN model building."""
    print("Testing model building...")
    
    # Build different architectures
    for arch in ['standard', 'deep', 'lightweight']:
        model = build_cnn_model(
            input_shape=(64, 64, 1),
            num_classes=10,
            architecture=arch
        )
        model = compile_model(model)
        
        # Test forward pass
        test_input = np.random.randn(1, 64, 64, 1).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        assert output.shape == (1, 10), f"Wrong output shape for {arch}"
    
    print("  ✓ Model building tests passed")


def test_centralized_training():
    """Test centralized training (short run)."""
    print("Testing centralized training...")
    
    # Create small dataset
    data, labels = create_synthetic_data(num_samples=50, num_classes=10)
    data = normalize_data(data)
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset(data, labels)
    
    # Create datasets
    train_ds, val_ds, test_ds = prepare_datasets(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
        batch_size=8
    )
    
    # Build model
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10, architecture='lightweight')
    model = compile_model(model, learning_rate=0.01)
    
    # Train for 2 epochs
    trainer = CentralizedTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        save_dir='/tmp/test_checkpoints',
        experiment_name='test_centralized'
    )
    trainer.train(epochs=2, verbose=0)
    
    # Evaluate
    metrics = trainer.evaluate(test_ds)
    assert len(metrics) >= 2, "Should have at least 2 metrics"
    
    print("  ✓ Centralized training tests passed")


def test_federated_training():
    """Test federated training (short run)."""
    print("Testing federated training...")
    
    # Create small dataset
    data, labels = create_synthetic_data(num_samples=50, num_classes=10)
    data = normalize_data(data)
    
    # Split
    indices = np.random.permutation(len(data))
    split_point = int(len(data) * 0.8)
    train_data = data[indices[:split_point]]
    train_labels = labels[indices[:split_point]]
    test_data = data[indices[split_point:]]
    test_labels = labels[indices[split_point:]]
    
    # Create federated datasets
    client_datasets = create_federated_datasets(
        train_data, train_labels,
        num_clients=3,
        batch_size=8,
        iid=True
    )
    
    from data.tf_pipeline import create_tf_dataset
    val_ds = create_tf_dataset(train_data[:10], train_labels[:10], batch_size=8, shuffle=False)
    test_ds = create_tf_dataset(test_data, test_labels, batch_size=8, shuffle=False)
    
    # Build model
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10, architecture='lightweight')
    model = compile_model(model, learning_rate=0.01)
    
    # Train for 2 rounds
    trainer = FedAvgTrainer(
        model=model,
        client_datasets=client_datasets,
        val_dataset=val_ds,
        num_clients=3,
        save_dir='/tmp/test_checkpoints',
        experiment_name='test_federated'
    )
    trainer.train(num_rounds=2, client_epochs=1, verbose=0)
    
    # Evaluate
    metrics = trainer.evaluate(test_ds)
    assert len(metrics) >= 2, "Should have at least 2 metrics"
    
    print("  ✓ Federated training tests passed")


def test_evaluation():
    """Test evaluation utilities."""
    print("Testing evaluation...")
    
    # Create small dataset and model
    data, labels = create_synthetic_data(num_samples=30, num_classes=10)
    data = normalize_data(data)
    
    from data.tf_pipeline import create_tf_dataset
    test_ds = create_tf_dataset(data, labels, batch_size=8, shuffle=False)
    
    # Build and train a dummy model
    model = build_cnn_model(input_shape=(64, 64, 1), num_classes=10, architecture='lightweight')
    model = compile_model(model)
    
    # Quick train
    model.fit(test_ds, epochs=1, verbose=0)
    
    # Evaluate
    evaluator = ModelEvaluator(model, save_dir='/tmp/test_results')
    metrics = evaluator.evaluate_dataset(test_ds)
    
    assert 'accuracy' in metrics, "Missing accuracy"
    assert 'f1_weighted' in metrics, "Missing F1 score"
    
    print("  ✓ Evaluation tests passed")


def main():
    """Run all tests."""
    print("="*60)
    print("Running Implementation Tests")
    print("="*60)
    print()
    
    try:
        test_data_loading()
        test_tf_pipeline()
        test_federated_datasets()
        test_model_building()
        test_centralized_training()
        test_federated_training()
        test_evaluation()
        
        print()
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        return 0
        
    except Exception as e:
        print()
        print("="*60)
        print(f"Test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
