"""
Test suite for mmWave MIMO FL TensorFlow project.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


def test_data_loader():
    """Test data loading functionality."""
    print("\n" + "="*60)
    print("Testing Data Loader Module")
    print("="*60)
    
    from data_loader import (
        create_synthetic_data, 
        split_data,
        normalize_range_azimuth
    )
    
    # Test synthetic data generation
    print("\n1. Testing synthetic data generation...")
    X, y = create_synthetic_data(num_samples=100, num_classes=10, image_shape=(64, 64))
    assert X.shape == (100, 64, 64, 1), f"Expected shape (100, 64, 64, 1), got {X.shape}"
    assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"
    assert len(np.unique(y)) == 10, "Should have 10 unique classes"
    print("   ✓ Synthetic data generation works")
    
    # Test data splitting
    print("\n2. Testing data splitting...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    assert len(X_train) == 70, f"Expected 70 training samples, got {len(X_train)}"
    assert len(X_val) == 15, f"Expected 15 validation samples, got {len(X_val)}"
    assert len(X_test) == 15, f"Expected 15 test samples, got {len(X_test)}"
    print("   ✓ Data splitting works")
    
    # Test normalization
    print("\n3. Testing normalization...")
    X_normalized = normalize_range_azimuth(X[:, :, :, 0], method='minmax')
    assert X_normalized.min() >= 0 and X_normalized.max() <= 1, "MinMax normalization failed"
    print("   ✓ Normalization works")
    
    print("\n✓ All data loader tests passed!")
    return True


def test_model():
    """Test model creation and compilation."""
    print("\n" + "="*60)
    print("Testing Model Module")
    print("="*60)
    
    from model import (
        create_cnn_model,
        create_lightweight_cnn,
        compile_model
    )
    
    # Test standard CNN
    print("\n1. Testing standard CNN creation...")
    model = create_cnn_model(input_shape=(128, 128, 1), num_classes=10)
    assert model is not None, "Model creation failed"
    assert len(model.layers) > 0, "Model has no layers"
    print(f"   ✓ Standard CNN created with {len(model.layers)} layers")
    
    # Test lightweight CNN
    print("\n2. Testing lightweight CNN creation...")
    light_model = create_lightweight_cnn(input_shape=(128, 128, 1), num_classes=10)
    assert light_model is not None, "Lightweight model creation failed"
    print(f"   ✓ Lightweight CNN created with {len(light_model.layers)} layers")
    
    # Test compilation
    print("\n3. Testing model compilation...")
    compiled_model = compile_model(model, learning_rate=0.001)
    assert compiled_model.optimizer is not None, "Model not compiled"
    print("   ✓ Model compilation works")
    
    # Test prediction shape
    print("\n4. Testing model output shape...")
    dummy_input = np.random.randn(1, 128, 128, 1).astype(np.float32)
    output = compiled_model.predict(dummy_input, verbose=0)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    print("   ✓ Model output shape correct")
    
    print("\n✓ All model tests passed!")
    return True


def test_data_pipeline():
    """Test data pipeline functionality."""
    print("\n" + "="*60)
    print("Testing Data Pipeline Module")
    print("="*60)
    
    from data_pipeline import (
        create_tf_dataset,
        prepare_datasets_for_training,
        create_client_data_from_splits
    )
    from data_loader import create_synthetic_data
    
    # Generate test data
    X, y = create_synthetic_data(num_samples=100, num_classes=10, image_shape=(64, 64))
    
    # Test tf.data.Dataset creation
    print("\n1. Testing tf.data.Dataset creation...")
    dataset = create_tf_dataset(X, y, batch_size=16, shuffle=True)
    assert dataset is not None, "Dataset creation failed"
    for batch_x, batch_y in dataset.take(1):
        assert batch_x.shape[0] <= 16, "Batch size incorrect"
        assert len(batch_x.shape) == 4, "Batch shape incorrect"
    print("   ✓ tf.data.Dataset creation works")
    
    # Test federated data split
    print("\n2. Testing federated data splitting...")
    client_X, client_y = create_client_data_from_splits(
        X, y, num_clients=5, shuffle=True
    )
    assert len(client_X) == 5, f"Expected 5 clients, got {len(client_X)}"
    total_samples = sum(len(cx) for cx in client_X)
    assert total_samples == 100, f"Expected 100 total samples, got {total_samples}"
    print("   ✓ Federated data splitting works")
    
    print("\n✓ All data pipeline tests passed!")
    return True


def test_training():
    """Test training functionality."""
    print("\n" + "="*60)
    print("Testing Training Functionality")
    print("="*60)
    
    from data_loader import create_synthetic_data, split_data
    from model import create_lightweight_cnn, compile_model
    
    # Generate minimal data
    print("\n1. Testing basic training...")
    X, y = create_synthetic_data(num_samples=50, num_classes=10, image_shape=(32, 32))
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
    
    model = create_lightweight_cnn(input_shape=(32, 32, 1), num_classes=10)
    model = compile_model(model, learning_rate=0.001)
    
    # Train for 1 epoch
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=8,
        verbose=0
    )
    
    assert 'loss' in history.history, "Training history missing loss"
    assert 'accuracy' in history.history, "Training history missing accuracy"
    print("   ✓ Basic training works")
    
    # Test evaluation
    print("\n2. Testing evaluation...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    assert loss >= 0, "Loss should be non-negative"
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"
    print(f"   ✓ Evaluation works (acc: {acc:.4f})")
    
    print("\n✓ All training tests passed!")
    return True


def test_federated_learning():
    """Test federated learning functionality."""
    print("\n" + "="*60)
    print("Testing Federated Learning Module")
    print("="*60)
    
    from federated_learning import FederatedAveraging
    from model import create_lightweight_cnn
    from data_loader import create_synthetic_data, split_data
    from data_pipeline import create_client_data_from_splits
    
    # Generate minimal data
    print("\n1. Testing FedAvg initialization...")
    fedavg = FederatedAveraging(
        model_fn=create_lightweight_cnn,
        num_classes=10,
        input_shape=(32, 32, 1),
        num_clients=3,
        clients_per_round=2,
        local_epochs=1
    )
    assert fedavg.global_model is not None, "Global model not created"
    print("   ✓ FedAvg initialized")
    
    # Test training round
    print("\n2. Testing federated training round...")
    X, y = create_synthetic_data(num_samples=60, num_classes=10, image_shape=(32, 32))
    client_X, client_y = create_client_data_from_splits(X, y, num_clients=3)
    client_datasets = list(zip(client_X, client_y))
    
    metrics = fedavg.train_round(client_datasets, round_num=1, verbose=0)
    assert 'avg_train_loss' in metrics, "Training metrics missing loss"
    assert 'avg_train_accuracy' in metrics, "Training metrics missing accuracy"
    print("   ✓ Federated training round works")
    
    print("\n✓ All federated learning tests passed!")
    return True


def test_evaluation():
    """Test evaluation functionality."""
    print("\n" + "="*60)
    print("Testing Evaluation Module")
    print("="*60)
    
    from evaluation import evaluate_model, compute_per_class_accuracy
    from data_loader import create_synthetic_data
    from model import create_lightweight_cnn, compile_model
    
    # Generate minimal data and train a simple model
    print("\n1. Testing model evaluation...")
    X, y = create_synthetic_data(num_samples=50, num_classes=10, image_shape=(32, 32))
    
    model = create_lightweight_cnn(input_shape=(32, 32, 1), num_classes=10)
    model = compile_model(model, learning_rate=0.001)
    model.fit(X[:40], y[:40], epochs=1, verbose=0)
    
    results = evaluate_model(model, X[40:], y[40:], verbose=0)
    assert 'accuracy' in results, "Results missing accuracy"
    assert 'confusion_matrix' in results, "Results missing confusion matrix"
    print("   ✓ Model evaluation works")
    
    # Test per-class accuracy
    print("\n2. Testing per-class accuracy...")
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    per_class = compute_per_class_accuracy(y_true, y_pred, num_classes=3)
    assert len(per_class) == 3, "Should have 3 classes"
    print("   ✓ Per-class accuracy computation works")
    
    print("\n✓ All evaluation tests passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    tests = [
        ("Data Loader", test_data_loader),
        ("Model", test_model),
        ("Data Pipeline", test_data_pipeline),
        ("Training", test_training),
        ("Federated Learning", test_federated_learning),
        ("Evaluation", test_evaluation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test failed with error: {str(e)}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
