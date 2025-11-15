"""
Example script for federated learning (FedAvg) on mmWave MIMO radar data.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from data import create_synthetic_data, normalize_data, split_dataset
from data.tf_pipeline import create_federated_datasets, create_tf_dataset
from models import build_cnn_model, compile_model
from training import train_federated
from evaluation import evaluate_model, ModelEvaluator


def main():
    """Main function for federated training example."""
    
    print("="*60)
    print("Federated Learning (FedAvg) Training Example")
    print("="*60)
    
    # Configuration
    NUM_CLASSES = 10
    NUM_SAMPLES = 1000
    HEIGHT, WIDTH = 64, 64
    BATCH_SIZE = 32
    NUM_ROUNDS = 50
    CLIENT_EPOCHS = 1
    NUM_CLIENTS = 5
    IID = True  # Set to False for non-IID data distribution
    
    # Step 1: Create/Load synthetic data
    print("\n1. Creating synthetic data...")
    data, labels = create_synthetic_data(
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
        height=HEIGHT,
        width=WIDTH
    )
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Step 2: Normalize data
    print("\n2. Normalizing data...")
    data = normalize_data(data, method='minmax')
    
    # Step 3: Split dataset (reserve test set, split train+val for clients)
    print("\n3. Splitting dataset...")
    train_val_ratio = 0.85
    test_ratio = 0.15
    
    # First split: train+val vs test
    indices = np.random.permutation(len(data))
    split_point = int(len(data) * train_val_ratio)
    
    train_val_data = data[indices[:split_point]]
    train_val_labels = labels[indices[:split_point]]
    test_data = data[indices[split_point:]]
    test_labels = labels[indices[split_point:]]
    
    # Further split train_val into train and val (80-20 of the 85%)
    split_point2 = int(len(train_val_data) * 0.8)
    train_data = train_val_data[:split_point2]
    train_labels = train_val_labels[:split_point2]
    val_data = train_val_data[split_point2:]
    val_labels = train_val_labels[split_point2:]
    
    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    
    # Step 4: Create federated datasets for clients
    print(f"\n4. Creating {NUM_CLIENTS} federated client datasets...")
    print(f"   Distribution: {'IID' if IID else 'Non-IID'}")
    client_datasets = create_federated_datasets(
        train_data,
        train_labels,
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        iid=IID
    )
    
    # Create validation and test datasets
    val_dataset = create_tf_dataset(
        val_data, val_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    test_dataset = create_tf_dataset(
        test_data, test_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    # Step 5: Build global model
    print("\n5. Building global CNN model...")
    model = build_cnn_model(
        input_shape=(HEIGHT, WIDTH, 1),
        num_classes=NUM_CLASSES,
        architecture='lightweight'  # Use lightweight for faster federated training
    )
    
    # Step 6: Compile model
    print("\n6. Compiling model...")
    model = compile_model(model, learning_rate=0.001)
    model.summary()
    
    # Step 7: Federated training
    print("\n7. Starting federated training...")
    print(f"   Number of clients: {NUM_CLIENTS}")
    print(f"   Number of rounds: {NUM_ROUNDS}")
    print(f"   Client epochs: {CLIENT_EPOCHS}")
    
    trained_model, eval_metrics = train_federated(
        client_datasets=client_datasets,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        model=model,
        num_rounds=NUM_ROUNDS,
        client_epochs=CLIENT_EPOCHS,
        save_dir='../checkpoints',
        experiment_name='fedavg_example'
    )
    
    # Step 8: Detailed evaluation with confusion matrix
    print("\n8. Generating detailed evaluation...")
    class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
    evaluator = ModelEvaluator(trained_model, save_dir='../results/federated')
    
    # Load training history for plotting
    import json
    from pathlib import Path
    history_path = Path('../checkpoints/fedavg_example/history.json')
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        evaluator.plot_training_history(history, save_name='federated_training_history.png')
    
    # Evaluate on test set
    metrics = evaluator.evaluate_dataset(test_dataset, class_names)
    
    print("\n" + "="*60)
    print("Federated Training Complete!")
    print("="*60)
    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Test F1-Score: {metrics['f1_weighted']:.4f}")
    print("\nResults saved to:")
    print("  - Checkpoints: ../checkpoints/fedavg_example/")
    print("  - Results: ../results/federated/")


if __name__ == '__main__':
    main()
