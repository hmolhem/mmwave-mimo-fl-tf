"""
Example script comparing centralized vs federated learning.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from data import create_synthetic_data, normalize_data, split_dataset
from data.tf_pipeline import prepare_datasets, create_federated_datasets, create_tf_dataset
from models import build_cnn_model, compile_model
from training import CentralizedTrainer, FedAvgTrainer
from evaluation import compare_models, ModelEvaluator


def main():
    """Main function for comparing centralized vs federated training."""
    
    print("="*60)
    print("Centralized vs Federated Learning Comparison")
    print("="*60)
    
    # Configuration
    NUM_CLASSES = 10
    NUM_SAMPLES = 1000
    HEIGHT, WIDTH = 64, 64
    BATCH_SIZE = 32
    EPOCHS = 30  # Reduced for quick comparison
    NUM_CLIENTS = 5
    
    # Create data
    print("\n1. Creating and preparing data...")
    data, labels = create_synthetic_data(
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
        height=HEIGHT,
        width=WIDTH
    )
    data = normalize_data(data, method='minmax')
    
    # Split dataset
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset(
        data, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Create test dataset (shared for both approaches)
    test_dataset = create_tf_dataset(
        test_data, test_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    # === CENTRALIZED TRAINING ===
    print("\n" + "="*60)
    print("Training Centralized Model")
    print("="*60)
    
    # Prepare centralized datasets
    train_dataset_cent, val_dataset_cent, _ = prepare_datasets(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
        batch_size=BATCH_SIZE,
        augment_train=True
    )
    
    # Build and compile centralized model
    model_cent = build_cnn_model(
        input_shape=(HEIGHT, WIDTH, 1),
        num_classes=NUM_CLASSES,
        architecture='standard'
    )
    model_cent = compile_model(model_cent, learning_rate=0.001)
    
    # Train centralized
    trainer_cent = CentralizedTrainer(
        model=model_cent,
        train_dataset=train_dataset_cent,
        val_dataset=val_dataset_cent,
        save_dir='../checkpoints',
        experiment_name='comparison_centralized'
    )
    trainer_cent.train(epochs=EPOCHS, verbose=1)
    
    # === FEDERATED TRAINING ===
    print("\n" + "="*60)
    print("Training Federated Model (FedAvg)")
    print("="*60)
    
    # Prepare federated datasets
    client_datasets = create_federated_datasets(
        train_data,
        train_labels,
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        iid=True
    )
    
    val_dataset_fed = create_tf_dataset(
        val_data, val_labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    # Build and compile federated model
    model_fed = build_cnn_model(
        input_shape=(HEIGHT, WIDTH, 1),
        num_classes=NUM_CLASSES,
        architecture='standard'
    )
    model_fed = compile_model(model_fed, learning_rate=0.001)
    
    # Train federated
    trainer_fed = FedAvgTrainer(
        model=model_fed,
        client_datasets=client_datasets,
        val_dataset=val_dataset_fed,
        num_clients=NUM_CLIENTS,
        save_dir='../checkpoints',
        experiment_name='comparison_federated'
    )
    trainer_fed.train(num_rounds=EPOCHS, client_epochs=1, verbose=1)
    
    # === COMPARISON ===
    print("\n" + "="*60)
    print("Comparing Models")
    print("="*60)
    
    models_dict = {
        'Centralized': trainer_cent.model,
        'Federated_FedAvg': trainer_fed.global_model
    }
    
    class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
    
    comparison_results = compare_models(
        models=models_dict,
        test_dataset=test_dataset,
        class_names=class_names,
        save_dir='../results/comparison'
    )
    
    # Plot training histories side by side
    print("\n4. Generating comparison visualizations...")
    
    # Load histories
    import json
    from pathlib import Path
    
    cent_history_path = Path('../checkpoints/comparison_centralized/history.json')
    fed_history_path = Path('../checkpoints/comparison_federated/history.json')
    
    if cent_history_path.exists() and fed_history_path.exists():
        import matplotlib.pyplot as plt
        
        with open(cent_history_path, 'r') as f:
            cent_history = json.load(f)
        with open(fed_history_path, 'r') as f:
            fed_history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy comparison
        axes[0].plot(cent_history['accuracy'], label='Centralized Train', linestyle='-')
        axes[0].plot(cent_history['val_accuracy'], label='Centralized Val', linestyle='--')
        axes[0].plot(fed_history['train_accuracy'], label='Federated Train', linestyle='-')
        axes[0].plot(fed_history['val_accuracy'], label='Federated Val', linestyle='--')
        axes[0].set_xlabel('Epoch/Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy Comparison')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss comparison
        axes[1].plot(cent_history['loss'], label='Centralized Train', linestyle='-')
        axes[1].plot(cent_history['val_loss'], label='Centralized Val', linestyle='--')
        axes[1].plot(fed_history['train_loss'], label='Federated Train', linestyle='-')
        axes[1].plot(fed_history['val_loss'], label='Federated Val', linestyle='--')
        axes[1].set_xlabel('Epoch/Round')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Comparison')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        comp_plot_path = Path('../results/comparison/training_comparison.png')
        comp_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(comp_plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {comp_plot_path}")
        plt.close()
    
    print("\n" + "="*60)
    print("Comparison Complete!")
    print("="*60)
    print("\nResults saved to ../results/comparison/")


if __name__ == '__main__':
    main()
