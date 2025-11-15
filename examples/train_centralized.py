"""
Example script for centralized training on mmWave MIMO radar data.
"""

import sys
sys.path.insert(0, '../src')

import numpy as np
from data import create_synthetic_data, normalize_data, split_dataset
from data.tf_pipeline import prepare_datasets
from models import build_cnn_model, compile_model
from training import train_centralized
from evaluation import evaluate_model, ModelEvaluator


def main():
    """Main function for centralized training example."""
    
    print("="*60)
    print("Centralized Training Example")
    print("="*60)
    
    # Configuration
    NUM_CLASSES = 10
    NUM_SAMPLES = 1000
    HEIGHT, WIDTH = 64, 64
    BATCH_SIZE = 32
    EPOCHS = 50
    
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
    
    # Step 3: Split dataset
    print("\n3. Splitting dataset...")
    train_data, train_labels, val_data, val_labels, test_data, test_labels = split_dataset(
        data, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    
    # Step 4: Create TF datasets
    print("\n4. Creating TensorFlow datasets...")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels,
        batch_size=BATCH_SIZE,
        augment_train=True
    )
    
    # Step 5: Build model
    print("\n5. Building CNN model...")
    model = build_cnn_model(
        input_shape=(HEIGHT, WIDTH, 1),
        num_classes=NUM_CLASSES,
        architecture='standard'
    )
    
    # Step 6: Compile model
    print("\n6. Compiling model...")
    model = compile_model(model, learning_rate=0.001)
    model.summary()
    
    # Step 7: Train model
    print("\n7. Starting centralized training...")
    trained_model, eval_metrics = train_centralized(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        model=model,
        epochs=EPOCHS,
        save_dir='../checkpoints',
        experiment_name='centralized_example'
    )
    
    # Step 8: Detailed evaluation with confusion matrix
    print("\n8. Generating detailed evaluation...")
    class_names = [f"Class_{i}" for i in range(NUM_CLASSES)]
    evaluator = ModelEvaluator(trained_model, save_dir='../results/centralized')
    
    # Load training history for plotting
    import json
    from pathlib import Path
    history_path = Path('../checkpoints/centralized_example/history.json')
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        evaluator.plot_training_history(history)
    
    # Evaluate on test set
    metrics = evaluator.evaluate_dataset(test_dataset, class_names)
    
    print("\n" + "="*60)
    print("Centralized Training Complete!")
    print("="*60)
    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Test F1-Score: {metrics['f1_weighted']:.4f}")
    print("\nResults saved to:")
    print("  - Checkpoints: ../checkpoints/centralized_example/")
    print("  - Results: ../results/centralized/")


if __name__ == '__main__':
    main()
