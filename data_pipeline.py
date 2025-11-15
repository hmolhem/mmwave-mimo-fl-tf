"""
TensorFlow data pipeline utilities for efficient data loading and preprocessing.
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Callable


def create_tf_dataset(X: np.ndarray,
                      y: np.ndarray,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      shuffle_buffer_size: int = 1000,
                      augment: bool = False,
                      prefetch: bool = True) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from numpy arrays.
    
    Args:
        X: Input data array
        y: Labels array
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        shuffle_buffer_size: Buffer size for shuffling
        augment: Whether to apply data augmentation
        prefetch: Whether to prefetch batches
        
    Returns:
        TensorFlow Dataset
    """
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    
    # Apply augmentation if requested
    if augment:
        dataset = dataset.map(
            lambda x, y: (augment_image(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch the data
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """
    Apply data augmentation to a single image.
    
    Args:
        image: Input image tensor
        
    Returns:
        Augmented image tensor
    """
    # Random flip left-right
    image = tf.image.random_flip_left_right(image)
    
    # Random flip up-down
    image = tf.image.random_flip_up_down(image)
    
    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Clip values to valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


def create_federated_datasets(X: np.ndarray,
                              y: np.ndarray,
                              num_clients: int = 10,
                              batch_size: int = 32,
                              shuffle: bool = True,
                              iid: bool = True,
                              random_seed: int = 42) -> list:
    """
    Create federated datasets by partitioning data across clients.
    
    Args:
        X: Input data array
        y: Labels array
        num_clients: Number of federated clients
        batch_size: Batch size for each client
        shuffle: Whether to shuffle data
        iid: Whether to use IID (independent and identically distributed) partitioning
        random_seed: Random seed for reproducibility
        
    Returns:
        List of tf.data.Dataset objects, one per client
    """
    np.random.seed(random_seed)
    n_samples = len(X)
    
    if iid:
        # IID partitioning: randomly assign samples to clients
        indices = np.random.permutation(n_samples)
        client_indices = np.array_split(indices, num_clients)
    else:
        # Non-IID partitioning: sort by labels and partition
        sorted_indices = np.argsort(y)
        client_indices = np.array_split(sorted_indices, num_clients)
    
    # Create dataset for each client
    client_datasets = []
    for idx_list in client_indices:
        X_client = X[idx_list]
        y_client = y[idx_list]
        
        client_dataset = create_tf_dataset(
            X_client, y_client,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_buffer_size=min(len(X_client), 1000)
        )
        client_datasets.append(client_dataset)
    
    return client_datasets


def get_dataset_info(dataset: tf.data.Dataset) -> dict:
    """
    Get information about a tf.data.Dataset.
    
    Args:
        dataset: TensorFlow Dataset
        
    Returns:
        Dictionary with dataset information
    """
    info = {}
    
    # Get element spec
    element_spec = dataset.element_spec
    if isinstance(element_spec, tuple):
        info['input_shape'] = element_spec[0].shape
        info['label_shape'] = element_spec[1].shape
        info['input_dtype'] = element_spec[0].dtype
        info['label_dtype'] = element_spec[1].dtype
    
    return info


def create_client_data_from_splits(X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   num_clients: int,
                                   samples_per_client: Optional[int] = None,
                                   shuffle: bool = True,
                                   random_seed: int = 42) -> Tuple[list, list]:
    """
    Split training data into client-specific datasets.
    
    Args:
        X_train: Training data
        y_train: Training labels
        num_clients: Number of clients
        samples_per_client: Number of samples per client (None for equal split)
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed
        
    Returns:
        Tuple of (list of X arrays, list of y arrays) for each client
    """
    np.random.seed(random_seed)
    n_samples = len(X_train)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    if samples_per_client is None:
        # Equal split
        client_X = np.array_split(X_train, num_clients)
        client_y = np.array_split(y_train, num_clients)
    else:
        # Fixed samples per client
        client_X = []
        client_y = []
        start_idx = 0
        for i in range(num_clients):
            end_idx = min(start_idx + samples_per_client, n_samples)
            client_X.append(X_train[start_idx:end_idx])
            client_y.append(y_train[start_idx:end_idx])
            start_idx = end_idx
            if start_idx >= n_samples:
                break
    
    return client_X, client_y


def prepare_datasets_for_training(X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_val: np.ndarray,
                                  y_val: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  batch_size: int = 32,
                                  augment_train: bool = True) -> Tuple[tf.data.Dataset, 
                                                                        tf.data.Dataset,
                                                                        tf.data.Dataset]:
    """
    Prepare train, validation, and test datasets.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size
        augment_train: Whether to augment training data
        
    Returns:
        Tuple of (train_ds, val_ds, test_ds)
    """
    train_ds = create_tf_dataset(X_train, y_train, batch_size=batch_size, 
                                 shuffle=True, augment=augment_train)
    val_ds = create_tf_dataset(X_val, y_val, batch_size=batch_size, 
                               shuffle=False, augment=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size=batch_size, 
                                shuffle=False, augment=False)
    
    return train_ds, val_ds, test_ds
