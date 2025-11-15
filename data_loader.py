"""
Data loading utilities for IEEE DataPort mmWave MIMO radar range-azimuth dataset.
"""

import numpy as np
from scipy.io import loadmat
from typing import Tuple, List, Optional
import os
import glob


def load_mat_file(filepath: str) -> dict:
    """
    Load a .mat file and return its contents.
    
    Args:
        filepath: Path to the .mat file
        
    Returns:
        Dictionary containing the loaded data
    """
    try:
        data = loadmat(filepath)
        return data
    except Exception as e:
        raise ValueError(f"Error loading .mat file {filepath}: {str(e)}")


def extract_range_azimuth_data(mat_data: dict, 
                                data_key: str = 'data',
                                label_key: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract range-azimuth data and labels from loaded .mat dictionary.
    
    Args:
        mat_data: Dictionary from loadmat()
        data_key: Key for the data array in the .mat file
        label_key: Key for the label array in the .mat file
        
    Returns:
        Tuple of (data, labels) as numpy arrays
    """
    # Extract data and labels, handling common .mat file structures
    if data_key in mat_data:
        data = mat_data[data_key]
    else:
        # Try to find the data by excluding MATLAB metadata keys
        valid_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(valid_keys) == 0:
            raise ValueError(f"No valid data keys found in .mat file")
        data_key = valid_keys[0]
        data = mat_data[data_key]
    
    labels = mat_data.get(label_key, None)
    
    return data, labels


def normalize_range_azimuth(data: np.ndarray, 
                            method: str = 'minmax') -> np.ndarray:
    """
    Normalize range-azimuth data.
    
    Args:
        data: Input data array
        method: Normalization method ('minmax' or 'standard')
        
    Returns:
        Normalized data array
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        data_min = np.min(data, axis=(1, 2), keepdims=True)
        data_max = np.max(data, axis=(1, 2), keepdims=True)
        normalized = (data - data_min) / (data_max - data_min + 1e-8)
    elif method == 'standard':
        # Standardization (z-score)
        data_mean = np.mean(data, axis=(1, 2), keepdims=True)
        data_std = np.std(data, axis=(1, 2), keepdims=True)
        normalized = (data - data_mean) / (data_std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def load_dataset_from_directory(data_dir: str,
                                pattern: str = '*.mat',
                                num_classes: int = 10,
                                normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multiple .mat files from a directory.
    
    Args:
        data_dir: Directory containing .mat files
        pattern: File pattern to match
        num_classes: Number of classes for classification
        normalize: Whether to normalize the data
        
    Returns:
        Tuple of (data, labels) as numpy arrays
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")
    
    mat_files = glob.glob(os.path.join(data_dir, pattern))
    if len(mat_files) == 0:
        raise ValueError(f"No .mat files found in {data_dir}")
    
    all_data = []
    all_labels = []
    
    for mat_file in sorted(mat_files):
        mat_data = load_mat_file(mat_file)
        data, labels = extract_range_azimuth_data(mat_data)
        
        if data is not None:
            all_data.append(data)
        if labels is not None:
            all_labels.append(labels)
    
    # Concatenate all data
    if all_data:
        data_array = np.concatenate(all_data, axis=0) if len(all_data) > 1 else all_data[0]
    else:
        raise ValueError("No data extracted from .mat files")
    
    if all_labels:
        labels_array = np.concatenate(all_labels, axis=0) if len(all_labels) > 1 else all_labels[0]
    else:
        labels_array = None
    
    # Normalize if requested
    if normalize and data_array is not None:
        data_array = normalize_range_azimuth(data_array)
    
    # Ensure data has channel dimension for CNN
    if len(data_array.shape) == 3:
        # Add channel dimension: (N, H, W) -> (N, H, W, 1)
        data_array = np.expand_dims(data_array, axis=-1)
    
    return data_array, labels_array


def create_synthetic_data(num_samples: int = 1000,
                          num_classes: int = 10,
                          image_shape: Tuple[int, int] = (128, 128)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic range-azimuth data for testing purposes.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
        image_shape: Shape of each range-azimuth image (height, width)
        
    Returns:
        Tuple of (data, labels) as numpy arrays
    """
    # Generate random range-azimuth maps
    data = np.random.randn(num_samples, image_shape[0], image_shape[1], 1).astype(np.float32)
    
    # Add some structure to make it more realistic
    for i in range(num_samples):
        class_idx = i % num_classes
        # Create patterns based on class
        center_x = image_shape[0] // 2 + (class_idx - num_classes // 2) * 5
        center_y = image_shape[1] // 2 + (class_idx - num_classes // 2) * 5
        
        # Add a bright spot at class-dependent location
        y, x = np.ogrid[:image_shape[0], :image_shape[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= (10 + class_idx)**2
        data[i, mask, 0] += 2.0
    
    # Normalize
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    
    # Generate labels
    labels = np.arange(num_samples) % num_classes
    
    return data.astype(np.float32), labels.astype(np.int32)


def split_data(data: np.ndarray, 
               labels: np.ndarray,
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               shuffle: bool = True,
               random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Input data array
        labels: Labels array
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_train = data[train_idx], labels[train_idx]
    X_val, y_val = data[val_idx], labels[val_idx]
    X_test, y_test = data[test_idx], labels[test_idx]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
