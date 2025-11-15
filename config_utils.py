"""
Configuration utilities for loading and managing settings.
"""

import json
import os
from typing import Dict, Any


def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str = 'config.json'):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get data-related configuration."""
    return config.get('data', {})


def get_training_config(config: Dict[str, Any], mode: str = 'centralized') -> Dict[str, Any]:
    """
    Get training-related configuration.
    
    Args:
        config: Full configuration dictionary
        mode: Training mode ('centralized' or 'federated')
        
    Returns:
        Training configuration for the specified mode
    """
    training_config = config.get('training', {})
    return training_config.get(mode, {})


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get model-related configuration."""
    return config.get('model', {})


def get_paths_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get paths configuration."""
    return config.get('paths', {})


def setup_directories(config: Dict[str, Any]):
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    paths = get_paths_config(config)
    
    for key, path in paths.items():
        if path and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config: Original configuration dictionary
        updates: Dictionary with updates
        
    Returns:
        Updated configuration
    """
    import copy
    updated_config = copy.deepcopy(config)
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return deep_update(updated_config, updates)


# Example usage
if __name__ == '__main__':
    # Load default configuration
    config = load_config('config.json')
    
    print("="*60)
    print("Current Configuration")
    print("="*60)
    print_config(config)
    
    # Setup directories
    print("\n" + "="*60)
    print("Setting up directories")
    print("="*60)
    setup_directories(config)
    
    # Example: Update and save configuration
    updates = {
        'training': {
            'centralized': {
                'epochs': 100
            }
        }
    }
    
    updated_config = update_config(config, updates)
    print("\n" + "="*60)
    print("Updated Configuration (epochs changed to 100)")
    print("="*60)
    print_config(updated_config.get('training', {}))
