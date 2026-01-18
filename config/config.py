"""
Global configuration for SemSeg-LEM project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Global configuration manager."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to YAML configuration file. If None, use defaults.
        """
        # Default configuration
        self.config = self._get_default_config()

        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            # Training settings
            'training': {
                'batch_size': 4,
                'num_epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'num_workers': 4,
                'device': 'cuda',
                'seed': 42,
                'save_interval': 10,
                'val_interval': 1,
            },

            # Model settings
            'model': {
                'name': 'unet',  # Model name from registry
                'num_classes': 2,
                'in_channels': 3,
                'pretrained': False,
            },

            # Data settings
            'data': {
                'image_size': (256, 256),
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'augmentation': True,
            },

            # Loss settings
            'loss': {
                'name': 'dice',  # dice, focal, lovasz, etc.
                'weight': None,  # Class weights
            },

            # Optimizer settings
            'optimizer': {
                'name': 'adam',  # adam, sgd, adamw
                'momentum': 0.9,  # For SGD
            },

            # Scheduler settings
            'scheduler': {
                'name': 'cosine',  # cosine, step, plateau
                'step_size': 30,  # For StepLR
                'gamma': 0.1,  # For StepLR
            },
        }

    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)

        # Update default config with user config
        self._update_config(self.config, user_config)

    def _update_config(self, default: Dict, update: Dict):
        """Recursively update configuration."""
        for key, value in update.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._update_config(default[key], value)
            else:
                default[key] = value

    def save_to_file(self, config_file: str):
        """Save configuration to YAML file."""
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def get(self, key: str, default=None):
        """Get configuration value by key (supports nested keys with '.')."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value by key (supports nested keys with '.')."""
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def __repr__(self):
        return f"Config({self.config})"


# Global configuration instance
config = Config()
