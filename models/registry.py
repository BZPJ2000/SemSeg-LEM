"""
Model registry for unified model management.
Allows dynamic model loading by name.
"""

import importlib
from typing import Dict, Callable, Any, Optional


class ModelRegistry:
    """Registry for managing model architectures."""

    def __init__(self):
        self._models: Dict[str, Callable] = {}
        self._register_builtin_models()

    def register(self, name: str, model_class: Callable):
        """
        Register a model class.

        Args:
            name: Model name (e.g., 'unet', 'acc_unet')
            model_class: Model class or factory function
        """
        if name in self._models:
            print(f"Warning: Model '{name}' already registered. Overwriting.")
        self._models[name] = model_class

    def get(self, name: str, **kwargs) -> Any:
        """
        Get model instance by name.

        Args:
            name: Model name
            **kwargs: Arguments to pass to model constructor

        Returns:
            Model instance

        Raises:
            ValueError: If model name not found
        """
        if name not in self._models:
            raise ValueError(
                f"Model '{name}' not found. Available models: {self.list()}"
            )

        model_class = self._models[name]
        return model_class(**kwargs)

    def list(self) -> list:
        """List all registered model names."""
        return sorted(self._models.keys())

    def _register_builtin_models(self):
        """Register built-in models."""
        # This will be populated as we migrate models
        pass


# Global registry instance
_registry = ModelRegistry()


def register_model(name: str):
    """
    Decorator to register a model.

    Usage:
        @register_model('my_model')
        class MyModel(nn.Module):
            ...
    """
    def decorator(model_class):
        _registry.register(name, model_class)
        return model_class
    return decorator


def get_model(name: str, **kwargs):
    """Get model by name."""
    return _registry.get(name, **kwargs)


def list_models():
    """List all available models."""
    return _registry.list()
