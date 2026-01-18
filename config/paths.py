"""
Path configuration for SemSeg-LEM project.
Centralized path management to avoid hardcoded paths.
"""

import os
from pathlib import Path


class PathConfig:
    """Centralized path configuration."""

    def __init__(self, project_root=None):
        """
        Initialize path configuration.

        Args:
            project_root: Project root directory. If None, auto-detect.
        """
        if project_root is None:
            # Auto-detect project root (directory containing this file's parent)
            self.project_root = Path(__file__).parent.parent.absolute()
        else:
            self.project_root = Path(project_root).absolute()

        # Data directories
        self.data_root = self.project_root / 'data_root'
        self.images_dir = self.data_root / 'images'
        self.labels_dir = self.data_root / 'labels'
        self.splits_dir = self.data_root / 'splits'

        # Output directories
        self.outputs_root = self.project_root / 'outputs'
        self.checkpoints_dir = self.outputs_root / 'checkpoints'
        self.logs_dir = self.outputs_root / 'logs'
        self.predictions_dir = self.outputs_root / 'predictions'
        self.visualizations_dir = self.outputs_root / 'visualizations'

        # Temporary directories
        self.temp_root = self.project_root / 'temp'
        self.temp_input = self.temp_root / 'input'
        self.temp_output = self.temp_root / 'output'

    def create_directories(self):
        """Create all necessary directories if they don't exist."""
        dirs = [
            self.data_root, self.images_dir, self.labels_dir, self.splits_dir,
            self.outputs_root, self.checkpoints_dir, self.logs_dir,
            self.predictions_dir, self.visualizations_dir,
            self.temp_root, self.temp_input, self.temp_output
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, model_name, epoch=None):
        """Get checkpoint file path."""
        if epoch is not None:
            filename = f"{model_name}_epoch_{epoch}.pth"
        else:
            filename = f"{model_name}_best.pth"
        return self.checkpoints_dir / filename

    def __repr__(self):
        return f"PathConfig(project_root={self.project_root})"


# Global path configuration instance
paths = PathConfig()
