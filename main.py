"""
Main entry point for SemSeg-LEM project.
Unified script for model selection, training, and prediction.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config, PathConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SemSeg-LEM: Semantic Segmentation with Multiple Models'
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'evaluate', 'visualize'],
        required=True,
        help='Operation mode'
    )

    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        default='unet',
        help='Model name (e.g., unet, acc_unet, brau_unet, seta_unet, swin_unet, kan_unet)'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )

    # Data paths
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to data directory'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )

    # List models
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models'
    )

    return parser.parse_args()


def list_available_models():
    """List all available models."""
    print("\n=== Available Models ===\n")

    models = {
        'Basic UNet': ['unet', 'unet_plus'],
        'Attention UNet': ['acc_unet', 'brau_unet', 'seta_unet', 'ege_unet'],
        'Transformer UNet': ['swin_unet', 'da_trans_unet'],
        'KAN UNet': ['kan_unet'],
    }

    for category, model_list in models.items():
        print(f"{category}:")
        for model in model_list:
            print(f"  - {model}")
        print()


def main():
    """Main function."""
    args = parse_args()

    # List models if requested
    if args.list_models:
        list_available_models()
        return

    # Initialize configuration
    config = Config(args.config)
    paths = PathConfig()

    # Override config with command line arguments
    if args.epochs is not None:
        config.set('training.num_epochs', args.epochs)
    if args.batch_size is not None:
        config.set('training.batch_size', args.batch_size)
    if args.lr is not None:
        config.set('training.learning_rate', args.lr)
    if args.device is not None:
        config.set('training.device', args.device)

    config.set('model.name', args.model)

    # Create necessary directories
    paths.create_directories()

    print(f"\n=== SemSeg-LEM ===")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Device: {config.get('training.device')}")
    print(f"Project Root: {paths.project_root}\n")

    # Execute based on mode
    if args.mode == 'train':
        from core.trainer import Trainer
        trainer = Trainer(config, paths)
        trainer.train()

    elif args.mode == 'predict':
        from core.predictor import Predictor
        predictor = Predictor(config, paths, args.checkpoint)
        predictor.predict()

    elif args.mode == 'evaluate':
        from core.evaluator import Evaluator
        evaluator = Evaluator(config, paths, args.checkpoint)
        evaluator.evaluate()

    elif args.mode == 'visualize':
        print("Visualization mode - launching GUI...")
        from gui.main_window import launch_gui
        launch_gui()


if __name__ == '__main__':
    main()
