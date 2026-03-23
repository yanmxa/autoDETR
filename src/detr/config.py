"""
DETR Configuration

Centralized configuration for training and evaluation.
This ensures consistency between train.py and evaluate.py.
"""

import torch


def get_model_config():
    """
    Get model architecture configuration.

    This config is shared between training and evaluation to ensure
    the model architecture matches when loading checkpoints.

    Returns:
        dict: Model configuration parameters
    """
    config = {
        'num_classes': 3,
        'num_queries': 25,
        'num_encoder_layers': 1,
        'num_decoder_layers': 1,
        'nheads': 8,
        'hidden_dim': 256,
        'dropout': 0.2,
    }
    return config


def get_training_config():
    """
    Get complete training configuration.

    Returns:
        dict: Training configuration with all hyperparameters
    """
    # Start with model config
    config = get_model_config()

    # Add training-specific settings
    config.update({
        # Data paths
        'train_data_dir': 'data/train',
        'test_data_dir': 'data/test',

        # Training hyperparameters
        'epochs': 150,
        'batch_size': 4,
        'learning_rate': 5e-5,
        'grad_clip_max_norm': 1.0,

        # Loss weights
        'loss_weights': {
            'class_weighting': 8.0,
            'bbox_weighting': 6.0,
            'giou_weighting': 2.0,
        },
        'eos_coef': 0.1,

        # Optimizer and scheduler
        'optimizer': 'Adam',
        'scheduler': 'ReduceLROnPlateau',
        'patience': 15,
        'lr_factor': 0.5,
        'min_lr': 5e-7,

        # Checkpoint configuration
        'pretrained_path': None,
        'checkpoint_dir': 'checkpoints',
        'save_interval': 10,

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

    return config


def get_evaluation_config():
    """
    Get evaluation configuration.

    Returns:
        dict: Evaluation configuration
    """
    # Start with model config to ensure architecture matches training
    config = get_model_config()

    # Add evaluation-specific settings
    config.update({
        # Data paths
        'test_data_dir': 'data/test',

        # Evaluation settings
        'batch_size': 4,
        'confidence_threshold': 0.10,
        'checkpoint_path': 'checkpoints/epoch_best.pt',
        'top_k': 1,

        # Visualization
        'num_images': 8,
        'save_path': 'evaluation_results.png',

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

    return config
