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
        # Model architecture (PHASE 2: PROVEN OPTIMAL for 85 samples)
        'num_classes': 3,
        'num_queries': 100,        # OPTIMAL: Don't reduce (Phase 4 proved 50 causes duplicates)
        'num_encoder_layers': 2,   # MINIMUM for DETR
        'num_decoder_layers': 2,   # MINIMUM for DETR
        'nheads': 8,
        'hidden_dim': 256,         # MINIMUM for attention mechanism
        'dropout': 0.1,            # OPTIMAL: Don't increase (Phase 4 proved 0.2 over-regularizes)
    }
    return config


def get_training_config():
    """
    Get complete training configuration.

    Returns:
        dict: Training configuration with all hyperparameters

    Notes:
        - Set 'pretrained_path' to None to skip loading pretrained weights
        - Set 'checkpoint_dir' to None to skip saving checkpoints
        - Adjust 'save_interval' to control checkpoint frequency
    """
    # Start with model config
    config = get_model_config()

    # Add training-specific settings
    config.update({
        # Data paths
        'train_data_dir': 'data_new/train',
        'test_data_dir': 'data_new/test',

        # Training hyperparameters (PHASE 2.5: Phase 2 + improved scheduler)
        'epochs': 120,             # Extended for ReduceLROnPlateau (Phase 2 best at epoch 71)
        'batch_size': 4,           # Good balance
        'learning_rate': 1e-4,     # Stable & effective
        'grad_clip_max_norm': 1.0, # Gradient clipping threshold

        # Loss weights (PHASE 2: BALANCED - DON'T CHANGE)
        'loss_weights': {
            'class_weighting': 2.0,  # BALANCED: Proven optimal
            'bbox_weighting': 10.0,  # STRONG localization: Proven optimal
            'giou_weighting': 5.0    # GOOD overlap: Proven optimal
        },
        'eos_coef': 0.02,          # MINIMAL background bias: Proven optimal

        # Optimizer and scheduler (PHASE 2.5: Phase 2 + Phase 4's improved scheduler)
        'optimizer': 'Adam',       # 'Adam' or 'AdamW'
        'scheduler': 'ReduceLROnPlateau',  # Phase 4's ONLY successful improvement
        'patience': 10,            # Reduce LR if no improvement for 10 epochs
        'lr_factor': 0.5,          # Reduce LR by 50% when triggered
        'min_lr': 1e-6,            # Minimum learning rate

        # Alternative: Use Phase 2's original scheduler (comment above, uncomment below)
        # 'scheduler': 'CosineAnnealingWarmRestarts',
        # 'T_0': 30,
        # 'T_mult': 1,
        # 'eta_min': 1e-6,

        # Checkpoint configuration
        'pretrained_path': None,  # Set to path string to load pretrained model, None to skip
        'checkpoint_dir': 'checkpoints',  # Directory to save checkpoints, None to skip
        'save_interval': 10,  # Save checkpoint every N epochs

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
        'test_data_dir': 'data_new/test',

        # Evaluation settings
        'batch_size': 4,
        'confidence_threshold': 0.2,  # LOWERED from 0.25 for small model
        'checkpoint_path': 'checkpoints/epoch_best.pt',  # Path to trained model

        # Visualization
        'num_samples': 4,  # Number of samples to visualize
        'save_path': 'evaluation_results.png',  # Where to save visualization

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

    return config
