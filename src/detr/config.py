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
        # Model architecture (PHASE 4: Enhanced Phase 2 - Best proven config)
        'num_classes': 3,
        'num_queries': 50,         # REDUCED from 100 (simpler scenes, Phase 2 used 100)
        'num_encoder_layers': 2,   # RESTORED from Phase 2 (was 1 in Phase 3)
        'num_decoder_layers': 2,   # RESTORED from Phase 2 (was 1 in Phase 3)
        'nheads': 8,
        'hidden_dim': 256,         # RESTORED from Phase 2 (was 128 in Phase 3)
        'dropout': 0.2,            # MODERATE regularization (was 0.3 in Phase 3, 0.1 in Phase 2)
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

        # Training hyperparameters (PHASE 4: Enhanced Phase 2)
        'epochs': 150,             # INCREASED from Phase 2 (100) for better convergence
        'batch_size': 4,           # RESTORED from Phase 2 (was 2 in Phase 3)
        'learning_rate': 1e-4,     # Same as Phase 2 (proven effective)
        'grad_clip_max_norm': 1.0, # Gradient clipping threshold

        # Loss weights (Balanced between Phase 2 and Phase 3)
        'loss_weights': {
            'class_weighting': 2.5,  # Phase 2=2.0, Phase 3=3.0 → middle ground
            'bbox_weighting': 12.0,  # Phase 2=10.0, Phase 3=15.0 → middle ground
            'giou_weighting': 6.0    # Phase 2=5.0, Phase 3=8.0 → middle ground
        },
        'eos_coef': 0.015,         # Phase 2=0.02, Phase 3=0.01 → middle ground

        # Optimizer and scheduler (PHASE 4: Adaptive LR for stability)
        'optimizer': 'Adam',       # 'Adam' or 'AdamW'
        'scheduler': 'ReduceLROnPlateau',  # CHANGED: Adaptive scheduling
        'patience': 10,            # Reduce LR if no improvement for 10 epochs
        'lr_factor': 0.5,          # Reduce LR by 50% when triggered
        'min_lr': 1e-6,            # Minimum learning rate

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
        'confidence_threshold': 0.26,  # LOWERED from 0.25 for small model
        'checkpoint_path': 'checkpoints/epoch_best.pt',  # Path to trained model

        # Visualization
        'num_samples': 4,  # Number of samples to visualize
        'save_path': 'evaluation_results.png',  # Where to save visualization

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

    return config
