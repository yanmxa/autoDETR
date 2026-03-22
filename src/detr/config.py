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
        # Model architecture (PHASE 7: Ultra-simplified for 170 samples)
        'num_classes': 3,
        'num_queries': 25,         # REDUCED 75%: 100→25 to reduce complexity
        'num_encoder_layers': 1,   # REDUCED 50%: Minimal transformer stack
        'num_decoder_layers': 1,   # REDUCED 50%: Minimal transformer stack
        'nheads': 8,
        'hidden_dim': 256,         # KEPT: Required for 8 heads (256/8=32 per head)
        'dropout': 0.2,            # INCREASED: More regularization for small dataset
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
        'train_data_dir': 'data/train',      # Using 85/15 split (170 train)
        'test_data_dir': 'data/test',        # 32 test samples

        # Training hyperparameters (PHASE 7.1: Extended training)
        'epochs': 150,             # INCREASED: Allow more fine-tuning time
        'batch_size': 4,           # Good balance
        'learning_rate': 1e-4,     # CRITICAL: Keep 1e-4 (Phase 1 proved 1e-5 fails)
        'grad_clip_max_norm': 1.0, # Gradient clipping threshold

        # Loss weights (PHASE 7.2: Moderate adjustment - gradual improvement)
        'loss_weights': {
            'class_weighting': 2.0,  # MODERATE: 1.5→2.0 (+33% not +67%) for balanced classification
            'bbox_weighting': 10.0,  # Keep Phase 5 value for accurate boxes
            'giou_weighting': 5.0    # Keep for better shape matching
        },
        'eos_coef': 0.03,          # MODERATE: 0.02→0.03 (+50% not +150%) for empty image detection

        # Optimizer and scheduler (PHASE 7.2: Proven stable schedule)
        'optimizer': 'Adam',       # 'Adam' or 'AdamW'
        'scheduler': 'ReduceLROnPlateau',  # Phase 4's ONLY successful improvement
        'patience': 10,            # RESTORED: Phase 7 proved this works well
        'lr_factor': 0.3,          # DECREASED: Gentler reduction (was 0.5) for late-stage
        'min_lr': 5e-7,            # DECREASED: Allow even lower LR for fine tuning

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
        'test_data_dir': 'data/test',

        # Evaluation settings
        'batch_size': 4,
        'confidence_threshold': 0.10,  # LOWERED from 0.25 for small model
        'checkpoint_path': 'checkpoints/epoch_best.pt',  # Path to trained model
        'top_k': 1,  # Keep top-k detections per image. Set to 1 for highest confidence only, None for all

        # Visualization
        'num_images': 8,  # Number of images to display (will auto-calculate grid layout)
        'save_path': 'evaluation_results.png',  # Where to save visualization

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })

    return config
