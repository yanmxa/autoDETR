"""
Utility functions for DETR project.
"""

from .rich_display import (
    # Dataset displays
    display_dataset_info,

    # Model displays
    display_model_info,
    display_checkpoint_loaded,

    # Training displays
    display_training_header,
    create_training_progress,
    display_checkpoint_saved,
    display_training_complete,
    display_training_error,
    display_info_message,

    # Image capture displays
    display_capture_banner,
    display_capture_session_info,
    display_capture_session_summary,
    create_capture_progress,

    # General print functions
    print_message,
    print_empty_line,
)

__all__ = [
    # Dataset
    "display_dataset_info",

    # Model
    "display_model_info",
    "display_checkpoint_loaded",

    # Training
    "display_training_header",
    "create_training_progress",
    "display_checkpoint_saved",
    "display_training_complete",
    "display_training_error",
    "display_info_message",

    # Image Capture
    "display_capture_banner",
    "display_capture_session_info",
    "display_capture_session_summary",
    "create_capture_progress",

    # General
    "print_message",
    "print_empty_line",
]
