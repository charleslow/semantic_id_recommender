"""
Inference module for semantic ID recommender.

Use `from src.inference.modal_app import Recommender` to import the Modal app.
Use `from src.inference.constants import ...` for shared constants.
"""

# Only export constants by default - modal_app has heavy dependencies
from .constants import (
    CATALOGUE_FILE,
    MODAL_CATALOGUE_PATH,
    MODAL_MODEL_PATH,
    MODAL_MOUNT_PATH,
    MODAL_SEMANTIC_IDS_PATH,
    MODAL_VOLUME_NAME,
    MODEL_DIR,
    SEMANTIC_IDS_FILE,
)

__all__ = [
    "CATALOGUE_FILE",
    "MODEL_DIR",
    "MODAL_CATALOGUE_PATH",
    "MODAL_MODEL_PATH",
    "MODAL_MOUNT_PATH",
    "MODAL_SEMANTIC_IDS_PATH",
    "MODAL_VOLUME_NAME",
    "SEMANTIC_IDS_FILE",
]
