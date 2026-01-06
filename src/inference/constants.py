"""
Shared constants for Modal deployment.

These paths are used by both the deploy script and the Modal app.
"""

# Modal volume name
MODAL_VOLUME_NAME = "semantic-recommender-model"

# Paths within the Modal volume (relative to /model mount point)
MODEL_DIR = "semantic-recommender"
CATALOGUE_FILE = "catalogue.jsonl"
SEMANTIC_IDS_FILE = "semantic_ids.jsonl"

# Full paths as seen by the Modal container
MODAL_MOUNT_PATH = "/model"
MODAL_MODEL_PATH = f"{MODAL_MOUNT_PATH}/{MODEL_DIR}"
MODAL_CATALOGUE_PATH = f"{MODAL_MOUNT_PATH}/{CATALOGUE_FILE}"
MODAL_SEMANTIC_IDS_PATH = f"{MODAL_MOUNT_PATH}/{SEMANTIC_IDS_FILE}"
