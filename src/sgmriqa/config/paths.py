"""Path constants for the evaluation framework."""

import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

DATA_ROOT = os.environ.get(
    "SGMRIQA_DATA_ROOT", os.path.join(PROJECT_ROOT, "data")
)
DATA_GENERATION_ROOT = os.environ.get(
    "SGMRIQA_DATA_GENERATION", os.path.join(PROJECT_ROOT, "data_generation")
)
DATA_PROCESSING_ROOT = os.environ.get(
    "SGMRIQA_DATA_PROCESSING", os.path.join(PROJECT_ROOT, "data_processing")
)

# ---------------------------------------------------------------------------
# GPT-4o generated Q&A files (for AR-Score and A-Score evaluation)
# ---------------------------------------------------------------------------
QA_DATA_DIR = os.path.join(DATA_GENERATION_ROOT, "qa_reasoning_data")

QA_FILES = {
    "val": {
        "brain_slice": os.path.join(QA_DATA_DIR, "brain", "gpt4o_brain_val_image_qa_image_qa_pairs.json"),
        "brain_volume": os.path.join(QA_DATA_DIR, "brain", "gpt4o_brain_val_volume_qa_volume_qa_pairs.json"),
        "knee_slice": os.path.join(QA_DATA_DIR, "knee", "gpt4o_knee_val_image_qa_image_qa_pairs.json"),
        "knee_volume": os.path.join(QA_DATA_DIR, "knee", "gpt4o_knee_val_volume_qa_volume_qa_pairs.json"),
    },
    "train": {
        "brain_slice": os.path.join(QA_DATA_DIR, "brain", "gpt4o_brain_train_image_qa_image_qa_pairs.json"),
        "brain_volume": os.path.join(QA_DATA_DIR, "brain", "gpt4o_brain_train_volume_qa_volume_qa_pairs.json"),
        "knee_slice": os.path.join(QA_DATA_DIR, "knee", "gpt4o_knee_train_image_qa_image_qa_pairs.json"),
        "knee_volume": os.path.join(QA_DATA_DIR, "knee", "gpt4o_knee_train_volume_qa_volume_qa_pairs.json"),
    },
}

# ---------------------------------------------------------------------------
# Volume metadata (ground truth bboxes for V-Score)
# ---------------------------------------------------------------------------
VAL_VOLUME_FILES = {
    "brain": os.path.join(DATA_PROCESSING_ROOT, "brain", "brain_val_volumes.json"),
    "knee": os.path.join(DATA_PROCESSING_ROOT, "knee", "knee_val_volumes.json"),
}

TRAIN_VOLUME_FILES = {
    "brain": os.path.join(DATA_PROCESSING_ROOT, "brain", "brain_train_volumes.json"),
    "knee": os.path.join(DATA_PROCESSING_ROOT, "knee", "knee_train_volumes.json"),
}

# ---------------------------------------------------------------------------
# Validation image directories (use val, NOT train)
# ---------------------------------------------------------------------------
VAL_IMAGE_DIRS = {
    "brain": os.path.join(DATA_ROOT, "val_labeled_raw_by_modality"),
    "knee": os.path.join(DATA_ROOT, "knee_val_labeled_raw"),
}

# ---------------------------------------------------------------------------
# Output directories — nested by eval level
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get(
    "SGMRIQA_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "outputs")
)

# Image-level outputs
IMAGE_INFERENCE_DIR = os.path.join(OUTPUT_DIR, "image_level", "inference")
IMAGE_EVALUATION_DIR = os.path.join(OUTPUT_DIR, "image_level", "evaluation")

# Video-level outputs
VIDEO_INFERENCE_DIR = os.path.join(OUTPUT_DIR, "video_level", "inference")
VIDEO_EVALUATION_DIR = os.path.join(OUTPUT_DIR, "video_level", "evaluation")

# Aggregate outputs
AGGREGATE_DIR = os.path.join(OUTPUT_DIR, "aggregate")

# Deprecated aliases (point to image_level for backwards compatibility)
INFERENCE_DIR = IMAGE_INFERENCE_DIR
EVALUATION_DIR = IMAGE_EVALUATION_DIR

# .env file for API keys
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")


def get_inference_dir(eval_mode: str) -> str:
    """Return the inference output directory for a given eval mode."""
    if eval_mode == "image":
        return IMAGE_INFERENCE_DIR
    elif eval_mode == "video":
        return VIDEO_INFERENCE_DIR
    else:
        raise ValueError(f"get_inference_dir expects 'image' or 'video', got '{eval_mode}'")


def get_evaluation_dir(eval_mode: str) -> str:
    """Return the evaluation output directory for a given eval mode."""
    if eval_mode == "image":
        return IMAGE_EVALUATION_DIR
    elif eval_mode == "video":
        return VIDEO_EVALUATION_DIR
    else:
        raise ValueError(f"get_evaluation_dir expects 'image' or 'video', got '{eval_mode}'")


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [
        OUTPUT_DIR,
        IMAGE_INFERENCE_DIR,
        IMAGE_EVALUATION_DIR,
        VIDEO_INFERENCE_DIR,
        VIDEO_EVALUATION_DIR,
        AGGREGATE_DIR,
    ]:
        os.makedirs(d, exist_ok=True)
