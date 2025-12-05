"""
Configuration settings for RAVEN RPM Solver
"""
import torch
from pathlib import Path

# ===== Dataset Type =====
# Options: "raven" or "iraven"
# I-RAVEN is a bias-corrected version of RAVEN that prevents shortcut learning
DATASET_TYPE = "iraven"

# ===== Dataset Size =====
# Options: "small", "medium", "large"
DATASET_SIZE = "large"

# ===== Paths =====
DATA_DIR = Path(f"./data/{DATASET_TYPE}_{DATASET_SIZE}")  # Use medium dataset for better training
MODELS_DIR = Path("./saved_models")
RESULTS_DIR = Path("./results")

# ===== Device =====
# Prefer MPS (Apple Silicon), then CUDA, then CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ===== Model Architecture =====
FEATURE_DIM = 512  # ResNet-18 output dimension
HIDDEN_DIM = 256   # Hidden dim for reasoning MLP
NUM_HEADS = 8      # Unused (for API compatibility)
NUM_LAYERS = 3     # Unused (for API compatibility)
DROPOUT = 0.1      # Light dropout
NUM_CHOICES = 8
USE_SIMPLE_ENCODER = False  # Use pretrained ResNet encoder

# ===== Training =====
BATCH_SIZE = 32         # Smaller batch for memory
EPOCHS = 8              # Stop early to prevent overfitting (best val was around epoch 5-6)
LEARNING_RATE = 1e-4    # This worked before
ENCODER_LR = 1e-4       # Same as main (not used now)
WEIGHT_DECAY = 1e-4     # Light regularization (original)
LABEL_SMOOTHING = 0.0   # No label smoothing
FREEZE_ENCODER = False  # Train encoder
PATIENCE = 5            # Early stopping
NUM_WORKERS = 2

# ===== Dataset =====
IMAGE_SIZE = 160
NUM_PANELS = 16  # 8 context + 8 choices

# ===== Symbolic Attributes =====
SHAPE_CLASSES = ['triangle', 'square', 'pentagon', 'hexagon', 'circle']
SIZE_CLASSES = ['small', 'medium', 'large']
COLOR_CLASSES = ['dark', 'gray', 'light', 'white']
COUNT_CLASSES = [str(i) for i in range(1, 10)]
POSITION_CLASSES = ['top-left', 'top', 'top-right', 'left', 'center', 
                    'right', 'bottom-left', 'bottom', 'bottom-right']

# ===== Random Seed =====
SEED = 42

