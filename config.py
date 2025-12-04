"""
Configuration settings for RAVEN RPM Solver
"""
import torch
from pathlib import Path

# ===== Paths =====
DATA_DIR = Path("./data/raven_medium")  # Use medium dataset for better training
MODELS_DIR = Path("./saved_models")
RESULTS_DIR = Path("./results")

# ===== Device =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Model Architecture =====
FEATURE_DIM = 512  # ResNet-18 output dimension
HIDDEN_DIM = 256   # Hidden dim for reasoning MLP
NUM_HEADS = 8      # Unused (for API compatibility)
NUM_LAYERS = 3     # Unused (for API compatibility)
DROPOUT = 0.1      # Light dropout
NUM_CHOICES = 8
USE_SIMPLE_ENCODER = False  # Use pretrained ResNet encoder

# ===== Training =====
BATCH_SIZE = 16         # Smaller batch for memory
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

