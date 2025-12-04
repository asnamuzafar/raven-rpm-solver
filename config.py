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
FEATURE_DIM = 512
HIDDEN_DIM = 512   # Back to 512 for sufficient capacity
NUM_HEADS = 8      # Back to 8 heads
NUM_LAYERS = 3     # 3 layers (balanced)
DROPOUT = 0.2      # Moderate dropout (0.4 was too high)
NUM_CHOICES = 8

# ===== Training =====
BATCH_SIZE = 32
EPOCHS = 30       # More epochs with early stopping
LEARNING_RATE = 1e-4   # Standard LR (3e-4 was too aggressive with regularization)
WEIGHT_DECAY = 0.01    # Back to 0.01 (0.05 was too high)
LABEL_SMOOTHING = 0.1  # Keep label smoothing
FREEZE_ENCODER = True  # IMPORTANT: Freeze encoder to focus learning on reasoner
PATIENCE = 10          # More patience for learning
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

