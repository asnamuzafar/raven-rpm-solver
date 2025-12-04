"""
Configuration settings for RAVEN RPM Solver
"""
import torch
from pathlib import Path

# ===== Paths =====
DATA_DIR = Path("./data/raven_small")
MODELS_DIR = Path("./saved_models")
RESULTS_DIR = Path("./results")

# ===== Device =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Model Architecture =====
FEATURE_DIM = 512
HIDDEN_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.1
NUM_CHOICES = 8

# ===== Training =====
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
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

