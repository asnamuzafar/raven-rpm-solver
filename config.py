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
HIDDEN_DIM = 256  # Reduced from 512 to limit capacity
NUM_HEADS = 4     # Reduced from 8
NUM_LAYERS = 2    # Reduced from 4 to prevent overfitting
DROPOUT = 0.4     # Increased from 0.1 for regularization
NUM_CHOICES = 8

# ===== Training =====
BATCH_SIZE = 16
EPOCHS = 30       # More epochs with early stopping
LEARNING_RATE = 3e-4  # Slightly higher LR for faster convergence
WEIGHT_DECAY = 0.05   # Increased from 0.01 for regularization
LABEL_SMOOTHING = 0.1  # Prevent overconfident predictions
FREEZE_ENCODER = True  # Freeze pretrained weights to prevent overfitting
PATIENCE = 7           # Early stopping patience
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

