# RAVEN RPM Solver

A neural-symbolic reasoning system for solving Raven's Progressive Matrices (RPM) puzzles. This project implements a complete pipeline from visual encoding to interactive demonstration, as specified in the project requirements.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Interactive Simulator](#interactive-simulator)
- [Architecture](#architecture)

---

## 🎯 Overview

This system solves RPM puzzles using multiple reasoning approaches:

- **Deep Learning**: Transformer and MLP-based relational reasoning
- **Symbolic**: Rule-based logical inference (constant, progression, XOR, distribution)
- **Hybrid**: Combined neural-symbolic approach

### Implemented Stages

| Stage | Description | File(s) |
|-------|-------------|---------|
| A | Visual Encoder (ResNet-18) | `models/encoder.py` |
| B | Symbolic Tokenizer | `models/tokenizer.py` |
| C | Deep Learning Reasoner | `models/reasoner.py` |
| D | Baseline Models | `models/baselines.py` |
| E | Evaluation Module | `utils/evaluation.py` |
| F | Interactive Simulator | `raven_simulator.py` |

---

## 📁 Project Structure

```
raven-rpm-solver/
├── config.py              # Configuration settings
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── test_setup.py          # Setup verification
├── raven_simulator.py     # Streamlit interactive app
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── models/                # Neural network models
│   ├── __init__.py
│   ├── encoder.py         # Stage A: ResNet-18 visual encoder
│   ├── tokenizer.py       # Stage B: Symbolic attribute predictor
│   ├── reasoner.py        # Stage C: Transformer & MLP reasoners
│   ├── baselines.py       # Stage D: CNN-Direct, RelationNet, Symbolic
│   └── full_model.py      # Combined encoder + reasoner
│
├── utils/                 # Utilities
│   ├── __init__.py
│   ├── dataset.py         # Dataset loading
│   └── evaluation.py      # Stage E: Evaluation metrics
│
├── data/                  # Dataset (generated)
│   └── raven_medium/      # Default dataset (14,000 puzzles)
│       ├── center_single/
│       ├── distribute_four/
│       └── ...
│
├── saved_models/          # Trained models (after training)
└── results/               # Evaluation results (after evaluation)
```

---

## 🚀 Quick Start

```bash
# 1. Clone/navigate to project
cd /path/to/project

# 2. Run setup script (creates venv, installs deps, generates data)
chmod +x setup.sh
./setup.sh                # Default: medium dataset (14,000 puzzles)
# OR: ./setup.sh small    # Small dataset (1,400 puzzles) - quick testing
# OR: ./setup.sh large    # Large dataset (70,000 puzzles) - best results

# 3. Activate environment
source venv/bin/activate

# 4. Train models (with early stopping and regularization)
python train.py --epochs 30

# 5. Evaluate
python evaluate.py

# 6. Launch simulator
streamlit run raven_simulator.py
```

---

## 🔧 Detailed Setup

### Prerequisites

- Python 3.9+
- pip or conda
- ~2GB disk space for data and models

### Step 1: Create Virtual Environment

**Option A: Using venv**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Option B: Using conda**
```bash
conda create -n raven python=3.10 -y
conda activate raven
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy, pandas, matplotlib, seaborn
- streamlit >= 1.28.0
- tqdm, Pillow, scipy

### Step 3: Generate RAVEN Dataset

**Recommended: Use the setup script** (handles all patches automatically):

```bash
./setup.sh medium    # Generates 14,000 puzzles (recommended)
```

Dataset size options:
| Size | Samples | Total Puzzles | Use Case |
|------|---------|---------------|----------|
| `small` | 200/config | 1,400 | Quick testing |
| `medium` | 2000/config | 14,000 | **Recommended** |
| `large` | 5000/config | 35,000 | Best accuracy |

**Manual generation** (if needed):

```bash
# Clone and patch RAVEN repository
git clone https://github.com/WellyZhang/RAVEN.git
cd RAVEN
python -m lib2to3 -w src/

# Apply patches (see setup.sh for details)
# ... patches for scipy, XML encoding, range(), RLE ...

# Generate dataset
mkdir -p ../data/raven_medium
PYTHONPATH=src python -m dataset.main --num-samples 2000 --save-dir ../data/raven_medium
```

### Step 4: Verify Setup

```bash
python test_setup.py
```

Expected output:
```
✓ PyTorch version: 2.x.x
✓ CUDA available: True/False
✓ Models imported successfully
✓ Utils imported successfully
✓ Config loaded (device=cpu/cuda, batch_size=16)
✓ Model forward pass works (output shape: torch.Size([2, 8]))
✓ Data found: data/raven_medium (14000 files)
==================================================
Setup test complete!
==================================================
```

---

## 🏋️ Training

### Train All Models

```bash
python train.py --data_dir ./data/raven_medium --epochs 30
```

### Anti-Overfitting Features

The training pipeline includes several regularization techniques:

| Feature | Default | Description |
|---------|---------|-------------|
| **Frozen Encoder** | `True` | Freezes pretrained ResNet weights |
| **Label Smoothing** | `0.1` | Prevents overconfident predictions |
| **Early Stopping** | `patience=7` | Stops if val loss doesn't improve |
| **Dropout** | `0.4` | High dropout for regularization |
| **Data Augmentation** | `True` | Noise, brightness, contrast on training |

### Options

```bash
python train.py --help

Options:
  --data_dir         Path to RAVEN data directory (default: ./data/raven_medium)
  --save_dir         Directory to save models (default: ./saved_models)
  --epochs           Number of training epochs (default: 30)
  --batch_size       Batch size (default: 16)
  --lr               Learning rate (default: 3e-4)
  --models           Models to train (default: transformer mlp cnn_direct relation_net)
  --seed             Random seed (default: 42)
  --freeze_encoder   Freeze pretrained encoder weights (default: True)
  --label_smoothing  Label smoothing factor (default: 0.1)
  --patience         Early stopping patience (default: 7)
```

### Examples

```bash
# Quick test (5 epochs, small dataset)
python train.py --data_dir ./data/raven_small --epochs 5

# Train only Transformer with more epochs
python train.py --models transformer --epochs 50

# Train with unfrozen encoder (if you have lots of data)
python train.py --data_dir ./data/raven_large --freeze_encoder False --epochs 50
```

### Training Output

Models are saved to `./saved_models/`:
- `transformer_model.pth`
- `mlp_relational_model.pth`
- `cnn_direct_model.pth`
- `relationnet_model.pth`
- `training_histories.json`

---

## 📊 Evaluation

### Run Evaluation

```bash
python evaluate.py --data_dir ./data/raven_medium --models_dir ./saved_models
```

### Options

```bash
python evaluate.py --help

Options:
  --data_dir      Path to RAVEN data directory
  --models_dir    Directory containing trained models
  --results_dir   Directory to save results (default: ./results)
  --visualize     Number of sample predictions to visualize (default: 5)
```

### Evaluation Metrics

As specified in goal.md, the evaluation includes:

1. **Accuracy** - Overall test accuracy
2. **Generalization** - Per-configuration accuracy (7 puzzle types)
3. **Sample Efficiency** - Performance with limited training data
4. **Computational Cost** - Inference time, parameter count
5. **Rule Traces** - Symbolic reasoning explanations

### Output Files

Results are saved to `./results/`:
- `comparison_table.csv` - Model comparison metrics
- `evaluation_results.json` - Detailed results
- `comparison_plots.png` - Accuracy and performance charts
- `generalization_plots.png` - Per-configuration accuracy
- `prediction_sample_*.png` - Sample predictions

---

## 🎮 Interactive Simulator

### Launch

```bash
streamlit run raven_simulator.py
```

Opens at `http://localhost:8501`

### Features

- **Upload Puzzles**: Load .npz files from RAVEN dataset
- **Multiple Models**: Compare Transformer, MLP, CNN-Direct, Symbolic
- **Rule Traces**: See symbolic reasoning explanations
- **Visualizations**: View 3×3 grid and answer choices

### Usage

1. Select a model from the sidebar
2. Upload a `.npz` puzzle file (from `data/raven_medium/*/`)
3. View the puzzle visualization
4. See model predictions and explanations

---

## 🏗️ Architecture

### Stage A: Visual Encoder

```
Input: 16 images (160×160 grayscale)
       ↓
ResNet-18 (modified for 1-channel input)
       ↓
Output: 16 × 512-dim feature vectors
```

### Stage B: Tokenizer

```
Input: 512-dim feature vector
       ↓
MLP Classifier
       ↓
Output: Symbolic attributes
        - Shape: triangle, square, pentagon, hexagon, circle
        - Size: small, medium, large
        - Color: dark, gray, light, white
        - Count: 1-9
        - Position: 9 grid positions
```

### Stage C: Transformer Reasoner

```
Input: 8 context features + 1 choice feature
       ↓
Positional Encoding + Type Embedding
       ↓
2-layer Transformer Encoder (4 heads, dropout=0.4)
       ↓
Score Head (MLP)
       ↓
Output: Score for this choice
```

### Stage D: Baseline Models

| Model | Description |
|-------|-------------|
| CNN-Direct | Concatenate all features → MLP classifier |
| RelationNet | Pairwise relations → Aggregate → Score |
| Symbolic | Rule detection (constant, progression, XOR, distribution) |
| Hybrid | DL scores + Symbolic scores (weighted) |

---

## 📈 Expected Results

### With Medium Dataset (14,000 samples, 30 epochs)

| Model | Test Accuracy | vs Random |
|-------|--------------|-----------|
| Transformer | ~25-40% | 2.0-3.2x |
| MLP-Relational | ~20-35% | 1.6-2.8x |
| CNN-Direct | ~15-20% | 1.2-1.6x |
| RelationNet | ~18-30% | 1.4-2.4x |
| Random | 12.5% | 1.0x |

### Tips for Higher Accuracy

| Improvement | How |
|-------------|-----|
| **More data** | `./setup.sh large` (70,000 puzzles) |
| **More epochs** | `--epochs 50` or `--epochs 100` |
| **Unfreeze encoder** | `--freeze_encoder False` (needs large dataset) |
| **GPU training** | Much faster, allows larger batches |

---

## 🔍 Troubleshooting

### Common Issues

**1. "No .npz files found"**
```bash
# Regenerate dataset using setup script
./setup.sh medium
```

**2. "CUDA out of memory"**
```bash
# Reduce batch size
python train.py --batch_size 8
```

**3. "Module not found"**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or: conda activate raven
pip install -r requirements.txt
```

**4. Streamlit not opening**
```bash
# Check if port 8501 is available
streamlit run raven_simulator.py --server.port 8502
```

**5. Model overfitting (high train acc, low val acc)**
```bash
# Use larger dataset
./setup.sh large
python train.py --data_dir ./data/raven_large --epochs 50

# Or increase regularization
python train.py --freeze_encoder True --label_smoothing 0.15
```

---

## 📚 References

- [RAVEN Dataset](https://github.com/WellyZhang/RAVEN)
- [Raven's Progressive Matrices](https://en.wikipedia.org/wiki/Raven%27s_Progressive_Matrices)

---

## 📝 License

This project is for educational purposes.

