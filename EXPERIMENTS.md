# RAVEN RPM Solver - Experiment Log

## Problem Summary
Training a deep learning model to solve RAVEN progressive matrix puzzles. The task is an 8-way classification problem (random chance = 12.5%).

---

## Experiments Tried

### 1. Initial Setup - Simple CNN Encoder + Transformer Reasoner (FROZEN encoder)
**Config:**
- `FREEZE_ENCODER = True`
- `BATCH_SIZE = 256`
- `LEARNING_RATE = 3e-4`
- Simple CNN encoder (trained from scratch)

**Result:** ❌ FAILED
- Train Acc: ~12% (random)
- Val Acc: ~8% (worse than random)
- **Issue:** Frozen encoder features not discriminative enough

---

### 2. Simple CNN Encoder (Unfrozen) + MLP Reasoner
**Config:**
- `FREEZE_ENCODER = False`
- `BATCH_SIZE = 32`
- `LEARNING_RATE = 1e-3`
- `USE_SIMPLE_ENCODER = True`
- BatchNorm in encoder

**Result:** ❌ FAILED
- Train Acc: ~12%
- Val Acc: ~12%
- **Issue:** Encoder producing nearly identical features (cosine similarity > 0.998)
- **Root cause:** BatchNorm causing feature collapse

---

### 3. Simple CNN with GroupNorm (instead of BatchNorm)
**Config:**
- GroupNorm instead of BatchNorm in encoder
- `feature_dim = 128`

**Result:** ❌ FAILED
- Features still too similar (cosine sim 0.98+)
- Model not learning

---

### 4. Pretrained ResNet-18 Encoder (Unfrozen) ✅ BEST SO FAR
**Config:**
- `USE_SIMPLE_ENCODER = False` (use ResNet-18 pretrained on ImageNet)
- `FREEZE_ENCODER = False`
- `BATCH_SIZE = 16`
- `LEARNING_RATE = 1e-4`
- `WEIGHT_DECAY = 1e-4`
- `LABEL_SMOOTHING = 0.0`
- `DROPOUT = 0.1`

**Result:** ✅ PARTIALLY WORKED (but overfitting)
- **Train Acc: 78.1%** ⬆️
- **Val Acc: 18.2%** ⬆️ (improved from 12.5% random!)
- **Issue:** Severe overfitting (train >> val)
- ResNet features have better diversity (cosine sim 0.93-0.99)

---

### 5. Pretrained ResNet-18 with FROZEN Encoder
**Config:**
- `FREEZE_ENCODER = True`
- `LEARNING_RATE = 1e-3`
- Only train reasoner

**Result:** ❌ FAILED
- Train Acc: ~12%
- Val Acc: ~13%
- **Issue:** Frozen ImageNet features alone not suitable for RAVEN

---

### 6. Differential Learning Rates (Low for encoder, High for reasoner)
**Config:**
- `ENCODER_LR = 1e-5`
- `LEARNING_RATE = 5e-4` (for reasoner)
- `FREEZE_ENCODER = False`

**Result:** ❌ FAILED
- Train Acc: ~12%
- Val Acc: ~8%
- **Issue:** Encoder not learning fast enough

---

### 7. High Dropout for Regularization
**Config:**
- `DROPOUT = 0.3`
- Additional dropout layers in reasoner (up to 0.6)

**Result:** ❌ FAILED
- Train Acc: ~12%
- Val Acc: ~3-5%
- **Issue:** Too much dropout preventing learning

---

### 8. Stronger Data Augmentation + Light Dropout
**Config:**
- `DROPOUT = 0.1` (reduced from 0.3)
- `LEARNING_RATE = 1e-4`
- `WEIGHT_DECAY = 1e-4`
- Stronger augmentation (noise, brightness, contrast all 0.5 prob)
- Removed extra dropout layers from reasoner

**Result:** ✅ WORKING (5 epochs)
- **Train Acc: 27.2%** ⬆️
- **Val Acc: 16.0%** ⬆️ (above random!)
- Gap is smaller than experiment #4 (less overfitting)

---

### 9. Transformer Reasoner (15 epochs, raven_medium)
**Config:**
- ResNet-18 encoder (pretrained, unfrozen)
- Transformer/Context-Choice Scorer reasoner
- `BATCH_SIZE = 32`
- `LEARNING_RATE = 1e-4`
- Dataset: raven_medium (8400 train)

**Result:** ✅ WORKING (matches exp #4)
- **Train Acc: 68.6%**
- **Val Acc: 18.4%** (best val_loss: 1.9508)
- Val accuracy plateaued at ~18% after epoch 5
- **Issue:** Severe overfitting, val loss increasing while train improves

---

### 10. Structure-Aware RelationNet (raven_large) ✅ NEW BEST!
**Config:**
- ResNet-18 encoder (pretrained, unfrozen)
- **New RelationNet architecture:** Row/Column/Diagonal relations
- `BATCH_SIZE = 32`
- `LEARNING_RATE = 1e-4`
- Dataset: **raven_large (21,000 train samples)**

**Result:** ✅ BEST SO FAR!
- **Train Acc: 51.0%**
- **Val Acc: 23.6%** ⬆️ NEW RECORD!
- Best val_loss: 1.7292 (at epoch 5)
- **Key insight:** Structure-aware design + larger dataset broke 20% barrier
- Still overfitting but less severe than Transformer

---

## Key Findings

### What WORKS:
1. **Pretrained ResNet-18 encoder** - produces diverse features (cosine sim < 0.99)
2. **Training encoder end-to-end** - ImageNet features alone aren't enough
3. **Low learning rate (1e-4)** - stable training
4. **No label smoothing** - allows sharper predictions
5. **Light dropout (0.1)** - any more prevents learning
6. **Structure-aware reasoning** - Modeling rows/columns/diagonals explicitly helps!
7. **More training data** - raven_large (21k) significantly better than raven_medium (8.4k)

### What DOESN'T WORK:
1. **Training CNN from scratch** - features collapse to near-identical
2. **BatchNorm in custom encoder** - causes feature collapse
3. **Freezing encoder** - features not discriminative for RAVEN
4. **Differential learning rates** - breaks training
5. **High dropout (0.3+)** - prevents learning entirely
6. **Large batch sizes (256)** - not enough gradient updates
7. **Naive pairwise relations** - All-pairs RN is slow and loses grid structure

### Current Best Model:
- **Architecture:** ResNet-18 (pretrained) + Structure-Aware RelationNet
- **Dataset:** raven_large (21,000 train samples)
- **Train Accuracy:** 51.0%
- **Val Accuracy:** 23.6% ✅
- **Status:** Best validation accuracy achieved!

---

## Recommended Next Steps

1. **Early stopping** - Best val was at epoch 5-6, consider stopping earlier
2. **More regularization** - Try stronger dropout or weight decay to reduce overfitting
3. **Hybrid approach** - Combine RelationNet with symbolic reasoning
4. **Try different architectures:**
   - Wild Relation Network (WReN)
   - CoPINet style contrastive learning
   - Multi-scale reasoning

---

## Architecture Summary

### Current Encoder (ResNet-18):
- Pretrained on ImageNet
- Modified first conv for grayscale input
- Output: 512-dim features per panel

### Best Reasoner (Structure-Aware RelationNet):
```
Row Relations:    Linear(512*3 → 256) → LayerNorm → ReLU → Linear(256 → 256) → ReLU  (×3 rows)
Column Relations: Linear(512*3 → 256) → LayerNorm → ReLU → Linear(256 → 256) → ReLU  (×3 cols)
Diagonal Relations: Linear(512*3 → 256) → LayerNorm → ReLU → Linear(256 → 256) → ReLU  (×2 diags)
Aggregator: Linear(256*8 → 512) → LayerNorm → ReLU → Dropout → Linear(512 → 256) → ReLU → Linear(256 → 1)
```

### Data (raven_large):
- 21,000 train, 7,000 val, 7,000 test samples
- 16 images per puzzle (8 context + 8 choices)
- 160x160 grayscale images

