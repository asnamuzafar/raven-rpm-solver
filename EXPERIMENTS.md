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

### 8. Stronger Data Augmentation + Light Dropout (Latest)
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
- **Status:** Promising, needs more epochs to see if val continues improving

---

## Key Findings

### What WORKS:
1. **Pretrained ResNet-18 encoder** - produces diverse features (cosine sim < 0.99)
2. **Training encoder end-to-end** - ImageNet features alone aren't enough
3. **Low learning rate (1e-4)** - stable training
4. **No label smoothing** - allows sharper predictions
5. **Light dropout (0.1)** - any more prevents learning

### What DOESN'T WORK:
1. **Training CNN from scratch** - features collapse to near-identical
2. **BatchNorm in custom encoder** - causes feature collapse
3. **Freezing encoder** - features not discriminative for RAVEN
4. **Differential learning rates** - breaks training
5. **High dropout (0.3+)** - prevents learning entirely
6. **Large batch sizes (256)** - not enough gradient updates

### Current Best Model:
- **Architecture:** ResNet-18 (pretrained) + Simple Context-Choice Scorer
- **Train Accuracy:** 78%
- **Val Accuracy:** 18.2%
- **Status:** Overfitting needs to be addressed

---

## Recommended Next Steps

1. **Early stopping** - Stop training around epoch 5-6 when val accuracy peaks
2. **Reduce encoder capacity** - Try ResNet-10 or smaller
3. **More training data** - Use full RAVEN dataset instead of medium
4. **Try different reasoner architectures:**
   - Wild Relation Network (WReN)
   - CoPINet style contrastive learning
   - Attention-based reasoner

---

## Architecture Summary

### Current Encoder (ResNet-18):
- Pretrained on ImageNet
- Modified first conv for grayscale input
- Output: 512-dim features per panel

### Current Reasoner:
```
Context Encoder: Linear(512*8 → 512) → LayerNorm → ReLU → Linear(512 → 256)
Choice Encoder: Linear(512 → 256) → LayerNorm → ReLU
Scorer: Linear(512 → 256) → ReLU → Dropout → Linear(256 → 128) → ReLU → Linear(128 → 1)
```

### Data:
- 8400 train, 2800 val, 2800 test samples
- 16 images per puzzle (8 context + 8 choices)
- 160x160 grayscale images

