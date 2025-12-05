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

### 10. Structure-Aware RelationNet (raven_large)
**Config:**
- ResNet-18 encoder (pretrained, unfrozen)
- **New RelationNet architecture:** Row/Column/Diagonal relations
- `BATCH_SIZE = 32`
- `LEARNING_RATE = 1e-4`
- Dataset: **raven_large (21,000 train samples)**

**Result:** ✅ WORKING
- **Train Acc: 51.0%**
- **Val Acc: 23.6%**
- Best val_loss: 1.7292 (at epoch 5)
- **Key insight:** Structure-aware design + larger dataset broke 20% barrier

---

### 11. I-RAVEN Medium Comparison (All Models)
**Config:**
- Dataset: **I-RAVEN medium (8,400 train samples)**
- I-RAVEN fixes "shortcut" issues in original RAVEN
- 5 epochs, all other settings same

**Results:**
| Model | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| MLP-Relational | 29.0% | **23.4%** | Best on medium |
| RelationNet | 26.6% | 20.6% | Good |
| Transformer | 28.8% | 19.8% | Good |
| CNN-Direct | 12.9% | 13.2% | ❌ No learning (expected) |

**Key insight:** I-RAVEN is easier to learn than RAVEN with same data size!

---

### 12. I-RAVEN Large (All Models) ✅ NEW BEST!
**Config:**
- Dataset: **I-RAVEN large (21,000 train samples)**
- 10 epochs, `BATCH_SIZE = 32`, `LEARNING_RATE = 1e-4`

**Results:**
| Model | Train Acc | Val Acc | Best Val Loss |
|-------|-----------|---------|---------------|
| **Transformer** | 42.7% | **26.9%** ✅ | 1.5911 |
| RelationNet | 44.0% | 26.2% | 1.6000 |
| MLP-Relational | 46.4% | 25.8% | 1.5990 |
| CNN-Direct | 13.1% | 11.4% | ❌ No learning |

**Key findings:**
- **Transformer wins on I-RAVEN large!** (26.9%)
- All relational models perform similarly (~26%)
- CNN-Direct confirms relational reasoning is required
- Val accuracy peaked around epoch 7-9, then declined (overfitting)

---

## Key Findings

### What WORKS:
1. **Pretrained ResNet-18 encoder** - produces diverse features (cosine sim < 0.99)
2. **Training encoder end-to-end** - ImageNet features alone aren't enough
3. **Low learning rate (1e-4)** - stable training
4. **No label smoothing** - allows sharper predictions
5. **Light dropout (0.1)** - any more prevents learning
6. **Structure-aware reasoning** - Modeling rows/columns/diagonals explicitly helps!
7. **More training data** - large (21k) significantly better than medium (8.4k)
8. **I-RAVEN over RAVEN** - I-RAVEN is fairer and models learn better on it

### What DOESN'T WORK:
1. **Training CNN from scratch** - features collapse to near-identical
2. **BatchNorm in custom encoder** - causes feature collapse
3. **Freezing encoder** - features not discriminative for RAVEN
4. **Differential learning rates** - breaks training
5. **High dropout (0.3+)** - prevents learning entirely
6. **Large batch sizes (256)** - not enough gradient updates
7. **Naive pairwise relations** - All-pairs RN is slow and loses grid structure
8. **CNN-Direct (no relational reasoning)** - proves RPM requires relational reasoning

### Current Best Model:
- **Architecture:** ResNet-18 (pretrained) + RuleAwareReasoner (Neuro-Symbolic)
- **Dataset:** I-RAVEN large (21,000 train samples)
- **Train Accuracy:** 44.0%
- **Val Accuracy:** 32.9% ✅ NEW BEST
- **Status:** 6% improvement over previous best (26.9%)!

---

## New: Contrastive Learning Approach (Pending Training)

### 13. ContrastiveReasoner with Multi-Loss Training
**Config:**
- ResNet-18 encoder (pretrained, unfrozen)
- **New ContrastiveReasoner architecture**
- **Combined loss:** CE + Contrastive + Ranking + Consistency
- Dataset: I-RAVEN large (21,000 train samples)

**New Components:**
- `models/contrastive_losses.py` - Multiple loss functions
- `models/reasoner_v2.py` - ContrastiveReasoner & DualContrastReasoner
- `train_contrastive.py` - Multi-loss training script

**Loss Components:**
- **ContrastiveLoss (0.5)** - Pull correct answer close, push wrong answers away
- **RankingLoss (0.3)** - Margin-based with hard negative mining
- **ConsistencyLoss (0.2)** - Row/column/diagonal pattern consistency

**Expected Results:**
- Target Val Acc: 50-70%
- Less overfitting due to contrastive regularization
- Better discrimination between similar choices

**To Run:**
```bash
python train_contrastive.py --data_dir ./data/iraven_large --epochs 15 --model contrastive
```

---

### 14. Neuro-Symbolic Reasoner with Supervised Attributes ✅ NEW BEST!
**Config:**
- ResNet-18 encoder (pretrained, unfrozen)
- **New RuleAwareReasoner architecture**
- **Supervised attribute extraction** from ground-truth `meta_matrix`
- **Rule prediction heads** (Constant/Progression/Distribute/Arithmetic)
- Multi-task loss: CE + 0.5 × Attribute Loss
- Dataset: I-RAVEN large (21,000 train samples)
- 15 epochs, `BATCH_SIZE = 32`, `LEARNING_RATE = 1e-4`

**Results:**
| Epoch | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 16.3% | 19.4% | Fast initial learning |
| 5 | 34.0% | 30.0% | Good progress |
| 10 | 44.0% | **32.9%** ✅ | **Best** |
| 15 | 48.8% | 32.4% | Slight overfit |

**Key Findings:**
- **32.9% validation accuracy** - beats previous best (26.9%) by **6%**!
- Supervised attribute extraction provides strong learning signal
- Rule prediction helps but overfitting still occurs after epoch 10
- Train accuracy reaches 48.8% (much higher than baseline)

**To Run:**
```bash
python train_neuro_symbolic.py --data_dir ./data/iraven_large --epochs 15 --model neuro_symbolic
```

---

## Recommended Next Steps

1. **Try higher attribute loss weight** - `--lambda_attr 1.0` to reduce overfitting
2. **Add data augmentation** - More aggressive augmentation
3. **Ensemble with baseline** - Combine neuro-symbolic with transformer predictions
4. **Per-configuration analysis** - Check which puzzle types improve most
5. **Fine-tune rule prediction** - Add auxiliary rule loss

---

## Architecture Summary

### Current Encoder (ResNet-18):
- Pretrained on ImageNet
- Modified first conv for grayscale input
- Output: 512-dim features per panel

### Best Reasoner (Transformer/Context-Choice Scorer):
```
Context Encoder: Linear(512*8 → 512) → LayerNorm → ReLU → Linear(512 → 256)
Choice Encoder: Linear(512 → 256) → LayerNorm → ReLU
Scorer: Linear(512 → 256) → ReLU → Dropout → Linear(256 → 128) → ReLU → Linear(128 → 1)
```

### Alternative Reasoners:
- **RelationNet:** Row/Column/Diagonal relations → Aggregator → Score (26.2%)
- **MLP-Relational:** Row/Column MLPs → Combine → Score (25.8%)

### Data (I-RAVEN large):
- 21,000 train, 7,000 val, 7,000 test samples
- 16 images per puzzle (8 context + 8 choices)
- 160x160 grayscale images
- 7 puzzle configurations (center_single, left_right, up_down, etc.)

