# RAVEN RPM Solver - Experiment Log

## Problem Summary
Training a deep learning model to solve RAVEN progressive matrix puzzles. The task is an 8-way classification problem (random chance = 12.5%).

**Published SOTA on I-RAVEN: 92.9%** ([Raven Solver, ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S0020025523003870))

---

## Our Best Results Summary

| Model | Val Accuracy | Notes |
|-------|--------------|-------|
| **Neuro-Symbolic (RuleAwareReasoner)** | **32.9%** | Best overall |
| RelationNet | 26.2% | Structure-aware |
| Transformer | 26.9% | Competitive |
| EfficientNet Encoder | 21.1% | Alternative encoder |
| DINOv2 Encoder | 13.7% | Did not converge |

---

## Experiments

### Experiment 1: Baseline CNN Encoder (Failed)
**Config:** Simple CNN encoder, frozen
**Result:** ~12% (random chance) - BatchNorm caused feature collapse

---

### Experiment 2: Pretrained ResNet-18 + Transformer
**Config:** ResNet-18 (ImageNet pretrained), unfrozen, Transformer reasoner
**Dataset:** I-RAVEN large (21,000 train)
**Result:** 26.9% val accuracy

---

### Experiment 3: Structure-Aware RelationNet
**Config:** RelationNet with row/column/diagonal relation modules
**Dataset:** I-RAVEN large
**Result:** 26.2% val accuracy

---

### Experiment 4: Neuro-Symbolic Reasoner ✅ BEST
**Config:**
- ResNet-18 encoder (pretrained, unfrozen)
- RuleAwareReasoner with supervised attribute extraction
- Multi-task loss: CrossEntropy + 0.5 × Attribute Loss
- Dataset: I-RAVEN large (21,000 train)
- 15 epochs, batch_size=32, lr=1e-4

**Results:**
| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 5 | 34.0% | 30.0% |
| 10 | 44.0% | **32.9%** ✅ |
| 15 | 48.8% | 32.4% |

**Command:**
```bash
python train_neuro_symbolic.py --data_dir ./data/iraven_large --epochs 15
```

---

### Experiment 5: Pretrained Vision Encoders
**Goal:** Test alternative pretrained encoders vs ResNet-18

**Results:**
| Encoder | Val Acc (5 epochs) | Params | Notes |
|---------|-------------------|--------|-------|
| ResNet-18 | 26-27% | 11M | **Best encoder** |
| EfficientNet-B0 | 21.1% | 4.6M | Fast but lower accuracy |
| DINOv2-ViT-S | 13.7% (stuck) | 22M | Failed to converge |

**Finding:** ResNet-18 remains the best encoder. Self-supervised features (DINOv2) don't transfer well to abstract shapes.

---

## Key Findings

### What Works:
1. **Pretrained ResNet-18 encoder** - Best feature extractor for RAVEN
2. **End-to-end training** - Fine-tuning encoder is essential
3. **Neuro-Symbolic approach** - Rule prediction + attribute extraction (32.9%)
4. **Structure-aware reasoning** - Explicit row/column/diagonal modeling
5. **I-RAVEN over RAVEN** - Fixes distribution biases
6. **More training data** - 21k samples significantly better than 8.4k

### What Doesn't Work:
1. **Frozen encoders** - Features not discriminative enough
2. **DINOv2/CLIP/EfficientNet** - Don't outperform ResNet-18
3. **High dropout (>0.3)** - Prevents learning
4. **Training CNN from scratch** - Feature collapse

---

## Gap to SOTA

| Metric | Our Best | Published SOTA |
|--------|----------|----------------|
| I-RAVEN Accuracy | 32.9% | 92.9% |

The ~60% gap indicates SOTA methods use more sophisticated approaches:
- Perception-to-reasoning pipelines
- Rule-based symbolic reasoning
- Answer set programming

---

## Code Structure

```
train.py                    # Main training script
train_neuro_symbolic.py     # Neuro-symbolic model (best: 32.9%)
models/
  encoder.py                # All encoders (ResNet, EfficientNet, DINOv2, CLIP)
  baselines.py              # RelationNet, CNNDirect, HybridReasoner
  reasoner.py               # Transformer, MLP reasoners
  rule_reasoner.py          # RuleAwareReasoner (neuro-symbolic)
```

---

## Recommended Commands

```bash
# Best model (Neuro-Symbolic) - 32.9% accuracy
python train_neuro_symbolic.py --data_dir ./data/iraven_large --epochs 15

# Standard models
python train.py --models relation_net --data_dir ./data/iraven_large --epochs 10
python train.py --models transformer --data_dir ./data/iraven_large --epochs 10

# Test different encoders
python train.py --encoder efficientnet --models relation_net --epochs 10
python train.py --encoder dinov2 --models relation_net --epochs 5
```
