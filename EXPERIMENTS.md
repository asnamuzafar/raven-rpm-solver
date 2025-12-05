# RAVEN RPM Solver - Experiment Log

## Problem Summary
Training a deep learning model to solve RAVEN progressive matrix puzzles. The task is an 8-way classification problem (random chance = 12.5%).

**Published SOTA on I-RAVEN: 92.9%** ([Raven Solver, ScienceDirect 2023](https://www.sciencedirect.com/science/article/abs/pii/S0020025523003870))

---

## Our Best Results Summary

| Model | Val Accuracy | Notes |
|-------|--------------|-------|
| **Neuro-Symbolic (RuleAwareReasoner)** | **32.9%** | Best overall |
| RelationNet + Attribute Supervision | 28.0% | Improved baseline |
| Transformer | 26.9% | Competitive |
| EfficientNet Encoder | 21.1% | Alternative encoder test |
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

### Experiment 5: RelationNet with Attribute Supervision
**Config:**
- RelationNet + SupervisedAttributeHead (shared module)
- Multi-task loss: CrossEntropy + 0.5 × Attribute Loss
- Added `--use_attr` flag to train.py

**Results:**
| LR | Epochs | Val Acc |
|----|--------|---------|
| 1e-4 | 15 | 27.7% |
| 3e-3 | 5 | **28.0%** ✅ |

**Command:**
```bash
python train.py --models relation_net --data_dir ./data/iraven_large --use_attr --epochs 15
```

---

### Experiment 6: Pretrained Vision Encoders
**Goal:** Test alternative pretrained encoders (EfficientNet, DINOv2) vs ResNet-18

**Results:**
| Encoder | Val Acc (5 epochs) | Params | Notes |
|---------|-------------------|--------|-------|
| EfficientNet-B0 | 21.1% | 4.6M | Fast, comparable |
| DINOv2-ViT-S | 13.7% (stuck) | 22M | Failed to converge |
| ResNet-18 (baseline) | 27-28% | 11M | Best encoder |

**Finding:** ResNet-18 remains the best encoder for RAVEN-style puzzles. DINOv2's self-supervised features don't transfer well to abstract shapes.

**Command:**
```bash
python train_pretrained.py --encoder efficientnet --epochs 10
```

---

## Key Findings

### What Works:
1. **Pretrained ResNet-18 encoder** - Best feature extractor for RAVEN
2. **End-to-end training** - Fine-tuning encoder is essential
3. **Supervised attribute extraction** - Ground-truth attributes improve learning
4. **Structure-aware reasoning** - Explicit row/column/diagonal modeling helps
5. **I-RAVEN over RAVEN** - I-RAVEN fixes distribution biases
6. **More training data** - 21k samples significantly better than 8.4k

### What Doesn't Work:
1. **Frozen encoders** - Features not discriminative enough
2. **DINOv2/CLIP** - Self-supervised features don't transfer to abstract reasoning
3. **High dropout (>0.3)** - Prevents learning
4. **Training from scratch** - Feature collapse with BatchNorm

---

## Gap to SOTA

| Metric | Our Best | Published SOTA |
|--------|----------|----------------|
| I-RAVEN Accuracy | 32.9% | 92.9% |

The significant gap (~60%) indicates that our architecture lacks the sophisticated reasoning capabilities of SOTA methods like the "Raven Solver" which uses:
- Perception-to-reasoning pipeline
- Rule-based symbolic reasoning
- Answer set programming

---

## Code Structure

```
train.py                    # Main training script (supports --use_attr)
train_neuro_symbolic.py     # Neuro-symbolic model training
train_pretrained.py         # Pretrained encoder comparison
models/
  attributes.py             # Shared SupervisedAttributeHead
  pretrained_encoders.py    # CLIP, DINOv2, EfficientNet encoders
  rule_reasoner.py          # RuleAwareReasoner
  baselines.py              # RelationNet (with attribute support)
```

---

## Recommended Commands

```bash
# Best model (Neuro-Symbolic)
python train_neuro_symbolic.py --data_dir ./data/iraven_large --epochs 15

# RelationNet with attribute supervision
python train.py --models relation_net --data_dir ./data/iraven_large --use_attr --epochs 15

# Test different encoders
python train_pretrained.py --encoder efficientnet --epochs 10
```
