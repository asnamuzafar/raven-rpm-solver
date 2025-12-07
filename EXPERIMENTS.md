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

### Experiment 6: Medium Dataset Fast Iteration (Dec 7, 2024)
**Goal:** Fast experimentation on medium dataset (8,400 train samples) to find better architectures before scaling.

**Results on I-RAVEN Medium:**
| Model | Trainable Params | Val Accuracy | Train Acc | Notes |
|-------|------------------|--------------|-----------|-------|
| **RelationNet** | 12.1M | **23.5%** | 57.6% | Best on medium |
| Neuro-Symbolic (λ_attr=2.0) | 12.9M | 22.1% | 39.8% | Higher attr loss |
| Transformer | 13.7M | 21.4% | 51.5% | Similar to RN |
| Frozen ResNet | 1.8M | 18.3% | 19.0% | Reasoner-only |
| SimpleConv 128-dim | 2.0M | 13.7% | 12.2% | ❌ Too weak |

**Key Finding: Severe Overfitting**
All models show massive train-val gap (train 50-60% vs val 20-23%). This indicates:
1. Model capacity is too high for dataset size
2. Need better regularization or data augmentation
3. Dataset size matters significantly (medium 23% vs large 33%)

**Commands:**
```bash
# RelationNet (best on medium)
python train.py --models relation_net --data_dir ./data/iraven_medium --epochs 10

# Neuro-symbolic with higher attribute loss
python train_neuro_symbolic.py --data_dir ./data/iraven_medium --epochs 10 --lambda_attr 2.0

# Quick test with frozen encoder
python train_neuro_symbolic.py --data_dir ./data/iraven_medium --epochs 5 --freeze_encoder
```

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
