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

### Experiment 7: SCL Baseline (Dec 7, 2024)
**Config:**
- SCL Reasoner (Row/Col attention + Contrastive Loss)
- ResNet-18 Encoder (pretrained)
- No attribute supervision
- Dataset: I-RAVEN large (35K samples)

**Results:**
| Epoch | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 15.1% | 16.7% | Learning started |
| 5 | 22.2% | 21.0% | Steady progress |
| 8 | 28.5% | **22.8%** | Peaked then stalled |

**Finding:** SCL architecture learns but overfits quickly without attribute supervision or stronger regularization. Accuracy (22.8%) is comparable to baseline models but below best neuro-symbolic (32.9%).

---

### Experiment 8: SCL + Attribute Supervision (Dec 7, 2024)
**Config:**
- SCL Reasoner
- Attribute Supervision (λ_attr=1.0)
- batch_size=64
- Dataset: I-RAVEN large

**Results:**
| Epoch | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 13.8% | 15.5% | Slow start |
| 5 | 21.6% | 21.2% | Plateaued early |
| 7 | 24.8% | 21.1% | Overfitting started |

**Finding:** Adding strong attribute supervision (λ=1.0) didn't help with batch_size=64. The model stalled at 21%. Might need larger batch size for proper contrastive learning or lower attribute weight.

---

### Experiment 9: SCL Large Batch (Dec 7, 2024)
**Config:**
- SCL Reasoner
- Batch Size = 128
- Attribute Supervision (λ=0.5)

**Results:**
| Epoch | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 13.5% | 15.2% | Slower start than batch 64 |
| 3 | 18.6% | 19.8% | Parity with batch 64 |
| 4 | 19.8% | 19.1% | Regression |

**Finding:** Large batch size didn't magically solve the issue. The SCL-inspired architecture itself might be too simple or needs components from the original paper (GNNs).

---

### Experiment 10: Neuro-Symbolic + SCL Contrastive Loss (Dec 7, 2024)
**Config:**
- Model: Neuro-Symbolic (RuleAwareReasoner)
- Loss: CrossEntropy + Attribute (0.5) + Contrastive (1.0)
- Dataset: I-RAVEN large (35K samples)
- Batch Size: 64

**Results:**
| Epoch | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 16.1% | 20.5% | Fast start |
| 5 | 27.1% | **27.3%** | Peaked early |
| 10 | 41.7% | 26.3% | Strong overfitting |

**Analysis:**
Adding contrastive loss improved convergence speed but did not beat the baseline 32.9%. The validation accuracy peaked at 27.3% and then degraded while training accuracy continued to climb (overfitting).
**CRITICAL FINDING:** `ResNetVisualEncoder` uses `avgpool` which suppresses spatial information. This likely prevents the model from solving "Position" based rules, acting as a hard ceiling on performance.

---

### Experiment 11: SPATIAL REASONING (Dec 7, 2024) - STOPPED
**Config:**
- Model: Neuro-Symbolic + SpatialResNet
- Encoder: **Spatial ResNet (No Pooling, CoordConv)**
- Loss: CrossEntropy + Contrastive (1.0) + Attribute (0.5)
- Dataset: I-RAVEN large

**Result:** Reached **23.3%** Val Acc at Epoch 10.
**Analysis:** Learning was too slow compared to baseline (which hit 27% at Ep 5). The SpatialAdapter (training from scratch) needs stronger supervision than just the contrastive loss to learn position mapping quickly.

### Experiment 12: SPATIAL + POSITION SUPERVISION (Dec 7, 2024) - FAILED
**Config:**
- Model: Neuro-Symbolic + SpatialResNet + **Position Head**
- Encoder: Spatial ResNet (No Pooling, CoordConv)
- Loss: CrossEntropy + Contrastive (1.0) + Attribute (1.0) incl. Position
- Dataset: I-RAVEN large

**Result:** Reached **27.1%** Val Acc.
**Analysis:** Severe Overfitting! Train Acc reached **75.6%**. The explicit position supervision worked too well—the model memorized the absolute position of every object in the training set but failed to generalize the rules to new positions. The "Spatial Adapter" (flattening grid to vector) breaks translational invariance.

### Experiment 13: SPATIAL CONVOLUTIONAL REASONER (Dec 7, 2024) - STOPPED
**Config:**
- Model: **Spatial Convolutional Reasoner** (Phase 5)
- Encoder: Spatial ResNet (Flatten=False) -> (B, 512, 5, 5)
- Reasoner: Conv2D over stacked panels (Translation Invariant)
- Loss: CrossEntropy + Contrastive (1.0)
- Dataset: I-RAVEN large

**Result:** Reached **34.0%** Val Acc (Peak).
**Analysis:** Massive Overfitting! Train Acc reached **90%**. The Convolutional Reasoner is powerful enough to solve the task (proving the architecture works), but it memorized the training set. It needs regularization.

### Experiment 14: CONV REASONER + ATTRIBUTE SUPERVISION (Dec 7, 2024) - RUNNING (Retry 4)
**Config:**
- Model: Spatial Convolutional Reasoner + **Aux Attribute Head**
- Loss: CrossEntropy + Contrastive (1.0) + **Attribute (1.0)**
- Dropout: **0.5** (Verified active in Conv Blocks)
- Dataset: I-RAVEN large

### Experiment 14: CONV REASONER + ATTRIBUTE SUPERVISION (Dec 7, 2024) - STOPPED
**Config:**
- Model: Spatial Convolutional Reasoner + **Aux Attribute Head**
- Loss: CrossEntropy + Contrastive (1.0) + **Attribute (1.0)**
- Dropout: **0.5** (Verified active in Conv Blocks)
- Dataset: I-RAVEN large

**Result:** Reached **24.9%** Val Acc (Peak) then degraded.
**Analysis:** FAILED. Regularization hurt the peak performance (down from 34% in Exp 13) but **did not stop overfitting** (Train reached 58% while Val stuck at 23%).
**Conclusion:** Simple regularization (dropout/aux loss) is not enough. The model is still memorizing "visual shortcuts" instead of "rules". We need **Structural Consistency Learning (SCL)** to force the model to verify that the answer choice follows the *same rule* as the context.




## Gap to SOTA

| Metric | Our Best | Published SOTA |
|--------|----------|----------------|
| I-RAVEN Accuracy | 32.9% | 92.9% |

The ~60% gap indicates SOTA methods use more sophisticated approaches:
- Perception-to-reasoning pipelines
- Rule-based symbolic reasoning
- Answer set programming

### Experiment 15: SCL + SPATIAL CONV + ATTR SUP (Dec 7, 2024) - RUNNING
**Config:**
- Model: Spatial Convolutional Reasoner + Aux Attr + **SCL Loss**
- Loss: CE + Contrastive + Attr + **Consistency `MSE(Row0, Row1)`**
- Dropout: 0.5
- Dataset: I-RAVEN large

**Hypothesis:** Structural Consistency Learning (SCL) is the key. By forcing the latent rule embedding of Row 0 to match Row 1, the model cannot just memorize visual patterns. It *must* extract a rule that is invariant across rows. This should solve the overfitting.

**Status:**
- Ep 1: 13.3% Train / 15.4% Val (SCL Loss 0.09) - Model learning constraints.
- Ep 2: 15.4% Train / 17.4% Val (SCL Loss 0.027) - Rules aligning.
Slow but steady. Overfitting is completely suppressed (Val > Train).

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
python train.py --encoder dinov2 --models relation_net --epochs 5
```

---

### Experiment 16: Rule-Aware Reasoner V2 (Spatial + SCL + Attr) (Dec 8, 2024) - FAILED
**Config:**
- **Model:** RuleAwareReasonerV2
- **Architecture:** Spatial Rule Encoder (3-layer Conv) + Attr Head + SCL w/ explicit Consistency Scoring
- **Loss:** CE + Contrastive(1.0) + SCL(1.0) + Attr(1.0)
- **Score:** Validity + Consistency (-MSE)
- **Dataset:** I-RAVEN large (21k train)

**Result:** Reached **24.2%** Val Acc at Epoch 17.
**Analysis:**
FAILED. The model followed the same pattern as Exp 13 and 14:
1.  Early epochs (1-5) showed promise with Val > Train.
2.  By Epoch 10, Train Acc skyrocketed (surpassing 30%) while Val Acc plateaued at 24%.
3.  By Epoch 18, Train Acc hit **60%** while Val Acc remained at **23-24%**.

**Conclusion:**
The "Rule Encoder" is too powerful/flexible. It is memorizing the specific visual features of the context panels instead of extracting abstract rules. Even with SCL forcing `Rule(Row0) == Rule(Row1)`, the encoder likely found a way to map "visual similarity" to the same embedding space rather than "logical rule". 

We must constrain the bottleneck even further or switch to a **scattering transform** or **hard-coded rule checks** (symbolic only) to prevent visual memorization.


### Experiment 17: Rule-Aware Reasoner V3 (Neuro-Symbolic)
- **Status:** FAILED
- **Date:** 2025-12-08
- **Description:** Pivoted to a strict Neuro-Symbolic approach. Trained a `PerceptionNet` to extract attributes (Type, Size, Color, etc.) and a deterministic `SymbolicReasoner` to predict answers.
- **Outcome:** Perception training plateaued at ~80% accuracy.
- **Analysis:**
    - The I-RAVEN metadata schema is highly variable. "Row 0" of the attribute matrix might be "Type" in `center_single` but "Number" in `distribute_four`.
    - Without a perfect schema mapping (which is complex to reverse-engineer), the supervised learning signals were noisy/incorrect.
- **Conclusion:** Disentangling perception from reasoning is too brittle without a clean, unified dataset schema. Reverting to End-to-End learning (V2) but focusing on architecture simplification to solve overfitting.

### Experiment 18: Regularized Rule-Aware Reasoner V2 (Dec 8, 2024) - PAUSED (Promising)
**Config:**
- **Model:** Regularized V2 (SimpleSpatialEncoder + RuleAwareReasonerV2)
- **Encoder:** 4-Layer Simple CNN (No Pretraining) -> AdaptiveAvgPool -> MLP
- **Regularization:** Dropout 0.5, Weight Decay 1e-4, Augmentation True
- **Loss:** CE + Con(1.0) + SCL(0.1) + Attr(0.5)
- **Batch Size:** 32

**Hypothesis:**
Replacing the powerful `SpatialResNet` with a simple CNN trained from scratch will prevent the massive overfitting seen in Exp 13-16. Reducing SCL weight allows the model to focus on task accuracy first.

**Status:**
- Ep 1: 14.5% Train / 15.5% Val (Good start, no overfitting)
- Ep 2: 15.5% Train / 16.3% Val (Stable growth)
- **Stopped manually to conclude session.**
