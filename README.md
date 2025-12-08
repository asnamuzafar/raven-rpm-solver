# RAVEN Visual Reasoning Project

A neural-symbolic reasoning system for solving visual reasoning puzzles. This project implements:
- **I-RAVEN**: Raven's Progressive Matrices (abstract reasoning)
- **Sort-of-CLEVR**: Relational visual reasoning

## 📁 Project Structure

```
raven-rpm-solver/
├── train_iraven.py          # I-RAVEN training (Transformer, RelationNet, etc.)
├── train_clevr.py           # CLEVR training (Relation Network)
├── sort_of_clevr_generator.py  # Dataset generator for CLEVR
├── evaluate.py              # Evaluation script
├── raven_simulator.py       # Interactive Streamlit demo
├── config.py                # Configuration settings
│
├── models/                  # All neural network models
│   ├── encoder.py           # Visual encoders (ResNet, EfficientNet, DINO)
│   ├── baselines.py         # RelationNet, CNN-Direct, Symbolic, Hybrid
│   ├── reasoner.py          # Transformer & MLP reasoners
│   ├── relation_network_clevr.py  # CLEVR Relation Network
│   ├── tokenizer.py         # Symbolic attribute predictor
│   └── ...                  # Other experimental models
│
├── utils/                   # Utilities
│   ├── dataset.py           # Dataset loading
│   └── evaluation.py        # Metrics
│
├── data/                    # Datasets (generated)
│   ├── iraven_large/        # I-RAVEN puzzles
│   └── sort_of_clevr/       # Sort-of-CLEVR
│
├── saved_models/            # Trained model checkpoints
├── archive/                 # Old experimental code
└── EXPERIMENTS.md           # Experiment log with results
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train on Sort-of-CLEVR (Recommended - High Accuracy)
```bash
# Generate dataset
python train_clevr.py --generate --num_images 5000

# Train Relation Network (achieves 64%+ accuracy)
python train_clevr.py --model rn --epochs 30

# Compare with baseline
python train_clevr.py --model baseline --epochs 20
```

### 3. Train on I-RAVEN (Challenging)
```bash
# First generate I-RAVEN dataset (see setup.sh)
./setup_iraven.sh large

# Train different models
python train_iraven.py --model transformer --epochs 20
python train_iraven.py --model relation_net --epochs 20
python train_iraven.py --model neuro_symbolic --epochs 30
python train_iraven.py --model cnn_direct --epochs 20
```

---

## 📊 Results Summary

### Sort-of-CLEVR (Visual Relational Reasoning)
| Model | Test Accuracy | Relational | Non-Relational |
|-------|---------------|------------|----------------|
| **Relation Network** | **64.7%** | 64.6% | 64.9% |
| CNN Baseline | 52.6% | 54.2% | 50.9% |

### I-RAVEN (Abstract Reasoning)
| Model | Val Accuracy | Notes |
|-------|--------------|-------|
| **Spatial Conv** | **34.0%** | Best I-RAVEN model |
| Neuro-Symbolic | 32.9% | Rule-aware reasoner |
| Transformer | 26.9% | |
| RelationNet | 26.2% | |
| CNN-Direct | ~15% | Baseline |

*Note: I-RAVEN SOTA is 92.9% (requires specialized architectures)*

---

## 🎯 Command Line Options

### train_iraven.py
```bash
python train_iraven.py [OPTIONS]

Options:
  --model {transformer,relation_net,neuro_symbolic,cnn_direct}
  --encoder {resnet,simple}     Visual encoder type
  --freeze_encoder              Freeze pretrained weights
  --epochs N                    Number of epochs (default: 20)
  --batch_size N                Batch size (default: 32)
  --lr RATE                     Learning rate (default: 1e-4)
  --dropout RATE                Dropout rate (default: 0.4)
  --data_dir PATH               Dataset directory
```

### train_clevr.py
```bash
python train_clevr.py [OPTIONS]

Options:
  --generate                    Generate dataset first
  --num_images N                Images to generate (default: 5000)
  --model {rn,baseline}         Model type
  --epochs N                    Number of epochs (default: 30)
  --batch_size N                Batch size (default: 64)
  --lr RATE                     Learning rate (default: 1e-4)
```

---

## 🎮 Interactive Demo
```bash
streamlit run raven_simulator.py
```

---

## 📚 Key Findings

1. **Relation Network beats CNN** on relational tasks (12% improvement)
2. **I-RAVEN is extremely difficult** - requires specialized neuro-symbolic approaches
3. **Sort-of-CLEVR** is better for demonstrating relational reasoning capabilities
4. **Overfitting** is the main challenge on I-RAVEN (50%+ train-val gap)

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed experiment logs.

---

## 📝 References

- [RAVEN Dataset](https://github.com/WellyZhang/RAVEN)
- [I-RAVEN](https://github.com/husheng12345/I-RAVEN)
- [Relation Networks](https://arxiv.org/abs/1706.01427)
- [Sort-of-CLEVR](https://arxiv.org/abs/1706.01427)
