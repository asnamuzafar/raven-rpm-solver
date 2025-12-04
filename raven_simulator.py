"""
Stage F: RAVEN RPM Puzzle Solver - Interactive Simulator

An interactive Streamlit application for visualizing and solving
Raven's Progressive Matrices puzzles using neural reasoning.

Run with: streamlit run raven_simulator.py

Features as per goal.md:
- Upload or select RPM puzzles
- View 3×3 grid
- Observe how each reasoning engine processes it
- Display predicted answer from each model
- Show attention maps for DL models
- Show transparent rule traces for symbolic models
"""
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
try:
    from models import create_model, load_model, SymbolicReasoner, SymbolicTokenizer
    from models.encoder import RAVENFeatureExtractor
    from config import DEVICE, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Warning: Could not import models: {e}")


def load_trained_models(models_dir: Path, device: str) -> dict:
    """
    Load all trained models from the saved_models directory.
    Returns dict mapping model name to (model, model_type).
    """
    models = {}
    models_dir = Path(models_dir)
    
    model_type_map = {
        'transformer': 'transformer',
        'mlp_relational': 'mlp',
        'mlp-relational': 'mlp',
        'cnn_direct': 'cnn_direct',
        'cnn-direct': 'cnn_direct',
        'relationnet': 'relation_net',
        'relation_net': 'relation_net',
        'hybrid': 'hybrid'
    }
    
    if not models_dir.exists():
        return models
    
    for model_file in models_dir.glob("*_model.pth"):
        try:
            checkpoint = torch.load(model_file, map_location=device)
            model_name = checkpoint.get('model_name', model_file.stem)
            
            # Determine model type from filename
            model_type = 'transformer'
            for key, mtype in model_type_map.items():
                if key in model_file.stem.lower().replace('-', '_'):
                    model_type = mtype
                    break
            
            model = create_model(model_type=model_type, pretrained_encoder=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            models[model_name] = {
                'model': model,
                'type': model_type,
                'val_acc': checkpoint.get('best_val_acc', 0),
                'val_loss': checkpoint.get('best_val_loss', 0)
            }
        except Exception as e:
            print(f"Warning: Could not load {model_file}: {e}")
    
    return models


def run_inference(model, imgs: np.ndarray, device: str) -> tuple:
    """
    Run inference on a puzzle using the model.
    Returns (predicted_idx, probabilities)
    """
    # Prepare input: normalize to [0, 1] and add batch dimension
    x = torch.from_numpy(imgs.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(logits.argmax(dim=1).item())
    
    return pred_idx, probs


def get_attention_weights(model, imgs: np.ndarray, device: str, choice_idx: int = 0) -> np.ndarray:
    """
    Extract attention weights from transformer model for visualization.
    Returns attention matrix or None if not a transformer.
    """
    x = torch.from_numpy(imgs.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        # Get features from encoder
        ctx_feat, choice_feat = model.encoder(x)
        
        # Check if reasoner has attention method
        if hasattr(model.reasoner, 'get_attention_weights'):
            attn_weights = model.reasoner.get_attention_weights(ctx_feat, choice_feat, choice_idx)
            if attn_weights:
                # Return last layer attention
                return attn_weights[-1][0].cpu().numpy()  # (num_heads, seq_len, seq_len)
    
    return None


class SymbolicAnalyzer:
    """
    Rule-based analyzer for puzzle patterns.
    Implements rules from goal.md: constant, progression, XOR, distribution.
    Provides transparent rule traces for interpretability.
    """
    
    def __init__(self):
        self.symbolic_reasoner = SymbolicReasoner() if MODELS_AVAILABLE else None
    
    def analyze_puzzle(self, imgs: np.ndarray) -> dict:
        """
        Analyze pixel patterns to infer rules.
        Returns dict with detected rules and explanations.
        """
        results = {
            'rules_detected': [],
            'explanations': [],
            'row_analysis': {},
            'confidence': 0.0
        }
        panels = [imgs[i] for i in range(8)]
        
        # Compute panel statistics
        means = [float(p.mean()) for p in panels]
        stds = [float(p.std()) for p in panels]
        
        results['explanations'].append("**Row-wise Analysis:**")
        
        # Analyze Row 0 (panels 0, 1, 2)
        row0_means = means[0:3]
        row0_diff1 = row0_means[1] - row0_means[0]
        row0_diff2 = row0_means[2] - row0_means[1]
        results['row_analysis']['row0'] = {
            'means': row0_means,
            'diffs': [row0_diff1, row0_diff2]
        }
        
        # Analyze Row 1 (panels 3, 4, 5)
        row1_means = means[3:6]
        row1_diff1 = row1_means[1] - row1_means[0]
        row1_diff2 = row1_means[2] - row1_means[1]
        results['row_analysis']['row1'] = {
            'means': row1_means,
            'diffs': [row1_diff1, row1_diff2]
        }
        
        # Check CONSTANT rule (similar intensities)
        try:
            row0_var = np.var(row0_means)
            row1_var = np.var(row1_means)
            if row0_var < 50 and row1_var < 50:
                results['rules_detected'].append('CONSTANT')
                results['explanations'].append(
                    f"  ✓ **CONSTANT rule**: Row intensities are stable "
                    f"(Row0 var={row0_var:.1f}, Row1 var={row1_var:.1f})"
                )
                results['confidence'] += 0.3
        except:
            pass
        
        # Check PROGRESSION rule (arithmetic sequence)
        try:
            prog_threshold = 20
            if abs(row0_diff2 - row0_diff1) < prog_threshold and abs(row1_diff2 - row1_diff1) < prog_threshold:
                results['rules_detected'].append('PROGRESSION')
                results['explanations'].append(
                    f"  ✓ **PROGRESSION rule**: Intensities change systematically "
                    f"(Row0: {row0_diff1:.1f}→{row0_diff2:.1f}, Row1: {row1_diff1:.1f}→{row1_diff2:.1f})"
                )
                results['confidence'] += 0.3
        except:
            pass
        
        # Check DISTRIBUTION rule (all different)
        try:
            row0_unique = len(set([round(m, -1) for m in row0_means]))
            row1_unique = len(set([round(m, -1) for m in row1_means]))
            if row0_unique == 3 and row1_unique == 3:
                results['rules_detected'].append('DISTRIBUTION')
                results['explanations'].append(
                    f"  ✓ **DISTRIBUTION rule**: Each position has unique intensity"
                )
                results['confidence'] += 0.2
        except:
            pass
        
        # Check for XOR-like patterns (correlation analysis)
        try:
            corr01 = np.corrcoef(panels[0].flatten(), panels[1].flatten())[0, 1]
            corr12 = np.corrcoef(panels[1].flatten(), panels[2].flatten())[0, 1]
            if corr01 < 0.5 and corr12 < 0.5:
                results['rules_detected'].append('XOR')
                results['explanations'].append(
                    f"  ✓ **XOR rule**: Panels show combination pattern "
                    f"(corr01={corr01:.2f}, corr12={corr12:.2f})"
                )
                results['confidence'] += 0.2
        except:
            pass
        
        if not results['rules_detected']:
            results['explanations'].append(
                "  ⚠️ No clear rule detected from pixel analysis - using neural reasoning"
            )
        else:
            results['explanations'].append(
                f"\n**Confidence**: {results['confidence']*100:.0f}%"
            )
            
        return results
    
    def predict_answer(self, imgs: np.ndarray, rules: list) -> tuple:
        """
        Predict the answer based on detected rules.
        Returns (predicted_idx, explanations, choice_scores)
        """
        panels = imgs[:8]
        choices = imgs[8:]
        
        # Compute row 2 partial stats (panels 6, 7)
        row2_means = [float(panels[6].mean()), float(panels[7].mean())]
        
        # Predict missing value based on rules
        predictions = []
        explanations = []
        
        if 'CONSTANT' in rules:
            # Predict same as row average
            pred_mean = np.mean(row2_means)
            explanations.append(f"CONSTANT: expecting intensity ~{pred_mean:.1f}")
            predictions.append(('constant', pred_mean))
        
        if 'PROGRESSION' in rules:
            # Predict continuation of sequence
            diff = row2_means[1] - row2_means[0]
            pred_mean = row2_means[1] + diff
            explanations.append(f"PROGRESSION: expecting intensity ~{pred_mean:.1f} (diff={diff:.1f})")
            predictions.append(('progression', pred_mean))
        
        # Score each choice
        choice_scores = []
        for i, choice in enumerate(choices):
            choice_mean = float(choice.mean())
            score = 0
            for rule_type, pred_val in predictions:
                # Score based on how close the choice is to prediction
                diff = abs(choice_mean - pred_val)
                score += max(0, 100 - diff) / 100
            choice_scores.append((i, score, choice_mean))
        
        # Best choice is highest score
        if choice_scores:
            choice_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx = choice_scores[0][0]
            return best_idx, explanations, choice_scores
        
        return 0, ["No prediction possible"], []


def plot_attention_heatmap(attn_weights: np.ndarray, title: str = "Attention Weights"):
    """
    Plot attention weights as a heatmap.
    """
    if attn_weights is None:
        return None
    
    # Average over heads if multiple
    if len(attn_weights.shape) == 3:
        attn = attn_weights.mean(axis=0)
    else:
        attn = attn_weights
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = [f'P{i}' for i in range(8)] + ['Choice']
    
    sns.heatmap(
        attn, 
        ax=ax, 
        cmap='Blues', 
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='.2f',
        vmin=0,
        vmax=1
    )
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    
    return fig


# ===== Streamlit App Configuration =====
st.set_page_config(
    page_title="RAVEN Puzzle Solver",
    page_icon="🧩",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 1rem;
}
.stMetric {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
}
.success-box {
    background-color: #d4edda;
    border: 2px solid #28a745;
    padding: 1rem;
    border-radius: 10px;
    color: #155724;
}
.error-box {
    background-color: #f8d7da;
    border: 2px solid #dc3545;
    padding: 1rem;
    border-radius: 10px;
    color: #721c24;
}
.info-box {
    background-color: #d1ecf1;
    border: 2px solid #17a2b8;
    padding: 1rem;
    border-radius: 10px;
    color: #0c5460;
}
</style>
""", unsafe_allow_html=True)

# ===== Header =====
st.markdown('<h1 class="main-header">🧩 RAVEN RPM Puzzle Solver</h1>', unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #666; font-size: 1.1rem;'>
Interactive demonstration of neural reasoning on Raven's Progressive Matrices
</p>
""", unsafe_allow_html=True)

# ===== Load Models =====
@st.cache_resource
def get_trained_models():
    """Load trained models (cached for performance)"""
    if not MODELS_AVAILABLE:
        return {}
    return load_trained_models(Path("./saved_models"), DEVICE)

trained_models = get_trained_models()

# ===== Sidebar =====
st.sidebar.header("⚙️ Settings")

# Build model selection list based on available trained models
model_options = []
if "Transformer" in trained_models or any("transformer" in k.lower() for k in trained_models):
    model_options.append("Transformer (Primary)")
if "MLP-Relational" in trained_models or any("mlp" in k.lower() for k in trained_models):
    model_options.append("MLP-Relational")
if "CNN-Direct" in trained_models or any("cnn" in k.lower() for k in trained_models):
    model_options.append("CNN-Direct")
if "RelationNet" in trained_models or any("relation" in k.lower() for k in trained_models):
    model_options.append("Relation Network")

model_options.append("Symbolic Analyzer")
if len([k for k in trained_models if k]) >= 2:
    model_options.append("Compare All")

# Default to available options or fallback
if not model_options:
    model_options = ["Symbolic Analyzer"]

model_choice = st.sidebar.selectbox(
    "Select Reasoning Model",
    model_options
)

show_probabilities = st.sidebar.checkbox("Show Probability Distribution", value=True)
show_attention = st.sidebar.checkbox("Show Attention Maps (Transformer)", value=True)
show_explanation = st.sidebar.checkbox("Show Rule Explanation", value=True)

st.sidebar.markdown("---")

# Show loaded models info
if trained_models:
    st.sidebar.markdown("### 🤖 Loaded Models")
    for name, info in trained_models.items():
        acc = info.get('val_acc', 0)
        st.sidebar.markdown(f"- **{name}**: {acc:.1%} acc")
else:
    st.sidebar.warning("No trained models found. Run `train.py` first or use Symbolic Analyzer.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This simulator demonstrates neural reasoning 
on Raven's Progressive Matrices (RPM) puzzles.

**Models:**
- 🤖 **Transformer**: Attention-based reasoning
- 🔗 **MLP-Relational**: Row/column patterns
- 📊 **CNN-Direct**: Feature classification
- 📐 **Symbolic**: Rule-based logic (constant, progression, XOR, distribution)
""")

# ===== Helper Functions =====
def get_model_for_choice(model_choice: str, trained_models: dict):
    """Get the appropriate model for the user's selection."""
    choice_lower = model_choice.lower()
    
    for name, info in trained_models.items():
        name_lower = name.lower()
        if "transformer" in choice_lower and "transformer" in name_lower:
            return info['model'], name
        if "mlp" in choice_lower and ("mlp" in name_lower or "relational" in name_lower):
            return info['model'], name
        if "cnn" in choice_lower and "cnn" in name_lower:
            return info['model'], name
        if "relation" in choice_lower and "relation" in name_lower:
            return info['model'], name
    
    # Fallback to first available model
    if trained_models:
        first_key = list(trained_models.keys())[0]
        return trained_models[first_key]['model'], first_key
    
    return None, None

# ===== Main Content =====
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload Puzzle")
    uploaded_file = st.file_uploader(
        "Upload a RAVEN puzzle (.npz file)",
        type=['npz'],
        help="Upload a .npz file from the RAVEN dataset"
    )

with col2:
    st.subheader("📊 Quick Stats")
    st.markdown("""
    <div class="info-box">
    <strong>Puzzle Format:</strong><br>
    • 3×3 context grid<br>
    • 8 answer choices<br>
    • 160×160 grayscale images
    </div>
    """, unsafe_allow_html=True)

# ===== Display Puzzle =====
if uploaded_file is not None:
    st.markdown("---")
    
    try:
        # Load puzzle
        data = np.load(uploaded_file)
        imgs = data["image"]
        target = int(data["target"])
        
        # Display context panels
        st.subheader("🖼️ Context Panels (3×3 Grid)")
        fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
        fig1.patch.set_facecolor('#f0f2f6')
        
        for i in range(8):
            row, col = divmod(i, 3)
            axes1[row, col].imshow(imgs[i], cmap='gray')
            axes1[row, col].set_title(f'Panel {i}', fontsize=10)
            axes1[row, col].axis('off')
        
        # Question mark for missing panel
        axes1[2, 2].text(0.5, 0.5, '?', fontsize=60, ha='center', va='center',
                        color='#667eea', fontweight='bold')
        axes1[2, 2].set_facecolor('#f8f9fa')
        axes1[2, 2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        
        # Display choices
        st.subheader("🎯 Answer Choices")
        fig2, axes2 = plt.subplots(2, 4, figsize=(12, 6))
        fig2.patch.set_facecolor('#f0f2f6')
        
        for i in range(8):
            row, col = divmod(i, 4)
            axes2[row, col].imshow(imgs[8+i], cmap='gray')
            title = f'Choice {i}'
            if i == target:
                title += ' ✓'
                for spine in axes2[row, col].spines.values():
                    spine.set_edgecolor('#28a745')
                    spine.set_linewidth(3)
            axes2[row, col].set_title(title, fontsize=10)
            axes2[row, col].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        
        # ===== Model Predictions =====
        st.markdown("---")
        st.subheader("🎯 Model Predictions")
        
        if model_choice == "Symbolic Analyzer":
            # Use symbolic rule-based analyzer
            analyzer = SymbolicAnalyzer()
            analysis = analyzer.analyze_puzzle(imgs)
            
            st.write("**Rule Detection (Symbolic Analysis):**")
            for exp in analysis['explanations']:
                st.markdown(exp)
            
            # Predict based on detected rules
            if analysis['rules_detected']:
                pred_idx, pred_explanations, choice_scores = analyzer.predict_answer(
                    imgs, analysis['rules_detected']
                )
                
                st.write("\n**Prediction Reasoning:**")
                for exp in pred_explanations:
                    st.write(f"• {exp}")
                
                if choice_scores:
                    st.write("\n**Choice Scores:**")
                    score_df = {f"Choice {i}": f"{s:.2f}" for i, s, _ in choice_scores[:4]}
                    st.write(score_df)
                
                confidence = analysis['confidence']
            else:
                # Fallback to first choice if no rules detected
                pred_idx = 0
                confidence = 0.0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Answer", f"Choice {pred_idx}")
            with col2:
                st.metric("Rules Found", len(analysis['rules_detected']))
            with col3:
                st.metric("Ground Truth", f"Choice {target}")
            
            if pred_idx == target:
                st.success("✅ Correct prediction!")
            else:
                st.error(f"❌ Incorrect. Correct answer is Choice {target}")
                
        elif model_choice == "Compare All":
            # Compare all available models
            st.write("**Comparison of All Models:**")
            
            results = {}
            
            # Run each trained model
            for name, info in trained_models.items():
                try:
                    model = info['model']
                    pred_idx, probs = run_inference(model, imgs, DEVICE)
                    results[name] = {
                        'pred': pred_idx,
                        'conf': float(probs[pred_idx]),
                        'correct': pred_idx == target
                    }
                except Exception as e:
                    st.warning(f"Error with {name}: {e}")
            
            # Add symbolic analyzer
            analyzer = SymbolicAnalyzer()
            analysis = analyzer.analyze_puzzle(imgs)
            if analysis['rules_detected']:
                sym_pred, _, _ = analyzer.predict_answer(imgs, analysis['rules_detected'])
            else:
                sym_pred = 0
            results['Symbolic'] = {
                'pred': sym_pred,
                'conf': analysis['confidence'],
                'correct': sym_pred == target
            }
            
            # Display results
            if results:
                cols = st.columns(len(results))
                for col, (name, res) in zip(cols, results.items()):
                    with col:
                        st.markdown(f"**{name}**")
                        st.metric("Prediction", f"Choice {res['pred']}")
                        st.metric("Confidence", f"{res['conf']:.1%}")
                        if res['correct']:
                            st.success("✓ Correct")
                        else:
                            st.error("✗ Wrong")
            
            # Summary
            st.markdown("---")
            correct_models = [n for n, r in results.items() if r['correct']]
            st.info(f"**Summary**: {len(correct_models)}/{len(results)} models predicted correctly")
            
        else:
            # Use selected neural model
            model, model_name = get_model_for_choice(model_choice, trained_models)
            
            if model is not None:
                pred_idx, probs = run_inference(model, imgs, DEVICE)
                confidence = float(probs[pred_idx])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Answer", f"Choice {pred_idx}")
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col3:
                    st.metric("Ground Truth", f"Choice {target}")
                
                if pred_idx == target:
                    st.success(f"✅ Correct prediction by {model_name}!")
                else:
                    st.error(f"❌ Incorrect. Model predicted Choice {pred_idx}, but correct answer is Choice {target}")
                
                # Show probability distribution
                if show_probabilities:
                    st.subheader("Probability Distribution")
                    fig3, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#28a745' if i == target else '#667eea' for i in range(8)]
                    bars = ax.bar(range(8), probs, color=colors, alpha=0.8)
                    
                    # Highlight predicted choice
                    bars[pred_idx].set_edgecolor('red')
                    bars[pred_idx].set_linewidth(3)
                    
                    ax.set_xlabel('Choice')
                    ax.set_ylabel('Probability')
                    ax.set_xticks(range(8))
                    ax.axhline(y=0.125, color='r', linestyle='--', alpha=0.5, label='Random (12.5%)')
                    ax.legend()
                    ax.set_title(f'{model_name} - Probability per Choice')
                    st.pyplot(fig3)
                    plt.close()
                
                # Show attention map for transformer
                if show_attention and "transformer" in model_choice.lower():
                    st.subheader("🔍 Attention Visualization")
                    attn_weights = get_attention_weights(model, imgs, DEVICE, pred_idx)
                    if attn_weights is not None:
                        fig_attn = plot_attention_heatmap(
                            attn_weights, 
                            f"Attention Weights (Choice {pred_idx})"
                        )
                        if fig_attn:
                            st.pyplot(fig_attn)
                            plt.close()
                            st.caption("Shows how much attention each panel pays to other panels when evaluating the selected choice.")
                    else:
                        st.info("Attention visualization not available for this model.")
            else:
                # Fallback: simulate prediction (when no trained models available)
                st.warning("⚠️ No trained model found. Please run `python train.py` first.")
                st.info("Showing simulated prediction for demonstration purposes.")
                
                np.random.seed(hash(uploaded_file.name) % 2**32)
                probs = np.random.dirichlet(np.ones(8) * 0.5)
                pred_idx = np.argmax(probs)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Answer", f"Choice {pred_idx} (simulated)")
                with col2:
                    st.metric("Confidence", f"{probs[pred_idx]:.1%}")
                with col3:
                    st.metric("Ground Truth", f"Choice {target}")
        
        # ===== Explanation =====
        if show_explanation:
            st.markdown("---")
            st.subheader("📝 How the Model Reasons")
            
            st.markdown("""
            **Reasoning Pipeline (per goal.md stages):**
            
            1. **Stage A - Visual Encoding**
               - Each panel is processed through ResNet-18 (CNN)
               - Produces 512-dimensional feature vectors
               
            2. **Stage B - Tokenization**
               - Features are converted to symbolic attributes
               - Predicts: shape, size, color, count, position
               
            3. **Stage C - Deep Learning Reasoning**
               - Transformer uses self-attention to find relationships
               - Analyzes row-wise and column-wise patterns
               - Scores each candidate answer
               
            4. **Stage D - Alternative Reasoners**
               - *CNN-Direct*: Simple classification baseline
               - *RelationNet*: Pairwise relational reasoning
               - *Symbolic*: Explicit rule-based logic
               - *Hybrid*: Combines DL + Symbolic
               
            5. **Rule Types Detected:**
               - *Constant*: Same attribute across row/column
               - *Progression*: Systematic change (+1, +2, +3)
               - *Distribution*: Each value appears once per row/column
               - *XOR*: Combination rule between elements
            """)
            
    except Exception as e:
        st.error(f"Error loading puzzle: {e}")
        st.info("Please upload a valid .npz file from the RAVEN dataset")

else:
    # Show instructions when no file is uploaded
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px;'>
        <h3>👆 Upload a RAVEN puzzle to get started</h3>
        <p style='color: #666;'>
        Supported format: .npz files from the RAVEN dataset<br>
        Each file contains a 3×3 puzzle with 8 answer choices
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show sample puzzles from data directory
    data_dirs = [Path("./data/raven_medium"), Path("./data/raven_small")]
    sample_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            sample_files.extend(list(data_dir.rglob("*.npz"))[:5])
    
    if sample_files:
        st.markdown("---")
        st.subheader("📁 Sample Puzzles Available")
        st.write("Found puzzle files in the data directory. Upload one to get started!")
        for f in sample_files[:5]:
            st.code(str(f))

# ===== Footer =====
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    <p><strong>RAVEN RPM Solver</strong> - Neural-Symbolic Reasoning System</p>
    <p>Built with PyTorch & Streamlit | Stages A-F Implementation (goal.md)</p>
</div>
""", unsafe_allow_html=True)
