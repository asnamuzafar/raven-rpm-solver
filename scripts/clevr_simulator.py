"""
Sort-of-CLEVR Interactive Simulator
====================================
Professional demonstration tool for visual relational reasoning.
Run with: streamlit run scripts/clevr_simulator.py
"""

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from models.clevr.relation_network import SortOfCLEVRModel, BaselineCNN
    from datasets.sort_of_clevr_generator import COLORS, SHAPES, generate_image
    MODELS_AVAILABLE = True
except ImportError:
    # Soft pastel colors for a cleaner look
    COLORS = [
        ((255, 107, 107), 'coral'),      # soft red
        ((78, 205, 196), 'teal'),        # soft cyan
        ((107, 137, 255), 'periwinkle'), # soft blue
        ((255, 230, 109), 'butter'),     # soft yellow
        ((255, 159, 128), 'peach'),      # soft orange
        ((162, 155, 254), 'lavender'),   # soft purple
    ]
    SHAPES = ['circle', 'rectangle']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Page config
st.set_page_config(page_title="Sort-of-CLEVR Demo", page_icon="🔮", layout="wide")

# Force light theme with visible text
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #f8f9fa !important;
    }
    [data-testid="stHeader"] {
        background-color: #f8f9fa !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px !important;
    }
    #MainMenu, footer, .stDeployButton {display: none !important;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCTIONS
# ============================================================================

@st.cache_resource
def load_models(save_dir):
    models = {}
    save_dir = Path(save_dir)
    
    for name, filename in [('Relation Network', 'clevr_rn_best.pt'), ('CNN Baseline', 'clevr_baseline_best.pt')]:
        path = save_dir / filename
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=DEVICE)
                ModelClass = SortOfCLEVRModel if 'rn' in filename else BaselineCNN
                model = ModelClass().to(DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models[name] = {
                    'model': model,
                    'accuracy': checkpoint.get('val_acc', 0),
                    'rel_acc': checkpoint.get('val_rel_acc', 0),
                    'nonrel_acc': checkpoint.get('val_nonrel_acc', 0),
                }
            except Exception as e:
                pass
    return models


def generate_scene():
    if not MODELS_AVAILABLE:
        objects = [(np.random.randint(20, 108), np.random.randint(20, 108), i % 6, i % 2, 12) for i in range(6)]
        return None, objects
    return generate_image()


def plot_scene(objects, highlight_idx=None):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='#f8f9fa')
    ax.set_facecolor('#ffffff')
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)
    ax.set_aspect('equal')
    ax.axis('off')
    
    for i, obj in enumerate(objects):
        x, y, color_idx, shape_idx = obj[0], obj[1], obj[2], obj[3]
        size = obj[4] if len(obj) > 4 else 12
        color = tuple(c/255 for c in COLORS[color_idx][0])
        
        is_hl = highlight_idx is not None and i == highlight_idx
        ec = '#374151' if is_hl else 'none'  # no border except on highlight
        lw = 1.5 if is_hl else 0
        
        if SHAPES[shape_idx] == 'circle':
            ax.add_patch(plt.Circle((x, y), size, facecolor=color, edgecolor=ec, linewidth=lw))
        else:
            ax.add_patch(plt.Rectangle((x-size, y-size), size*2, size*2, facecolor=color, edgecolor=ec, linewidth=lw))
    
    plt.tight_layout()
    return fig


def create_question(objects, q_type):
    if not objects:
        return None, None, None, None
    
    idx = np.random.randint(0, len(objects))
    color_name = COLORS[objects[idx][2]][1]
    
    if q_type == 'relational':
        x1, y1 = objects[idx][0], objects[idx][1]
        nearest = min([i for i in range(len(objects)) if i != idx], 
                      key=lambda i: (objects[i][0]-x1)**2 + (objects[i][1]-y1)**2)
        return f"What shape is closest to the {color_name} object?", SHAPES[objects[nearest][3]], nearest, idx
    else:
        return f"What is the shape of the {color_name} object?", SHAPES[objects[idx][3]], idx, idx


def run_inference(model, img, question_tensor):
    model.eval()
    with torch.no_grad():
        if img is not None:
            img_t = torch.tensor(np.array(img).transpose(2,0,1).astype(np.float32)/255.0).unsqueeze(0)
        else:
            img_t = torch.randn(1, 3, 128, 128)
        logits = model(img_t.to(DEVICE), question_tensor.to(DEVICE))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        return int(logits.argmax(1).item()), probs


# ============================================================================
# MAIN APP
# ============================================================================

# Title - visible on any theme
st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700; color: #1f2937;">
        🔮 Sort-of-CLEVR
    </h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; color: #6b7280;">
        Visual Relational Reasoning Demonstration
    </p>
</div>
""", unsafe_allow_html=True)

# Horizontal line
st.markdown("<hr style='border: 1px solid #e5e7eb; margin: 0 0 1.5rem 0;'>", unsafe_allow_html=True)

# Load models
models = load_models(Path("./saved_models"))

# Controls
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 3])
with col_ctrl1:
    new_scene = st.button("🔄 New Scene")
with col_ctrl2:
    q_type_select = st.selectbox("Question", ["Relational", "Non-Relational"], label_visibility="collapsed")

# Scene state
if 'objects' not in st.session_state or new_scene:
    img, objects = generate_scene()
    st.session_state['objects'] = objects
    st.session_state['img'] = img

objects = st.session_state['objects']
q_type = 'relational' if q_type_select == "Relational" else 'non_relational'
question, answer, answer_idx, query_idx = create_question(objects, q_type)

# Main layout
st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
col_scene, col_qa = st.columns([1, 1], gap="large")

with col_scene:
    st.markdown("<p style='font-weight: 600; color: #374151; margin-bottom: 0.5rem;'>📸 SCENE</p>", unsafe_allow_html=True)
    
    fig = plot_scene(objects, highlight_idx=query_idx)
    st.pyplot(fig)
    plt.close()
    
    # Legend
    legend = " ".join([f"<span style='display:inline-flex;align-items:center;margin-right:12px;'>"
                       f"<span style='width:12px;height:12px;border-radius:50%;background:rgb{c[0]};margin-right:4px;'></span>"
                       f"<span style='color:#4b5563;font-size:13px;'>{c[1]}</span></span>" 
                       for c in COLORS])
    st.markdown(f"<div style='background:#f3f4f6;padding:10px;border-radius:8px;margin-top:10px;'>{legend}</div>", unsafe_allow_html=True)

with col_qa:
    st.markdown("<p style='font-weight: 600; color: #374151; margin-bottom: 0.5rem;'>❓ QUESTION & ANSWER</p>", unsafe_allow_html=True)
    
    if question:
        # Question box
        badge = "🔗 Relational" if q_type == 'relational' else "📍 Non-Relational"
        st.markdown(f"""
        <div style="background: #e0f2fe; border-radius: 12px; padding: 20px; margin-bottom: 15px; border: 1px solid #bae6fd;">
            <p style="margin: 0 0 8px 0; font-size: 12px; color: #0369a1; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">{badge}</p>
            <p style="margin: 0; font-size: 18px; color: #0c4a6e; font-weight: 600;">{question}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Ground truth
        st.markdown(f"""
        <div style="display: inline-block; background: #f3f4f6; border: 2px solid #e5e7eb; border-radius: 8px; padding: 8px 16px; margin-bottom: 20px;">
            <span style="color: #374151; font-weight: 600;">Ground Truth: </span>
            <span style="color: #1f2937; font-weight: 700;">{answer}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Model predictions
        if models:
            st.markdown("<p style='font-weight: 600; color: #374151; margin: 20px 0 10px 0;'>🤖 MODEL PREDICTIONS</p>", unsafe_allow_html=True)
            
            q_tensor = torch.zeros(1, 8)
            q_tensor[0, objects[query_idx][2]] = 1
            q_tensor[0, 6 if q_type == 'relational' else 7] = 1
            
            for name, info in models.items():
                pred_idx, probs = run_inference(info['model'], st.session_state['img'], q_tensor)
                pred = SHAPES[pred_idx] if pred_idx < len(SHAPES) else "?"
                correct = pred == answer
                conf = probs[pred_idx] * 100
                
                bg = "#10b981" if correct else "#ef4444"
                icon = "✓" if correct else "✗"
                
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; color: #1f2937;">{name}</span>
                        <span style="background: {bg}; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 600;">{icon} {pred}</span>
                    </div>
                    <div style="background: #e5e7eb; height: 6px; border-radius: 3px; margin-top: 10px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #6366f1, #8b5cf6); height: 100%; width: {conf}%; border-radius: 3px;"></div>
                    </div>
                    <p style="margin: 5px 0 0 0; font-size: 12px; color: #6b7280;">Confidence: {conf:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No trained models found. Train with: `python train_clevr.py --model rn --epochs 30`")

# Footer
st.markdown("<hr style='border: 1px solid #e5e7eb; margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)
with st.expander("ℹ️ About"):
    st.markdown("""
    <p style="color: #374151; font-size: 14px; line-height: 1.6;">
        <strong style="color: #1f2937;">Sort-of-CLEVR</strong> is a visual reasoning benchmark that tests relational understanding.
        <br><br>
        <strong>Results:</strong> Relation Networks achieve <span style="color: #10b981; font-weight: 600;">86.9%</span> accuracy 
        vs CNN Baseline's <span style="color: #ef4444; font-weight: 600;">52.6%</span>.
        <br><br>
        <strong>Key Insight:</strong> Standard CNNs fail at relational questions ("Which object is closest to X?") 
        because they lack explicit mechanisms to compare objects. Relation Networks process all object pairs explicitly.
    </p>
    """, unsafe_allow_html=True)
