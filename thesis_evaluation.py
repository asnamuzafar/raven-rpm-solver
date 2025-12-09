#!/usr/bin/env python3
"""
Thesis Evaluation Visualizations
================================
Publication-quality graphs and evaluation figures for the RAVEN RPM Solver thesis.

This script generates comprehensive visualizations organized into categories:
1. Model Comparison - Overall performance across architectures
2. Experiment Progression - Learning dynamics and overfitting analysis
3. Sort-of-CLEVR Results - Relational reasoning benchmark
4. SOTA Gap Analysis - Performance relative to published results
5. Architecture Analysis - Parameter efficiency and inference time

All figures are saved as high-resolution PNG files suitable for publication.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palette - modern, publication-friendly
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Magenta
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#3B3B3B',      # Dark gray
    'light': '#E8E8E8',        # Light gray
    'gradient': ['#2E86AB', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA']
}

# Create output directory
OUTPUT_DIR = Path(__file__).parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA FROM EXPERIMENTS.md
# ============================================================================

# I-RAVEN Model Comparison (Best Results)
IRAVEN_MODELS = {
    'Neuro-Symbolic\n(RuleAwareReasoner)': {'val_acc': 32.9, 'params': 12.9, 'best': True},
    'Transformer': {'val_acc': 26.9, 'params': 13.7},
    'RelationNet': {'val_acc': 26.2, 'params': 12.1},
    'EfficientNet\nEncoder': {'val_acc': 21.1, 'params': 4.6},
    'DINOv2\nEncoder': {'val_acc': 13.7, 'params': 22.0},
}

# Neuro-Symbolic Training Progression (Exp 4)
NEURO_SYMBOLIC_TRAINING = {
    'epochs': [5, 10, 15],
    'train_acc': [34.0, 44.0, 48.8],
    'val_acc': [30.0, 32.9, 32.4]
}

# Experiment Overview (I-RAVEN attempts)
EXPERIMENT_OVERVIEW = [
    ('Exp 1: CNN Baseline', 12.0, 'Failed'),
    ('Exp 2: ResNet+Transformer', 26.9, 'Moderate'),
    ('Exp 3: RelationNet', 26.2, 'Moderate'),
    ('Exp 4: Neuro-Symbolic', 32.9, 'Best'),
    ('Exp 5: EfficientNet', 21.1, 'Weak'),
    ('Exp 7: SCL Baseline', 22.8, 'Moderate'),
    ('Exp 10: SCL+Contrastive', 27.3, 'Moderate'),
    ('Exp 13: Spatial Conv', 34.0, 'Overfit'),
    ('Exp 16: RuleAware V2', 24.2, 'Overfit'),
]

# Sort-of-CLEVR Results
CLEVR_MODELS = {
    'CNN Baseline': {'overall': 52.6, 'relational': 54.2, 'non_rel': 50.9, 'params': 4.6},
    'Relation Network\n(Original)': {'overall': 64.7, 'relational': 64.6, 'non_rel': 64.9, 'params': 0.295},
    'Relation Network\n(Improved)': {'overall': 86.9, 'relational': 73.8, 'non_rel': 99.9, 'params': 1.76},
}

# Medium Dataset Results (Exp 6) - Overfitting Analysis
MEDIUM_DATASET = {
    'RelationNet': {'train': 57.6, 'val': 23.5},
    'Neuro-Symbolic': {'train': 39.8, 'val': 22.1},
    'Transformer': {'train': 51.5, 'val': 21.4},
    'Frozen ResNet': {'train': 19.0, 'val': 18.3},
}


def fig1_iraven_model_comparison():
    """
    Figure 1: I-RAVEN Model Comparison
    Main bar chart showing validation accuracy across different architectures.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(IRAVEN_MODELS.keys())
    accuracies = [IRAVEN_MODELS[m]['val_acc'] for m in models]
    is_best = [IRAVEN_MODELS[m].get('best', False) for m in models]
    
    # Create gradient colors
    colors = [COLORS['primary'] if not b else COLORS['success'] for b in is_best]
    
    bars = ax.bar(range(len(models)), accuracies, color=colors, 
                  edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontweight='bold', fontsize=11)
    
    # Random baseline
    ax.axhline(y=12.5, color=COLORS['neutral'], linestyle='--', 
               linewidth=2, alpha=0.7, label='Random Chance (12.5%)')
    
    # SOTA reference
    ax.axhline(y=92.9, color=COLORS['accent'], linestyle=':', 
               linewidth=2, alpha=0.8, label='Published SOTA (92.9%)')
    
    ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Model Architecture', fontweight='bold')
    ax.set_title('I-RAVEN: Model Architecture Comparison', fontweight='bold', pad=15)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='-')
    
    # Add annotation for best model
    ax.annotate('Best Result', 
               xy=(0, 32.9), xytext=(1.5, 50),
               arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2),
               fontsize=11, fontweight='bold', color=COLORS['success'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_iraven_model_comparison.png')
    plt.close()
    print(f"✓ Saved: fig1_iraven_model_comparison.png")


def fig2_training_dynamics():
    """
    Figure 2: Training Dynamics - Neuro-Symbolic Model
    Shows train/val accuracy over epochs, highlighting overfitting.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = NEURO_SYMBOLIC_TRAINING['epochs']
    train = NEURO_SYMBOLIC_TRAINING['train_acc']
    val = NEURO_SYMBOLIC_TRAINING['val_acc']
    
    # Plot lines with markers
    ax.plot(epochs, train, 'o-', color=COLORS['primary'], linewidth=2.5, 
            markersize=10, label='Training Accuracy', markeredgecolor='white', 
            markeredgewidth=2)
    ax.plot(epochs, val, 's-', color=COLORS['success'], linewidth=2.5, 
            markersize=10, label='Validation Accuracy', markeredgecolor='white',
            markeredgewidth=2)
    
    # Fill the gap (overfitting zone)
    ax.fill_between(epochs, train, val, alpha=0.2, color=COLORS['accent'],
                    label='Generalization Gap')
    
    # Mark best validation
    best_idx = val.index(max(val))
    ax.scatter([epochs[best_idx]], [val[best_idx]], s=200, c=COLORS['accent'], 
               zorder=5, marker='*', edgecolors='white', linewidths=2)
    ax.annotate(f'Best: {max(val):.1f}%', 
               xy=(epochs[best_idx], val[best_idx]),
               xytext=(epochs[best_idx]+1.5, val[best_idx]+3),
               fontsize=11, fontweight='bold', color=COLORS['success'])
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Neuro-Symbolic Model: Training Dynamics', fontweight='bold', pad=15)
    ax.set_ylim(25, 55)
    ax.set_xlim(3, 17)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_training_dynamics.png')
    plt.close()
    print(f"✓ Saved: fig2_training_dynamics.png")


def fig3_overfitting_analysis():
    """
    Figure 3: Overfitting Analysis on Medium Dataset
    Compares train vs. validation accuracy across models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(MEDIUM_DATASET.keys())
    train_accs = [MEDIUM_DATASET[m]['train'] for m in models]
    val_accs = [MEDIUM_DATASET[m]['val'] for m in models]
    gaps = [t - v for t, v in zip(train_accs, val_accs)]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Grouped bars
    bars1 = ax.bar(x - width/2, train_accs, width, label='Training Accuracy',
                   color=COLORS['primary'], edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, val_accs, width, label='Validation Accuracy',
                   color=COLORS['success'], edgecolor='white', linewidth=2)
    
    # Add gap annotations
    for i, (t, v, g) in enumerate(zip(train_accs, val_accs, gaps)):
        # Draw gap line
        ax.plot([i, i], [v, t], color=COLORS['accent'], linewidth=2, linestyle='--')
        ax.annotate(f'Gap: {g:.1f}%', 
                   xy=(i, (t+v)/2), xytext=(i+0.3, (t+v)/2),
                   fontsize=9, color=COLORS['accent'], fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('Overfitting Analysis: I-RAVEN Medium Dataset (8,400 samples)', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.axhline(y=12.5, color=COLORS['neutral'], linestyle='--', 
               linewidth=1.5, alpha=0.5, label='Random Chance')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 70)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_overfitting_analysis.png')
    plt.close()
    print(f"✓ Saved: fig3_overfitting_analysis.png")


def fig4_clevr_comparison():
    """
    Figure 4: Sort-of-CLEVR Benchmark Results
    Shows overall, relational, and non-relational accuracy.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    
    models = list(CLEVR_MODELS.keys())
    overall = [CLEVR_MODELS[m]['overall'] for m in models]
    relational = [CLEVR_MODELS[m]['relational'] for m in models]
    non_rel = [CLEVR_MODELS[m]['non_rel'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    # Create grouped bars
    bars1 = ax.bar(x - width, overall, width, label='Overall',
                   color=COLORS['primary'], edgecolor='white', linewidth=2)
    bars2 = ax.bar(x, relational, width, label='Relational',
                   color=COLORS['secondary'], edgecolor='white', linewidth=2)
    bars3 = ax.bar(x + width, non_rel, width, label='Non-Relational',
                   color=COLORS['accent'], edgecolor='white', linewidth=2)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('Sort-of-CLEVR: Relational Reasoning Benchmark', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha='center')
    ax.axhline(y=50, color=COLORS['neutral'], linestyle='--', 
               linewidth=1.5, alpha=0.5, label='Random Chance (50%)')
    ax.legend(loc='upper left', framealpha=0.9, ncol=2)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight the key finding
    ax.annotate('Near-perfect on\nnon-relational tasks!', 
               xy=(2 + width, 99.9), xytext=(2.3, 85),
               arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2),
               fontsize=10, fontweight='bold', color=COLORS['success'],
               ha='left')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_clevr_comparison.png')
    plt.close()
    print(f"✓ Saved: fig4_clevr_comparison.png")


def fig5_sota_gap():
    """
    Figure 5: Gap to State-of-the-Art
    Visual comparison showing our best result vs. published SOTA.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['Our Best\n(I-RAVEN)', 'Published SOTA\n(I-RAVEN)']
    values = [32.9, 92.9]
    colors = [COLORS['primary'], COLORS['success']]
    
    bars = ax.barh(categories, values, color=colors, height=0.5,
                   edgecolor='white', linewidth=3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.annotate(f'{val:.1f}%',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center',
                   fontsize=14, fontweight='bold')
    
    # Draw gap annotation
    gap = values[1] - values[0]
    ax.annotate('', xy=(values[1], 0.25), xytext=(values[0], 0.25),
               arrowprops=dict(arrowstyle='<->', color=COLORS['accent'], 
                              lw=3, shrinkA=5, shrinkB=5))
    ax.text((values[0] + values[1])/2, 0.25, f'Gap: {gap:.0f}%', 
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            color=COLORS['accent'])
    
    ax.set_xlabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Performance Gap: Our Result vs. Published SOTA', 
                 fontweight='bold', pad=15)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    # Add random chance line
    ax.axvline(x=12.5, color=COLORS['neutral'], linestyle='--', 
               linewidth=2, alpha=0.7)
    ax.text(13.5, 1.3, 'Random\n(12.5%)', fontsize=9, 
            color=COLORS['neutral'], va='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_sota_gap.png')
    plt.close()
    print(f"✓ Saved: fig5_sota_gap.png")


def fig6_experiment_progression():
    """
    Figure 6: Experiment Progression on I-RAVEN
    Shows how different approaches performed across experiments.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [e[0] for e in EXPERIMENT_OVERVIEW]
    accs = [e[1] for e in EXPERIMENT_OVERVIEW]
    categories = [e[2] for e in EXPERIMENT_OVERVIEW]
    
    # Color by category
    cat_colors = {
        'Failed': COLORS['neutral'],
        'Weak': '#E8E8E8',
        'Moderate': COLORS['primary'],
        'Best': COLORS['success'],
        'Overfit': COLORS['accent']
    }
    colors = [cat_colors[c] for c in categories]
    
    bars = ax.bar(range(len(names)), accs, color=colors, 
                  edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Random baseline
    ax.axhline(y=12.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Experiment', fontweight='bold')
    ax.set_title('I-RAVEN: Experimental Progression', fontweight='bold', pad=15)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(':')[0] for n in names], rotation=45, ha='right')
    ax.set_ylim(0, 45)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_elements = [mpatches.Patch(facecolor=cat_colors[cat], 
                                      edgecolor='white', linewidth=2, label=cat)
                       for cat in ['Failed', 'Moderate', 'Best', 'Overfit']]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_experiment_progression.png')
    plt.close()
    print(f"✓ Saved: fig6_experiment_progression.png")


def fig7_parameter_efficiency():
    """
    Figure 7: Parameter Efficiency (Accuracy vs Parameters)
    Scatter plot showing accuracy/parameter trade-off.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Combine I-RAVEN and CLEVR data
    data = []
    for name, info in IRAVEN_MODELS.items():
        data.append({
            'name': name.replace('\n', ' '),
            'acc': info['val_acc'],
            'params': info['params'],
            'dataset': 'I-RAVEN'
        })
    for name, info in CLEVR_MODELS.items():
        data.append({
            'name': name.replace('\n', ' '),
            'acc': info['overall'],
            'params': info['params'],
            'dataset': 'Sort-of-CLEVR'
        })
    
    # Plot points
    for d in data:
        color = COLORS['primary'] if d['dataset'] == 'I-RAVEN' else COLORS['success']
        marker = 'o' if d['dataset'] == 'I-RAVEN' else 's'
        ax.scatter(d['params'], d['acc'], s=200, c=color, marker=marker,
                  edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(d['name'][:15], (d['params'], d['acc']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Parameters (Millions)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Model Efficiency: Accuracy vs. Parameter Count', 
                 fontweight='bold', pad=15)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['primary'],
                   markersize=12, label='I-RAVEN'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['success'],
                   markersize=12, label='Sort-of-CLEVR')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_parameter_efficiency.png')
    plt.close()
    print(f"✓ Saved: fig7_parameter_efficiency.png")


def fig8_summary_dashboard():
    """
    Figure 8: Summary Dashboard
    Multi-panel figure summarizing all key results.
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: I-RAVEN Best Result
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Neuro-Symbolic', 'Transformer', 'RelationNet', 'EfficientNet', 'DINOv2']
    accs = [32.9, 26.9, 26.2, 21.1, 13.7]
    colors = [COLORS['success']] + [COLORS['primary']] * 4
    bars = ax1.barh(models[::-1], accs[::-1], color=colors[::-1], 
                    edgecolor='white', linewidth=2)
    ax1.axvline(x=12.5, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Val Accuracy (%)')
    ax1.set_title('(A) I-RAVEN: Model Comparison', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Panel B: Sort-of-CLEVR Improvement
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['CNN\nBaseline', 'RN\n(Original)', 'RN\n(Improved)']
    accs = [52.6, 64.7, 86.9]
    colors = [COLORS['neutral'], COLORS['primary'], COLORS['success']]
    ax2.bar(models, accs, color=colors, edgecolor='white', linewidth=2)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7)
    for i, acc in enumerate(accs):
        ax2.annotate(f'{acc:.1f}%', (i, acc), 
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('(B) Sort-of-CLEVR: Improvement Journey', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    # Panel C: Key Metrics Table
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    table_data = [
        ['Metric', 'I-RAVEN', 'Sort-of-CLEVR'],
        ['Best Accuracy', '32.9%', '86.9%'],
        ['Published SOTA', '92.9%', 'N/A'],
        ['# Experiments', '19', '3'],
        ['Best Model', 'Neuro-Symbolic', 'Improved RN'],
        ['Key Challenge', 'Overfitting', 'Relational Tasks']
    ]
    table = ax3.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.35, 0.3, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax3.set_title('(C) Summary Statistics', fontweight='bold', pad=20)
    
    # Panel D: Key Findings
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    findings = [
        "✓ Neuro-symbolic reasoning achieves best I-RAVEN result (32.9%)",
        "✓ Severe overfitting remains the key challenge",
        "✓ Improved Relation Network achieves 86.9% on Sort-of-CLEVR",
        "✓ Non-relational tasks solved near-perfectly (99.9%)",
        "✓ 60% gap to SOTA suggests need for structural learning",
        "✓ Self-supervised encoders (DINOv2) fail on abstract shapes"
    ]
    
    y_pos = 0.9
    for finding in findings:
        ax4.text(0.05, y_pos, finding, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['light'], 
                         edgecolor='none'))
        y_pos -= 0.15
    
    ax4.set_title('(D) Key Findings', fontweight='bold', pad=20)
    
    fig.suptitle('RAVEN RPM Solver: Experimental Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(OUTPUT_DIR / 'fig8_summary_dashboard.png')
    plt.close()
    print(f"✓ Saved: fig8_summary_dashboard.png")


def fig9_architecture_diagram():
    """
    Figure 9: Architecture Overview
    Visual representation of the neuro-symbolic pipeline.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    # Define stages
    stages = [
        ('Stage A\nPerception', 'CNN Encoder\n(ResNet-18)', COLORS['primary']),
        ('Stage B\nTokenizer', 'Attribute\nExtraction', COLORS['secondary']),
        ('Stage C\nReasoner', 'Transformer/\nRelationNet', COLORS['accent']),
        ('Stage D\nBaselines', 'CNN Direct/\nSymbolic', COLORS['neutral']),
        ('Stage E\nEvaluation', 'Comparison\n& Metrics', COLORS['success']),
    ]
    
    box_width = 0.15
    box_height = 0.6
    gap = 0.05
    start_x = 0.05
    y = 0.2
    
    for i, (title, desc, color) in enumerate(stages):
        x = start_x + i * (box_width + gap)
        
        # Draw box
        rect = mpatches.FancyBboxPatch((x, y), box_width, box_height,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, alpha=0.8,
                                        edgecolor='white', linewidth=3)
        ax.add_patch(rect)
        
        # Add title
        ax.text(x + box_width/2, y + box_height - 0.1, title,
               ha='center', va='top', fontsize=11, fontweight='bold',
               color='white')
        
        # Add description
        ax.text(x + box_width/2, y + 0.2, desc,
               ha='center', va='center', fontsize=9,
               color='white')
        
        # Draw arrow to next
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + box_width + gap - 0.01, y + box_height/2),
                       xytext=(x + box_width + 0.01, y + box_height/2),
                       arrowprops=dict(arrowstyle='->', color=COLORS['neutral'], lw=2))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Neuro-Symbolic Pipeline Architecture (from goal.md)', 
                 fontweight='bold', pad=15, fontsize=14)
    
    plt.savefig(OUTPUT_DIR / 'fig9_architecture_diagram.png')
    plt.close()
    print(f"✓ Saved: fig9_architecture_diagram.png")


def create_results_table_latex():
    """
    Generate LaTeX table for thesis.
    """
    latex = r"""
\begin{table}[h]
\centering
\caption{Experimental Results Summary}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Val Acc (\%)} & \textbf{Params (M)} & \textbf{Dataset} \\
\midrule
\multicolumn{4}{l}{\textit{I-RAVEN Experiments}} \\
\midrule
Neuro-Symbolic (RuleAwareReasoner) & \textbf{32.9} & 12.9 & I-RAVEN Large \\
Transformer & 26.9 & 13.7 & I-RAVEN Large \\
RelationNet & 26.2 & 12.1 & I-RAVEN Large \\
EfficientNet Encoder & 21.1 & 4.6 & I-RAVEN Large \\
DINOv2 Encoder & 13.7 & 22.0 & I-RAVEN Large \\
\midrule
\multicolumn{4}{l}{\textit{Sort-of-CLEVR Experiments}} \\
\midrule
Improved Relation Network & \textbf{86.9} & 1.76 & Sort-of-CLEVR \\
Original Relation Network & 64.7 & 0.295 & Sort-of-CLEVR \\
CNN Baseline & 52.6 & 4.6 & Sort-of-CLEVR \\
\midrule
\multicolumn{4}{l}{\textit{Published Baselines}} \\
\midrule
Random Chance (8-way) & 12.5 & -- & I-RAVEN \\
Published SOTA & 92.9 & -- & I-RAVEN \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    output_path = OUTPUT_DIR / 'results_table.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"✓ Saved: results_table.tex")


def main():
    """Generate all thesis figures."""
    print("=" * 60)
    print("THESIS EVALUATION VISUALIZATIONS")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 60)
    
    # Generate all figures
    fig1_iraven_model_comparison()
    fig2_training_dynamics()
    fig3_overfitting_analysis()
    fig4_clevr_comparison()
    fig5_sota_gap()
    fig6_experiment_progression()
    fig7_parameter_efficiency()
    fig8_summary_dashboard()
    fig9_architecture_diagram()
    create_results_table_latex()
    
    print("-" * 60)
    print(f"✓ All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
