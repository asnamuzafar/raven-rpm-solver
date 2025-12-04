"""
RAVEN RPM Solver - Evaluation Script

Evaluate all trained models and generate comparison reports.

Usage:
    python evaluate.py --data_dir ./data/raven_small --models_dir ./saved_models
"""
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import *
from models import create_model, load_model
from utils import create_dataloaders, ModelEvaluator


def visualize_prediction(
    model,
    puzzle_path: str,
    device: str = 'cuda',
    save_path: str = None
):
    """Visualize model prediction on a single puzzle"""
    data = np.load(puzzle_path)
    imgs = data["image"]
    target = int(data["target"])
    
    # Prepare input
    x = torch.from_numpy(imgs.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(logits.argmax(dim=1).item())
    
    # Create visualization
    fig = plt.figure(figsize=(16, 8))
    
    # Plot 3x3 context grid
    for i in range(8):
        ax = plt.subplot(3, 6, (i // 3) * 6 + (i % 3) + 1)
        ax.imshow(imgs[i], cmap='gray')
        ax.set_title(f'Panel {i}', fontsize=9)
        ax.axis('off')
    
    # Question mark for missing cell
    ax = plt.subplot(3, 6, 3)
    ax.text(0.5, 0.5, '?', fontsize=40, ha='center', va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Plot choices with probabilities
    for i in range(8):
        row = i // 4
        col = i % 4
        ax = plt.subplot(3, 6, row * 6 + col + 4 + (6 if row == 1 else 0))
        
        ax.imshow(imgs[8+i], cmap='gray')
        
        # Color based on prediction
        if i == pred and i == target:
            title = f'C{i} ✓✓\n{probs[i]:.2f}'
            color = 'green'
        elif i == pred:
            title = f'C{i} ←pred\n{probs[i]:.2f}'
            color = 'red'
        elif i == target:
            title = f'C{i} (GT)\n{probs[i]:.2f}'
            color = 'blue'
        else:
            title = f'C{i}\n{probs[i]:.2f}'
            color = 'gray'
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_title(title, fontsize=8)
        ax.axis('off')
    
    status = "✓ CORRECT" if pred == target else "✗ WRONG"
    fig.suptitle(f'Predicted: {pred} | Actual: {target} | {status}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return pred == target


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAVEN models')
    parser.add_argument('--data_dir', type=str, default='./data/raven_small',
                        help='Path to RAVEN data directory')
    parser.add_argument('--models_dir', type=str, default='./saved_models',
                        help='Directory containing trained models')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save evaluation results')
    parser.add_argument('--visualize', type=int, default=5,
                        help='Number of sample predictions to visualize')
    args = parser.parse_args()
    
    device = DEVICE
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Models directory: {models_dir}")
    print(f"Results directory: {results_dir}")
    
    # Create dataloaders
    train_dl, val_dl, test_dl = create_dataloaders(
        args.data_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Load trained models
    model_files = list(models_dir.glob("*_model.pth"))
    if not model_files:
        print(f"No model files found in {models_dir}")
        print("Please run train.py first.")
        return
    
    models = {}
    model_type_map = {
        'transformer': 'transformer',
        'mlp_relational': 'mlp',
        'cnn_direct': 'cnn_direct',
        'relationnet': 'relation_net',
        'hybrid': 'hybrid'
    }
    
    for model_file in model_files:
        checkpoint = torch.load(model_file, map_location=device)
        model_name = checkpoint.get('model_name', model_file.stem)
        
        # Determine model type from filename
        model_type = 'transformer'  # default
        for key, mtype in model_type_map.items():
            if key in model_file.stem.lower():
                model_type = mtype
                break
        
        model = create_model(model_type=model_type, pretrained_encoder=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        models[model_name] = model
        print(f"Loaded: {model_name} from {model_file.name}")
    
    # Run evaluation
    evaluator = ModelEvaluator(device=device)
    results = evaluator.run_full_evaluation(models, val_dl, test_dl)
    
    # Per-configuration evaluation (generalization - required by goal.md)
    print("\n" + "="*60)
    print("GENERALIZATION: PER-CONFIGURATION ACCURACY")
    print("="*60)
    for name, model in models.items():
        model = model.to(device)
        config_acc = evaluator.evaluate_per_configuration(model, test_dl, name)
        print(f"\n{name}:")
        for config, acc in sorted(config_acc.items()):
            print(f"  {config}: {acc:.4f}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    df = evaluator.generate_comparison_table()
    print(df.to_string())
    
    # Save results
    evaluator.save_results(results_dir)
    
    # Plot results
    evaluator.plot_results(save_path=results_dir / 'comparison_plots.png')
    
    # Plot generalization (per-config accuracy)
    evaluator.plot_generalization(save_path=results_dir / 'generalization_plots.png')
    
    # Visualize sample predictions
    if args.visualize > 0:
        print(f"\n{'='*60}")
        print(f"SAMPLE PREDICTIONS")
        print(f"{'='*60}")
        
        # Get test files
        test_files = [batch[2] for batch in test_dl for _ in range(len(batch[2]))]
        test_files = test_files[:len(test_dl.dataset)]
        
        # Select random samples
        import random
        sample_files = random.sample(test_files[:100], min(args.visualize, len(test_files)))
        
        # Use best model for visualization
        best_model_name = max(results.keys(), 
                             key=lambda k: results[k]['test_accuracy'])
        best_model = models[best_model_name]
        
        print(f"\nUsing best model: {best_model_name}")
        for i, path in enumerate(sample_files):
            print(f"\nSample {i+1}:")
            save_path = results_dir / f'prediction_sample_{i+1}.png'
            visualize_prediction(best_model, path, device, save_path)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()

