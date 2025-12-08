"""
RAVEN RPM Solver - Evaluation Script

Stage E: Comprehensive evaluation module as per goal.md.
Evaluates all trained models and generates comparison reports.

Metrics (per goal.md):
• accuracy on standard test sets
• accuracy on unseen puzzle configurations (generalization)
• sample efficiency (performance with limited training data)
• rule-trace fidelity (how well symbolic models match ground truth rules)
• explanation quality (interpretability)
• computational cost (inference time, parameters)

Usage:
    python evaluate.py --data_dir ./data/raven_medium --models_dir ./saved_models
"""
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import *
from models import create_model, load_model, SymbolicReasoner, SymbolicTokenizer
from models.iraven.encoder import RAVENFeatureExtractor
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


def evaluate_symbolic_reasoning(
    encoder,
    tokenizer,
    symbolic_reasoner,
    dataloader,
    device,
    max_samples: int = 100
) -> dict:
    """
    Evaluate the symbolic rule-based reasoner.
    Required by goal.md for rule-trace fidelity evaluation.
    
    Args:
        encoder: Visual encoder to extract features
        tokenizer: Symbolic tokenizer to extract attributes
        symbolic_reasoner: Rule-based reasoning module
        dataloader: Test data loader
        device: Computation device
        max_samples: Maximum samples to evaluate
        
    Returns:
        Dictionary with symbolic reasoning evaluation results
    """
    from collections import defaultdict
    
    encoder.eval()
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    
    correct = 0
    total = 0
    rules_detected = defaultdict(int)
    detailed_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if total >= max_samples:
                break
                
            x, y, paths = batch
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            
            # Extract features
            ctx_feat, choice_feat = encoder(x)
            all_features = torch.cat([ctx_feat, choice_feat], dim=1)
            
            # Get symbolic attributes
            symbolic_output = tokenizer.to_symbolic(all_features)
            
            for i in range(batch_size):
                if total >= max_samples:
                    break
                
                # Get attributes for this puzzle
                puzzle_attrs = symbolic_output[i]
                context_attrs = puzzle_attrs[:8]
                choice_attrs = puzzle_attrs[8:]
                
                # Get symbolic prediction
                pred, scores, explanation = symbolic_reasoner.predict(context_attrs, choice_attrs)
                target = y[i].item()
                
                is_correct = pred == target
                if is_correct:
                    correct += 1
                total += 1
                
                # Get full trace for rule analysis
                full_trace = symbolic_reasoner.get_full_trace(context_attrs, choice_attrs)
                
                # Count detected rules
                for attr, attr_data in full_trace['detected_rules'].items():
                    for rule_info in attr_data.get('rules', []):
                        rule_name = rule_info[0] if isinstance(rule_info, tuple) else str(rule_info)
                        rules_detected[rule_name] += 1
                
                # Store detailed result (first 10 only)
                if len(detailed_results) < 10:
                    detailed_results.append({
                        'path': paths[i] if isinstance(paths[i], str) else str(paths[i]),
                        'predicted': pred,
                        'target': target,
                        'correct': is_correct,
                        'scores': scores,
                        'detected_rules': {
                            attr: [r[0] for r in data.get('rules', [])]
                            for attr, data in full_trace['detected_rules'].items()
                        }
                    })
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'rules_detected': dict(rules_detected),
        'detailed_results': detailed_results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAVEN models')
    parser.add_argument('--dataset', type=str, default='raven',
                        choices=['raven', 'iraven'],
                        help='Dataset type: raven (original) or iraven (bias-corrected)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (overrides --dataset if provided)')
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
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = f'./data/{args.dataset}_medium'
    
    print(f"Device: {device}")
    print(f"Dataset type: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Results directory: {results_dir}")
    
    # Create dataloaders
    train_dl, val_dl, test_dl = create_dataloaders(
        data_dir,
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
    
    # Rule-trace fidelity evaluation (required by goal.md)
    print("\n" + "="*60)
    print("RULE-TRACE FIDELITY: SYMBOLIC REASONING EVALUATION")
    print("="*60)
    
    # Create symbolic reasoner and tokenizer for evaluation
    symbolic_reasoner = SymbolicReasoner()
    tokenizer = SymbolicTokenizer(feature_dim=FEATURE_DIM)
    encoder = RAVENFeatureExtractor(pretrained=False).to(device)
    
    # Load encoder weights from first model
    if models:
        first_model = list(models.values())[0]
        if hasattr(first_model, 'encoder'):
            encoder.load_state_dict(first_model.encoder.state_dict())
    
    # Evaluate symbolic reasoner on a sample of test data
    print("\nEvaluating symbolic rule-based reasoner...")
    symbolic_results = evaluate_symbolic_reasoning(
        encoder, tokenizer, symbolic_reasoner, test_dl, device, max_samples=100
    )
    
    print(f"\nSymbolic Reasoner Results:")
    print(f"  Prediction Accuracy: {symbolic_results['accuracy']:.4f}")
    print(f"  Rules Detected Distribution:")
    for rule, count in symbolic_results.get('rules_detected', {}).items():
        print(f"    {rule}: {count}")
    
    # Save symbolic results
    results['symbolic_reasoner'] = symbolic_results
    
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

