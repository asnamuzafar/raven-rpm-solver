"""
Stage E: Comparison and Evaluation Module

Unified evaluation script for all models as specified in goal.md.
Measures:
• accuracy on standard test sets
• accuracy on unseen puzzle configurations (generalization)
• sample efficiency (performance with limited training data)
• rule-trace fidelity (how well symbolic models match ground truth rules)
• explanation quality (interpretability)
• computational cost (inference time, parameters)
"""
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import matplotlib.pyplot as plt
import re


def compute_metrics(
    predictions: List[int], 
    targets: List[int]
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of predicted labels
        targets: List of true labels
        
    Returns:
        Dictionary of metrics
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    accuracy = (predictions == targets).mean()
    
    # Per-class accuracy
    class_accs = {}
    for c in range(8):
        mask = targets == c
        if mask.sum() > 0:
            class_accs[f'class_{c}_acc'] = (predictions[mask] == c).mean()
    
    return {
        'accuracy': float(accuracy),
        'total_samples': len(targets),
        'correct': int((predictions == targets).sum()),
        **class_accs
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Tuple[float, List[Dict]]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        accuracy: Overall accuracy
        predictions: List of prediction details
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y, paths = batch
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            for i in range(len(preds)):
                predictions.append({
                    'path': paths[i],
                    'predicted': preds[i].item(),
                    'actual': y[i].item(),
                    'correct': (preds[i] == y[i]).item(),
                    'confidence': probs[i, preds[i]].item()
                })
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, predictions


class ModelEvaluator:
    """
    Comprehensive evaluation framework for all reasoning models.
    """
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = defaultdict(dict)
        
    def count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters"""
        if isinstance(model, nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return 0
    
    def measure_inference_time(
        self, 
        model: nn.Module, 
        sample_input: torch.Tensor,
        n_runs: int = 10
    ) -> float:
        """Measure average inference time in seconds"""
        model.eval()
        times = []
        
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.time()
                _ = model(sample_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)
        
        # Skip first run (warmup)
        return np.mean(times[1:]) if len(times) > 1 else times[0]
    
    def evaluate_single_model(
        self,
        model: nn.Module,
        model_name: str,
        val_dl: DataLoader,
        test_dl: DataLoader,
        sample_input: torch.Tensor
    ) -> Dict:
        """Evaluate a single model comprehensively"""
        print(f"\n--- Evaluating: {model_name} ---")
        
        # Parameter count
        n_params = self.count_parameters(model)
        print(f"  Parameters: {n_params:,}")
        
        # Inference time
        inf_time = self.measure_inference_time(model, sample_input)
        print(f"  Inference time: {inf_time*1000:.2f} ms")
        
        # Accuracy
        val_acc, _ = evaluate_model(model, val_dl, self.device)
        test_acc, test_preds = evaluate_model(model, test_dl, self.device)
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        results = {
            'parameters': n_params,
            'inference_time_ms': inf_time * 1000,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'predictions': test_preds
        }
        
        self.results[model_name] = results
        return results
    
    def run_full_evaluation(
        self,
        models_dict: Dict[str, nn.Module],
        val_dl: DataLoader,
        test_dl: DataLoader
    ) -> Dict:
        """
        Run comprehensive evaluation on all models.
        
        Args:
            models_dict: dict of {name: model}
            val_dl, test_dl: DataLoaders
            
        Returns:
            Dictionary of results for all models
        """
        print("=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        # Get a sample input for timing
        sample_batch = next(iter(val_dl))
        sample_input = sample_batch[0][:2].to(self.device)
        
        for name, model in models_dict.items():
            model = model.to(self.device)
            self.evaluate_single_model(model, name, val_dl, test_dl, sample_input)
        
        return dict(self.results)
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comparison DataFrame"""
        data = {}
        for name, result in self.results.items():
            data[name] = {
                'Parameters': result['parameters'],
                'Inference (ms)': round(result['inference_time_ms'], 2),
                'Val Accuracy': round(result['val_accuracy'], 4),
                'Test Accuracy': round(result['test_accuracy'], 4),
                'vs Random': f"{result['test_accuracy']/0.125:.2f}x"
            }
        
        df = pd.DataFrame(data).T
        df.index.name = 'Model'
        return df
    
    def plot_results(self, save_path: Optional[Path] = None):
        """Generate visualization plots"""
        df = self.generate_comparison_table()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Accuracy comparison
        ax = axes[0, 0]
        names = list(self.results.keys())
        val_accs = [self.results[n]['val_accuracy'] for n in names]
        test_accs = [self.results[n]['test_accuracy'] for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, val_accs, width, label='Validation', color='#2ecc71')
        ax.bar(x + width/2, test_accs, width, label='Test', color='#3498db')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.axhline(y=0.125, color='r', linestyle='--', label='Random (12.5%)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Parameter count
        ax = axes[0, 1]
        params = [self.results[n]['parameters'] for n in names]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        ax.bar(names, params, color=colors)
        ax.set_ylabel('Number of Parameters')
        ax.set_title('Model Complexity', fontweight='bold')
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Inference time
        ax = axes[1, 0]
        times = [self.results[n]['inference_time_ms'] for n in names]
        colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(names)))
        ax.bar(names, times, color=colors)
        ax.set_ylabel('Time (ms)')
        ax.set_title('Inference Time', fontweight='bold')
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Accuracy vs Complexity
        ax = axes[1, 1]
        scatter = ax.scatter(params, test_accs, s=100, c=range(len(names)), cmap='Set1')
        for i, name in enumerate(names):
            ax.annotate(name, (params[i], test_accs[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel('Parameters')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Accuracy vs Complexity Trade-off', fontweight='bold')
        ax.axhline(y=0.125, color='r', linestyle='--', alpha=0.5)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        
        plt.show()
        return fig
    
    def save_results(self, save_dir: Path):
        """Save evaluation results to files"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON (without predictions for size)
        json_results = {}
        for name, data in self.results.items():
            json_results[name] = {
                'parameters': data['parameters'],
                'inference_time_ms': data['inference_time_ms'],
                'val_accuracy': data['val_accuracy'],
                'test_accuracy': data['test_accuracy']
            }
        
        with open(save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save comparison table as CSV
        df = self.generate_comparison_table()
        df.to_csv(save_dir / 'comparison_table.csv')
        
        print(f"Results saved to {save_dir}")
    
    def evaluate_per_configuration(
        self,
        model: nn.Module,
        test_dl: DataLoader,
        model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate accuracy per puzzle configuration (generalization metric).
        Required by goal.md Stage E.
        """
        model.eval()
        config_correct = defaultdict(int)
        config_total = defaultdict(int)
        
        with torch.no_grad():
            for batch in test_dl:
                x, y, paths = batch
                x, y = x.to(self.device), y.to(self.device)
                
                logits = model(x)
                preds = logits.argmax(dim=1)
                
                for i, path in enumerate(paths):
                    # Extract config from path (e.g., "center_single", "distribute_four")
                    config = self._extract_config(path)
                    config_correct[config] += (preds[i] == y[i]).item()
                    config_total[config] += 1
        
        config_accuracy = {}
        for config in config_total:
            config_accuracy[config] = config_correct[config] / config_total[config]
        
        self.results[model_name]['per_config_accuracy'] = config_accuracy
        return config_accuracy
    
    def _extract_config(self, path: str) -> str:
        """Extract configuration name from file path"""
        # Match patterns like center_single, distribute_four, etc.
        configs = [
            'center_single', 'distribute_four', 'distribute_nine',
            'left_center_single_right_center_single',
            'up_center_single_down_center_single',
            'in_center_single_out_center_single',
            'in_distribute_four_out_center_single'
        ]
        for config in configs:
            if config in path:
                return config
        return 'unknown'
    
    def evaluate_sample_efficiency(
        self,
        model_factory: Callable,
        train_dataset,
        val_dl: DataLoader,
        fractions: List[float] = [0.1, 0.25, 0.5, 0.75, 1.0],
        epochs: int = 5,
        lr: float = 1e-4
    ) -> Dict[float, float]:
        """
        Evaluate how model performs with limited training data.
        Required by goal.md Stage E.
        
        Args:
            model_factory: Function that creates a new model instance
            train_dataset: Full training dataset
            val_dl: Validation dataloader
            fractions: Data fractions to test
            epochs: Training epochs per fraction
            lr: Learning rate
            
        Returns:
            Dict mapping fraction to validation accuracy
        """
        results = {}
        n_total = len(train_dataset)
        
        print("\nSample Efficiency Evaluation:")
        print("-" * 40)
        
        for frac in fractions:
            n_samples = int(n_total * frac)
            indices = random.sample(range(n_total), n_samples)
            subset = Subset(train_dataset, indices)
            train_dl = DataLoader(subset, batch_size=16, shuffle=True)
            
            # Create and train new model
            model = model_factory().to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(epochs):
                for batch in train_dl:
                    x, y, _ = batch
                    x, y = x.to(self.device), y.to(self.device)
                    
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            acc, _ = evaluate_model(model, val_dl, self.device)
            results[frac] = acc
            print(f"  {frac*100:5.1f}% data ({n_samples:4d} samples): {acc:.4f} accuracy")
        
        return results
    
    def plot_generalization(self, save_path: Optional[Path] = None):
        """Plot per-configuration accuracy (generalization analysis)"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = set()
        for model_name, data in self.results.items():
            if 'per_config_accuracy' in data:
                configs.update(data['per_config_accuracy'].keys())
        
        configs = sorted(configs)
        x = np.arange(len(configs))
        width = 0.8 / len(self.results)
        
        for i, (model_name, data) in enumerate(self.results.items()):
            if 'per_config_accuracy' in data:
                accs = [data['per_config_accuracy'].get(c, 0) for c in configs]
                ax.bar(x + i * width, accs, width, label=model_name)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Generalization: Per-Configuration Accuracy', fontweight='bold')
        ax.set_xticks(x + width * (len(self.results) - 1) / 2)
        ax.set_xticklabels([c.replace('_', '\n') for c in configs], fontsize=8)
        ax.axhline(y=0.125, color='r', linestyle='--', label='Random', alpha=0.7)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig

