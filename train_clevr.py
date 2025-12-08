"""
Training Script for Sort-of-CLEVR
==================================
Train Relation Network on Sort-of-CLEVR visual reasoning task.

Expected Results:
- Relation Network: 95%+ on both relational and non-relational
- CNN Baseline: 95%+ non-relational, ~60% relational
"""

import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from models.relation_network_clevr import SortOfCLEVRModel, BaselineCNN
from sort_of_clevr_generator import SortOfCLEVRDataset


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Track by question type
    correct_rel = 0
    total_rel = 0
    correct_nonrel = 0
    total_nonrel = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, questions, answers, q_types in pbar:
        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)
        q_types = q_types.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images, questions)
        loss = F.cross_entropy(logits, answers)
        
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct = (preds == answers)
        total_correct += correct.sum().item()
        total_samples += images.size(0)
        
        # By question type
        rel_mask = q_types == 1
        nonrel_mask = q_types == 0
        correct_rel += correct[rel_mask].sum().item()
        total_rel += rel_mask.sum().item()
        correct_nonrel += correct[nonrel_mask].sum().item()
        total_nonrel += nonrel_mask.sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100*total_correct/total_samples:.1f}%'
        })
    
    metrics = {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'relational_acc': correct_rel / total_rel if total_rel > 0 else 0,
        'non_relational_acc': correct_nonrel / total_nonrel if total_nonrel > 0 else 0,
    }
    
    return metrics


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    correct_rel = 0
    total_rel = 0
    correct_nonrel = 0
    total_nonrel = 0
    
    with torch.no_grad():
        for images, questions, answers, q_types in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            q_types = q_types.to(device)
            
            logits = model(images, questions)
            loss = F.cross_entropy(logits, answers)
            
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct = (preds == answers)
            total_correct += correct.sum().item()
            total_samples += images.size(0)
            
            rel_mask = q_types == 1
            nonrel_mask = q_types == 0
            correct_rel += correct[rel_mask].sum().item()
            total_rel += rel_mask.sum().item()
            correct_nonrel += correct[nonrel_mask].sum().item()
            total_nonrel += nonrel_mask.sum().item()
    
    metrics = {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'relational_acc': correct_rel / total_rel if total_rel > 0 else 0,
        'non_relational_acc': correct_nonrel / total_nonrel if total_nonrel > 0 else 0,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train on Sort-of-CLEVR')
    parser.add_argument('--data_dir', type=str, default='./data/sort_of_clevr')
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--model', type=str, default='rn', choices=['rn', 'baseline'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load datasets
    print(f"Loading data from {args.data_dir}...")
    train_dataset = SortOfCLEVRDataset(args.data_dir, split='train')
    val_dataset = SortOfCLEVRDataset(args.data_dir, split='val')
    test_dataset = SortOfCLEVRDataset(args.data_dir, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Create model
    if args.model == 'rn':
        model = SortOfCLEVRModel().to(device)
        model_name = 'Relation Network'
    else:
        model = BaselineCNN().to(device)
        model_name = 'CNN Baseline'
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"Parameters: {total_params:,}")
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    best_val_acc = 0
    history = []
    
    print(f"\n{'='*60}")
    print(f"Starting Training: {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {100*train_metrics['accuracy']:.1f}%")
        print(f"          Rel: {100*train_metrics['relational_acc']:.1f}%, Non-Rel: {100*train_metrics['non_relational_acc']:.1f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {100*val_metrics['accuracy']:.1f}%")
        print(f"          Rel: {100*val_metrics['relational_acc']:.1f}%, Non-Rel: {100*val_metrics['non_relational_acc']:.1f}%")
        
        history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
        })
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_rel_acc': val_metrics['relational_acc'],
                'val_nonrel_acc': val_metrics['non_relational_acc'],
            }, os.path.join(args.save_dir, f'{args.model}_clevr_best.pt'))
            print(f"  ✓ New best! Val Acc: {100*val_metrics['accuracy']:.1f}%")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Test Evaluation")
    print(f"{'='*60}")
    
    checkpoint = torch.load(os.path.join(args.save_dir, f'{args.model}_clevr_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, device)
    
    print(f"\nTest Results ({model_name}):")
    print(f"  Overall Accuracy: {100*test_metrics['accuracy']:.1f}%")
    print(f"  Relational:       {100*test_metrics['relational_acc']:.1f}%")
    print(f"  Non-Relational:   {100*test_metrics['non_relational_acc']:.1f}%")
    
    # Save history
    with open(os.path.join(args.save_dir, f'{args.model}_clevr_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final results
    results = {
        'model': model_name,
        'test_accuracy': test_metrics['accuracy'],
        'test_relational': test_metrics['relational_acc'],
        'test_non_relational': test_metrics['non_relational_acc'],
        'best_val_accuracy': best_val_acc,
    }
    
    with open(os.path.join(args.save_dir, f'{args.model}_clevr_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete! Results saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
