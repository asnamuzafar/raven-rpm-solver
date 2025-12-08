"""
Sort-of-CLEVR Training Script
==============================
Train Relation Network on Sort-of-CLEVR visual reasoning task.

Usage:
    # First generate the dataset
    python train_clevr.py --generate --num_images 5000
    
    # Then train
    python train_clevr.py --model rn --epochs 30
    python train_clevr.py --model baseline --epochs 20

Available Models:
    - rn       : Relation Network (achieves ~95% on relational tasks)
    - baseline : CNN + MLP baseline (fails on relational tasks)

Expected Results:
    - Relation Network: 95%+ on both relational and non-relational
    - CNN Baseline: 95% non-relational, ~60% relational
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

# Import models from models directory
from models.clevr.relation_network import SortOfCLEVRModel, BaselineCNN
from datasets.sort_of_clevr_generator import SortOfCLEVRDataset, generate_dataset


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
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100*total_correct/total_samples:.1f}%'})
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'relational_acc': correct_rel / total_rel if total_rel > 0 else 0,
        'non_relational_acc': correct_nonrel / total_nonrel if total_nonrel > 0 else 0,
    }


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
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'relational_acc': correct_rel / total_rel if total_rel > 0 else 0,
        'non_relational_acc': correct_nonrel / total_nonrel if total_nonrel > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train on Sort-of-CLEVR dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset first
  python train_clevr.py --generate --num_images 5000
  
  # Train Relation Network  
  python train_clevr.py --model rn --epochs 30
  
  # Train baseline for comparison
  python train_clevr.py --model baseline --epochs 20
        """
    )
    
    # Dataset generation
    parser.add_argument('--generate', action='store_true',
                        help='Generate Sort-of-CLEVR dataset')
    parser.add_argument('--num_images', type=int, default=10000,
                        help='Number of images to generate')
    parser.add_argument('--questions_per_image', type=int, default=10,
                        help='Questions per image')
    
    # Dataset/model
    parser.add_argument('--data_dir', type=str, default='./data/sort_of_clevr')
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--model', type=str, default='rn', 
                        choices=['rn', 'baseline'],
                        help='Model type: rn (Relation Network) or baseline (CNN+MLP)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Generate dataset if requested
    if args.generate:
        print(f"\n{'='*60}")
        print("Generating Sort-of-CLEVR Dataset")
        print(f"{'='*60}")
        generate_dataset(
            num_images=args.num_images,
            questions_per_image=args.questions_per_image,
            save_dir=args.data_dir
        )
        print("Dataset generation complete!")
        if not any([args.epochs]):
            return
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("Sort-of-CLEVR Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load datasets
    print(f"\nLoading data from {args.data_dir}...")
    train_dataset = SortOfCLEVRDataset(args.data_dir, split='train')
    val_dataset = SortOfCLEVRDataset(args.data_dir, split='val')
    test_dataset = SortOfCLEVRDataset(args.data_dir, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model_name = 'Relation Network' if args.model == 'rn' else 'CNN Baseline'
    if args.model == 'rn':
        model = SortOfCLEVRModel().to(device)
    else:
        model = BaselineCNN().to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"Parameters: {params:,}")
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training
    best_val_acc = 0
    history = []
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {100*train_metrics['accuracy']:.1f}%")
        print(f"          Rel: {100*train_metrics['relational_acc']:.1f}%, Non-Rel: {100*train_metrics['non_relational_acc']:.1f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {100*val_metrics['accuracy']:.1f}%")
        print(f"          Rel: {100*val_metrics['relational_acc']:.1f}%, Non-Rel: {100*val_metrics['non_relational_acc']:.1f}%")
        
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics})
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_rel_acc': val_metrics['relational_acc'],
                'val_nonrel_acc': val_metrics['non_relational_acc'],
            }, os.path.join(args.save_dir, f'clevr_{args.model}_best.pt'))
            print(f"  ✓ New best! Val Acc: {100*val_metrics['accuracy']:.1f}%")
    
    # Test evaluation
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")
    
    checkpoint = torch.load(os.path.join(args.save_dir, f'clevr_{args.model}_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = validate(model, test_loader, device)
    
    print(f"\nTest Results ({model_name}):")
    print(f"  Overall Accuracy: {100*test_metrics['accuracy']:.1f}%")
    print(f"  Relational:       {100*test_metrics['relational_acc']:.1f}%")
    print(f"  Non-Relational:   {100*test_metrics['non_relational_acc']:.1f}%")
    
    # Save
    with open(os.path.join(args.save_dir, f'clevr_{args.model}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    results = {
        'model': model_name,
        'test_accuracy': test_metrics['accuracy'],
        'test_relational': test_metrics['relational_acc'],
        'test_non_relational': test_metrics['non_relational_acc'],
    }
    with open(os.path.join(args.save_dir, f'clevr_{args.model}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
