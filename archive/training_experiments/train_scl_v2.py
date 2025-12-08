"""
Training Script for SCL-Inspired Model
=======================================
Experiment 19: Scattering Compositional Learner

Key Features:
1. Patch-based encoding (scattering transform)
2. Strong regularization (dropout, weight decay)
3. Consistency loss (r0 ≈ r1 ≈ r2)
4. Cosine annealing LR
5. Mixed precision training
"""

import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS,
    DEVICE, NUM_WORKERS, WEIGHT_DECAY
)
from models.scl_model import SCLModel, SCLLoss
from utils.dataset import RAVENDataset, get_split_files, create_dataloaders


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scaler, criterion, device, grad_clip=1.0):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    loss_components = {'ce': 0, 'contrast': 0, 'consistency': 0}
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Handle different batch formats
        if len(batch) == 3:
            images, targets, paths = batch
        else:
            images, targets, paths, meta = batch
        
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits, extras = model(images, return_extras=True)
            loss, losses = criterion(logits, targets, extras)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)
        
        for k, v in losses.items():
            if k in loss_components:
                loss_components[k] += v * images.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100*total_correct/total_samples:.1f}%'
        })
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    avg_components = {k: v/total_samples for k, v in loss_components.items()}
    
    return avg_loss, accuracy, avg_components


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            if len(batch) == 3:
                images, targets, paths = batch
            else:
                images, targets, paths, meta = batch
            
            images = images.to(device)
            targets = targets.to(device)
            
            with autocast():
                logits, extras = model(images, return_extras=True)
                loss, _ = criterion(logits, targets, extras)
            
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser(description='Train SCL Model')
    parser.add_argument('--data_dir', type=str, default='./data/iraven_large')
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_contrast', type=float, default=1.0)
    parser.add_argument('--lambda_consistency', type=float, default=0.5)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        return_meta=False
    )
    
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")
    
    # Create model
    model = SCLModel(
        patch_size=32,
        embed_dim=128,
        panel_dim=256,
        rule_dim=128,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = SCLLoss(
        lambda_contrast=args.lambda_contrast,
        lambda_consistency=args.lambda_consistency
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = []
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc, loss_components = train_epoch(
            model, train_loader, optimizer, scaler, criterion, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Log results
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {100*train_acc:.1f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {100*val_acc:.1f}%")
        print(f"  Loss Components - CE: {loss_components['ce']:.3f}, "
              f"Contrast: {loss_components['contrast']:.3f}, "
              f"Consistency: {loss_components['consistency']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Check for overfitting
        gap = train_acc - val_acc
        if gap > 0.15:
            print(f"  ⚠️  Overfitting warning: Train-Val gap = {100*gap:.1f}%")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'loss_components': loss_components,
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'scl_best.pt'))
            print(f"  ✓ New best model saved! Val Acc: {100*val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'scl_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Best Val Acc: {100*best_val_acc:.1f}%")
    print(f"Test Acc: {100*test_acc:.1f}%")
    
    # Save history
    with open(os.path.join(args.save_dir, 'scl_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best model saved to {args.save_dir}/scl_best.pt")
    

if __name__ == '__main__':
    main()
