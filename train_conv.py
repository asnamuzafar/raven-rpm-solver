"""
Experiment 13: Spatial Convolutional Reasoner (Phase 5)

Addresses the overfitting in Exps 11/12 by enforcing Translational Invariance.
Instead of flattening 5x5 feature maps to vectors (which destroys relative position info),
this model applies 2D Convolutions over the stacked panels to detect rules like "shift right"
regardless of absolute location.

Components:
- Encoder: SpatialResNet (flatten_output=False) -> Output (B, 512, 5, 5)
- Reasoner: SpatialConvolutionalReasoner (Conv2D over stacked maps)
- Loss: CrossEntropy + Contrastive (No Attribute Supervision needed for this test)

Usage:
    python train_conv.py --data_dir ./data/iraven_large --epochs 30 --batch_size 64
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from config import (
    DEVICE, NUM_WORKERS, WEIGHT_DECAY, DROPOUT
)
from models import create_model
from utils.dataset import RAVENDataset, get_split_files


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dataloaders_with_meta(data_dir: Path, batch_size: int = 32, num_workers: int = 2) -> tuple:
    train_files, val_files, test_files = get_split_files(data_dir)
    train_ds = RAVENDataset(train_files, augment=True, return_meta=False) # No meta needed for Conv
    val_ds = RAVENDataset(val_files, augment=False, return_meta=False)
    test_ds = RAVENDataset(test_files, augment=False, return_meta=False)
    
    # Standard collate is fine since no meta
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl

def contrastive_loss(logits, targets, margin=1.0):
    B = logits.shape[0]
    correct_scores = logits.gather(1, targets.unsqueeze(1))
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets.unsqueeze(1), False)
    incorrect_logits = logits.masked_select(mask).view(B, -1)
    diff = correct_scores - incorrect_logits
    losses = F.relu(margin - diff)
    return losses.mean()

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, total_epochs, lambda_contrast, use_amp):
    model.train()
    meters = {'total': 0.0, 'ce': 0.0, 'con': 0.0, 'acc': 0.0}
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    
    for batch in pbar:
        x, y, paths = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            # Model forward handles encoder+reasoner internally or we split it
            # FullRAVENModel handles it.
            # But let's verify if create_model returns FullRAVENModel.
            # Yes. But forward returns only logits.
            # If we want contrastive loss, we need logits.
            # Wait, contrastive loss uses logits. That's fine.
            logits = model(x)
            ce_loss = F.cross_entropy(logits, y)
            con_loss = contrastive_loss(logits, y, margin=1.0)
            loss = ce_loss + lambda_contrast * con_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        bs = y.size(0)
        total_samples += bs
        meters['total'] += loss.item() * bs
        meters['ce'] += ce_loss.item() * bs
        meters['con'] += con_loss.item() * bs
        meters['acc'] += (logits.argmax(1) == y).float().sum().item()
        pbar.set_postfix({'loss':f"{meters['total']/total_samples:.3f}", 'acc':f"{meters['acc']/total_samples:.3f}"})
    return {k: v / total_samples for k, v in meters.items()}

def validate(model, dataloader, device, epoch, total_epochs, use_amp):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Val]", leave=False)
    with torch.no_grad():
        for batch in pbar:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            with autocast(enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.set_postfix({'acc': f'{correct/total:.3f}'})
    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/iraven_large')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_contrast', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--no_amp', action='store_true')
    args = parser.parse_args()
    
    set_seed(42)
    device = DEVICE
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp and device == 'cuda'
    
    print("="*60)
    print("Experiment 13: SPATIAL CONVOLUTIONAL REASONER")
    print("="*60)
    
    train_dl, val_dl, test_dl = create_dataloaders_with_meta(Path(args.data_dir), args.batch_size, NUM_WORKERS)
    
    model = create_model(
        model_type='spatial_conv',     # <--- Phase 5 Reasoner
        pretrained_encoder=True,
        freeze_encoder=False,
        use_simple_encoder=False,
        encoder_type='spatial',        # <--- Spatial Encoder (flatten_output=False)
        feature_dim=512,
        hidden_dim=256,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler(enabled=use_amp)
    
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tm = train_epoch(model, train_dl, optimizer, scaler, device, epoch, args.epochs, args.lambda_contrast, use_amp)
        val_loss, val_acc = validate(model, val_dl, device, epoch, args.epochs, use_amp)
        scheduler.step()
        print(f"Epoch {epoch} | Train: {tm['acc']:.1%} (L={tm['total']:.3f}) | Val: {val_acc:.1%}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'spatial_conv_best.pth')
            print(f"  → New best: {val_acc:.1%}")

if __name__ == '__main__':
    main()
