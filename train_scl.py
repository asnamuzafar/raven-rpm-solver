"""
SCL-Inspired Training Script for RAVEN

Implements:
1. Contrastive loss (margin ranking between correct and incorrect)
2. Cross-entropy loss for classification
3. Attribute supervision loss (ground-truth attributes)
4. Mixed precision training for speed on RTX 3090
5. Auto-stop monitoring

Usage:
    python train_scl.py --data_dir ./data/iraven_large --epochs 30 --batch_size 64
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import DEVICE, NUM_WORKERS, WEIGHT_DECAY, DROPOUT
from models.scl_reasoner import create_scl_model
from utils.dataset import create_dataloaders


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def contrastive_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """Margin ranking loss."""
    B = logits.shape[0]
    correct_scores = logits.gather(1, targets.unsqueeze(1))
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets.unsqueeze(1), False)
    incorrect_logits = logits.masked_select(mask).view(B, -1)
    diff = correct_scores - incorrect_logits
    losses = F.relu(margin - diff)
    return losses.mean()


def compute_attribute_loss(
    attr_preds: dict,
    meta: dict,
    device: str
) -> torch.Tensor:
    """Compute attribute prediction loss."""
    if not meta or 'context' not in meta:
        return torch.tensor(0.0, device=device)
    
    # meta['context'] shape: (B, num_attrs, 8)
    context_gt = meta['context'].to(device)
    total_loss = torch.tensor(0.0, device=device)
    num_calc = 0
    
    # Mapping from index to head name in SCLReasoner
    # 0: Shape, 1: Size, 2: Color, 3: Number
    idx_to_name = {
        0: 'shape',
        1: 'obj_size',
        2: 'obj_color',
        3: 'obj_number'
    }
    
    for idx_attr, name in idx_to_name.items():
        if name in attr_preds and idx_attr < context_gt.shape[1]:
            # Preds: (B, 8, num_classes)
            logits = attr_preds[name]
            # Targets: (B, 8)
            targets = context_gt[:, idx_attr, :]
            
            # Use valid targets only (clamp to range)
            num_classes = logits.shape[-1]
            targets_clamped = targets.clamp(0, num_classes - 1)
            
            loss = F.cross_entropy(
                logits.reshape(-1, num_classes),
                targets_clamped.reshape(-1),
                ignore_index=-1
            )
            total_loss += loss
            num_calc += 1
            
    return total_loss / max(num_calc, 1)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    epoch: int,
    total_epochs: int,
    lambda_contrast: float = 0.5,
    lambda_attr: float = 0.5,
    use_amp: bool = True
):
    """Train for one epoch with multi-task losses."""
    model.train()
    
    meters = {'total': 0.0, 'ce': 0.0, 'con': 0.0, 'attr': 0.0, 'acc': 0.0}
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    
    for batch in pbar:
        # Check batch structure
        if len(batch) == 4:
            x, y, _, meta = batch
        else:
            x, y, _ = batch[:3]
            meta = {}
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            # Forward pass returning extras (attr predictions)
            logits, extras = model(x, return_extras=True)
            
            # 1. Main Classification Loss
            ce_loss = F.cross_entropy(logits, y)
            
            # 2. Contrastive Loss
            con_loss = contrastive_loss(logits, y, margin=1.0)
            
            # 3. Attribute Supervision Loss
            attr_loss = torch.tensor(0.0, device=device)
            if 'attr_preds' in extras:
                attr_loss = compute_attribute_loss(extras['attr_preds'], meta, device)
            
            # Combined
            loss = ce_loss + (lambda_contrast * con_loss) + (lambda_attr * attr_loss)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        current_bs = y.size(0)
        total_samples += current_bs
        meters['total'] += loss.item() * current_bs
        meters['ce'] += ce_loss.item() * current_bs
        meters['con'] += con_loss.item() * current_bs
        meters['attr'] += attr_loss.item() * current_bs
        
        acc = (logits.argmax(1) == y).float().sum().item()
        meters['acc'] += acc
        
        # Display average
        pbar.set_postfix({
            'loss': f"{meters['total']/total_samples:.3f}",
            'acc': f"{meters['acc']/total_samples:.3f}",
            'attr': f"{meters['attr']/total_samples:.3f}"
        })
    
    # Return averages
    return {k: v / total_samples for k, v in meters.items()}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    epoch: int,
    total_epochs: int,
    use_amp: bool = True
):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Val]", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            if len(batch) == 4:
                x, y, _, _ = batch
            else:
                x, y, _ = batch[:3]
            
            x, y = x.to(device), y.to(device)
            
            with autocast(enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.3f}',
                'acc': f'{correct/total:.3f}'
            })
    
    return total_loss / total, correct / total


def check_learning(history: dict, patience: int = 4) -> tuple:
    """Stops if no improvement."""
    if len(history['val_acc']) < 5:
        return True, "warmup"
    
    recent = history['val_acc'][-patience:]
    best_old = max(history['val_acc'][:-patience])
    best_recent = max(recent)
    
    # Needs to beat previous best by 0.5%
    if best_recent < best_old + 0.005:
        return False, f"stagnated (best: {best_old:.1%})"
        
    return True, "learning"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/iraven_large')
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_contrast', type=float, default=0.5)
    parser.add_argument('--lambda_attr', type=float, default=1.0)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--no_amp', action='store_true')
    args = parser.parse_args()
    
    set_seed(42)
    device = DEVICE
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp and device == 'cuda'
    
    print(f"Training SCL with Attribute Supervision (λ_attr={args.lambda_attr})")
    
    # Use create_dataloaders to ensure correct metadata collation
    train_dl, val_dl, test_dl = create_dataloaders(
        Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        return_meta=True  # Crucial for attribute loss
    )
    
    model = create_scl_model(
        pretrained_encoder=True,
        freeze_encoder=args.freeze_encoder
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler(enabled=use_amp)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    print("\nStarting Training...")
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_dl, optimizer, scaler, device, epoch, args.epochs,
            args.lambda_contrast, args.lambda_attr, use_amp
        )
        val_loss, val_acc = validate(model, val_dl, device, epoch, args.epochs, use_amp)
        scheduler.step()
        
        history['train_loss'].append(train_metrics['total'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch} | Train: {train_metrics['acc']:.1%} (Loss: {train_metrics['total']:.3f}, Attr: {train_metrics['attr']:.3f}) | Val: {val_acc:.1%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'scl_attr_best.pth')
            print(f"  → New best: {val_acc:.1%}")
            
        cont, reason = check_learning(history)
        if not cont:
            print(f"Stopping: {reason}")
            break

    print(f"Done. Best Acc: {best_acc:.1%}")
    with open(save_dir / 'history_scl_attr.json', 'w') as f:
        json.dump(history, f)

if __name__ == '__main__':
    main()
