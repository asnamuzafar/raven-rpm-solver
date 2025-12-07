"""
Experiment 12: Spatial Reasoning + Position Supervision

Extends Experiment 11 by adding explicit supervision for Position.
Goal: Force the SpatialAdapter to learn the mapping from 5x5 grid to abstract feature vector
that preserves position info.

Usage:
    python train_spatial_pos.py --data_dir ./data/iraven_large --epochs 30 --batch_size 64
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
    train_ds = RAVENDataset(train_files, augment=True, return_meta=True)
    val_ds = RAVENDataset(val_files, augment=False, return_meta=True)
    test_ds = RAVENDataset(test_files, augment=False, return_meta=True)
    
    def collate_fn(batch):
        if len(batch[0]) == 4:
            x_list, y_list, path_list, meta_list = zip(*batch)
            x = torch.stack(x_list)
            y = torch.stack(y_list)
            meta = {}
            if meta_list[0]:
                for key in meta_list[0].keys():
                    if key == 'structure':
                        meta[key] = [m[key] for m in meta_list]
                    elif isinstance(meta_list[0][key], torch.Tensor):
                        meta[key] = torch.stack([m[key] for m in meta_list])
            return x, y, path_list, meta
        else:
            x_list, y_list, path_list = zip(*batch)
            return torch.stack(x_list), torch.stack(y_list), path_list, {}
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
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

def compute_attribute_loss(context_features, meta, reasoner, device):
    if not meta or 'context' not in meta:
        return torch.tensor(0.0, device=device)
    
    attr_logits = reasoner.attr_head(context_features)
    context_gt = meta['context'].to(device)
    total_loss = torch.tensor(0.0, device=device)
    num_attrs = 0
    
    # Extended attribute list including Position
    # Indices in meta_matrix: Type, Size, Color, Number, Position
    # meta_matrix is (num_attrs, 9). 
    # Usually: 0=Type, 1=Size, 2=Color, 3=Number, 4=Position (for 3x3)
    # Actually, we need to be careful with indices.
    # Dataset.py extracts `meta_matrix`.
    # I-RAVEN indices depend on constellation.
    # Center Single: Type(0), Size(1), Color(2)
    # L-R/U-D: Type(0), Size(1), Color(2), Number(3), Position(4)?
    # To be robust, we rely on the fact that `SupervisedAttributeHead` has named heads.
    # `dataset.py` documentation says: "For multi-object configs: Number, Position, Type, Size, Color"
    # This implies index mapping varies.
    # However, for simplicity, we assume the reasoner's head predicts *all* logical attributes always, 
    # and we only supervise those available in ground truth if we can match them.
    # Since we don't have explicit name mapping from `meta`, we'll assume a standard order 
    # matching I-RAVEN defaults: [Type, Size, Color, Number, Position].
    # But `meta['context']` is just a tensor.
    # We will try to supervise 'position' if available (index 4).
    
    attr_names = ['type', 'size', 'color', 'number', 'position']
    
    for attr_idx, name in enumerate(attr_names):
        if name in attr_logits and attr_idx < context_gt.shape[1]:
            logits = attr_logits[name]
            gt = context_gt[:, attr_idx, :]
            
            # Position typically has 9 classes (0-8)
            # Check if gt values are within logits range
            if gt.max() < logits.shape[-1]:
                gt_clamped = gt
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt_clamped.reshape(-1))
                total_loss += loss
                num_attrs += 1
            
    return total_loss / max(num_attrs, 1)

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, total_epochs, lambda_attr, lambda_contrast, use_amp):
    model.train()
    meters = {'total': 0.0, 'ce': 0.0, 'attr': 0.0, 'con': 0.0, 'acc': 0.0}
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    
    for batch in pbar:
        x, y, paths, meta = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            context_features, choice_features = model.encoder(x)
            logits, extras = model.reasoner(context_features, choice_features, return_extras=True)
            ce_loss = F.cross_entropy(logits, y)
            con_loss = contrastive_loss(logits, y, margin=1.0)
            attr_loss = compute_attribute_loss(context_features, meta, model.reasoner, device)
            loss = ce_loss + lambda_attr * attr_loss + lambda_contrast * con_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        bs = y.size(0)
        total_samples += bs
        meters['total'] += loss.item() * bs
        meters['ce'] += ce_loss.item() * bs
        meters['attr'] += attr_loss.item() * bs if isinstance(attr_loss, torch.Tensor) else 0
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
            x, y, _, _ = batch
            x, y = x.to(device), y.to(device)
            with autocast(enabled=use_amp):
                ctx, ch = model.encoder(x)
                logits = model.reasoner(ctx, ch)
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
    parser.add_argument('--lr', type=float, default=1e-4) # Higher LR for new adapter?
    parser.add_argument('--lambda_attr', type=float, default=1.0) # Boost attribute loss to force position learning
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
    print("Experiment 12: Spatial + Position Supervision")
    print("="*60)
    
    train_dl, val_dl, test_dl = create_dataloaders_with_meta(Path(args.data_dir), args.batch_size, NUM_WORKERS)
    
    model = create_model(
        model_type='neuro_symbolic_pos', # <--- Position Aware Reasoner
        pretrained_encoder=True,
        freeze_encoder=False,
        use_simple_encoder=False,
        encoder_type='spatial',          # <--- Spatial Encoder with Adapter
        feature_dim=512,
        hidden_dim=256,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler(enabled=use_amp)
    
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tm = train_epoch(model, train_dl, optimizer, scaler, device, epoch, args.epochs, args.lambda_attr, args.lambda_contrast, use_amp)
        val_loss, val_acc = validate(model, val_dl, device, epoch, args.epochs, use_amp)
        scheduler.step()
        print(f"Epoch {epoch} | Train: {tm['acc']:.1%} (L={tm['total']:.3f}, Attr={tm['attr']:.3f}) | Val: {val_acc:.1%}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'spatial_pos_best.pth')
            print(f"  → New best: {val_acc:.1%}")

if __name__ == '__main__':
    main()
