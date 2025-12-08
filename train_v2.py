"""
Training Script for Rule-Aware Reasoner V2
==========================================
Experiment 16: Hybrid Spatial SCL

Model: RuleAwareReasonerV2
 Dataset: I-RAVEN
 Losses:
  1. Cross Entropy (Task)
  2. Contrastive (Task)
  3. SCL Consistency (MSE(r0, r1) + MSE(r1, r2_gt))
  4. Attribute Supervision (Auxiliary)

Goal: Maximize validation accuracy by combining spatial handling with strict rule consistency.
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from pathlib import Path

from config import (
    DEVICE, NUM_WORKERS, WEIGHT_DECAY, DROPOUT
)
from models.encoder_spatial import SpatialFeatureExtractor
from models.rule_aware_reasoner_v2 import RuleAwareReasonerV2
from utils.dataset import RAVENDataset, get_split_files, create_dataloaders

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RuleAwareModelV2(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        self.encoder = SpatialFeatureExtractor(pretrained=pretrained, flatten_output=False)
        self.reasoner = RuleAwareReasonerV2(feature_dim=512, hidden_dim=256, dropout=dropout)
        
    def forward(self, x, return_extras=False):
        # x: (B, 16, 1, 160, 160) or (B, 16, 160, 160)
        # Add channel dim if needed
        if x.dim() == 4:
            x = x.unsqueeze(2)
            
        ctx, choices = self.encoder(x, flatten=False) # (B, 8, 512, 5, 5)
        return self.reasoner(ctx, choices, return_extras=return_extras)

def contrastive_loss(logits, targets, margin=1.0):
    """
    Max-margin contrastive loss.
    L = max(0, margin - (score_correct - score_incorrect))
    """
    B = logits.shape[0]
    correct_scores = logits.gather(1, targets.unsqueeze(1))
    
    # Generate mask for incorrect scores
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets.unsqueeze(1), False)
    incorrect_logits = logits.masked_select(mask).view(B, -1)
    
    # Difference: Correct - Incorrect (We want this > margin)
    # L = max(0, margin - (pos - neg))
    diff = correct_scores - incorrect_logits
    losses = F.relu(margin - diff)
    return losses.mean()

def compute_scl_loss(extras, targets, device):
    """
    Consistency Loss:
    1. r0 ~= r1 (Row constraint)
    2. c0 ~= c1 (Col constraint)
    3. r1 ~= r2_correct (Answer constraint)
    """
    loss_row = F.mse_loss(extras['r0'], extras['r1'])
    loss_col = F.mse_loss(extras['c0'], extras['c1'])
    
    # Get r2/c2 embeddings for the *correct* answer
    # extras['r2_stack'] is (B, 8, D)
    B, _, D = extras['r2_stack'].shape
    
    # Gather correct answer embeddings
    r2_correct = extras['r2_stack'].gather(1, targets.view(B, 1, 1).expand(B, 1, D)).squeeze(1)
    c2_correct = extras['c2_stack'].gather(1, targets.view(B, 1, 1).expand(B, 1, D)).squeeze(1)
    
    loss_ans_r = F.mse_loss(r2_correct, extras['r1']) # r2 should match r1
    loss_ans_c = F.mse_loss(c2_correct, extras['c1']) # c2 should match c1
    
    return loss_row + loss_col + loss_ans_r + loss_ans_c

def compute_attr_loss(extras, meta, device):
    """
    Supervised Attribute Loss.
    extras contains 'attr_type', 'attr_size', etc. (B, 8, num_cls)
    meta contains 'context' (B, num_attrs, 8)
    """
    if 'context' not in meta:
        return torch.tensor(0.0, device=device)
        
    context_gt = meta['context'].to(device) # (B, num_attrs, 8)
    total_loss = 0.0
    num_attrs = 0
    
    # Map from output name to index in meta_matrix
    # Order in RAVEN: Type(0), Size(1), Color(2), Number(3), Position(4)..
    # Our heads: type, size, color, number
    # If single object: 0,1,2
    # If multi object: 0, 1, 2, 3? 
    # Let's assume standard order: Type, Size, Color, Number.
    # We will try to match dynamically based on iteration order
    
    keys = ['type', 'size', 'color', 'number']
    
    for idx, key in enumerate(keys):
        pred_key = f'attr_{key}'
        if pred_key in extras and idx < context_gt.shape[1]:
            logits = extras[pred_key] # (B, 8, C)
            gt = context_gt[:, idx, :] # (B, 8)
            num_cls = logits.shape[-1]
            
            # Clamp GT just in case
            gt = gt.clamp(0, num_cls - 1)
            
            loss = F.cross_entropy(
                logits.reshape(-1, num_cls),
                gt.reshape(-1),
                ignore_index=-1
            )
            total_loss += loss
            num_attrs += 1
            
    return total_loss / max(num_attrs, 1)

def train_epoch(model, dataloader, optimizer, scaler, device, config):
    model.train()
    meters = {'total': 0.0, 'scl': 0.0, 'acc': 0.0}
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for x, y, _, meta in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=True):
            logits, extras = model(x, return_extras=True)
            
            # 1. CE Loss
            ce_loss = F.cross_entropy(logits, y)
            
            # 2. Contrastive Loss
            con_loss = contrastive_loss(logits, y) if config['lambda_con'] > 0 else 0
            
            # 3. SCL Loss
            scl_loss = compute_scl_loss(extras, y, device) if config['lambda_scl'] > 0 else 0
            
            # 4. Attribute Loss
            attr_loss = compute_attr_loss(extras, meta, device) if config['lambda_attr'] > 0 else 0
            
            total_loss = ce_loss + \
                         config['lambda_con'] * con_loss + \
                         config['lambda_scl'] * scl_loss + \
                         config['lambda_attr'] * attr_loss
                         
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        bs = x.size(0)
        total_samples += bs
        meters['total'] += total_loss.item() * bs
        meters['scl'] += (scl_loss.item() if isinstance(scl_loss, torch.Tensor) else 0) * bs
        meters['attr'] = meters.get('attr', 0.0) + (attr_loss.item() if isinstance(attr_loss, torch.Tensor) else 0) * bs
        meters['acc'] += (logits.argmax(1) == y).sum().item()
        
        pbar.set_postfix({
            'L': f"{meters['total']/total_samples:.2f}", 
            'Acc': f"{meters['acc']/total_samples:.1%}",
            'SCL': f"{meters['scl']/total_samples:.2f}",
            'Attr': f"{meters['attr']/total_samples:.2f}"
        })
        
    return {k: v / total_samples for k, v in meters.items()}

def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y, _, _ in tqdm(dataloader, desc="Val", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/iraven_large')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_con', type=float, default=1.0)
    parser.add_argument('--lambda_scl', type=float, default=1.0)
    parser.add_argument('--lambda_attr', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4) # Added workers argument
    args = parser.parse_args()
    
    device = DEVICE
    set_seed(42)
    
    print(f"Dataset: {args.data_dir}")
    # Data Loaders
    # Enable strong augmentation for training
    train_dl, val_dl, test_dl = create_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.workers,
        return_meta=True
    )
    # Note: dataset.py handles "augment=True" internally for train set if we use create_dataloaders?
    # Let's check utils/dataset.py. Yes, create_dataloaders sets augment=True for train_ds.
    
    model = RuleAwareReasonerV2(device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # Added Weight Decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    config = {
        'lambda_con': args.lambda_con,
        'lambda_scl': args.lambda_scl,
        'lambda_attr': args.lambda_attr,
    }
    
    best_val = 0.0
    
    print("Starting Training...")
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, train_dl, optimizer, scaler, device, config)
        val_acc = validate(model, val_dl, device)
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Acc {metrics['acc']:.2%} (Loss {metrics['total']:.3f}) | Val Acc {val_acc:.2%}")
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), f"saved_models/rule_aware_v2_best.pth")
            print(f"  -> New Best Saved: {val_acc:.2%}")
            
if __name__ == "__main__":
    main()
