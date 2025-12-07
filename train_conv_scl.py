"""
Experiment 15: Spatial Convolutional Reasoner + Structural Consistency Learning (SCL)

Phase 7:
Combines Phase 5 (Conv Reasoner), Phase 6 (Attr Supervision), and Phase 7 (SCL).

SCL Loss:
Enforces that the rule embedding derived from Row 0 is identical to Row 1.
L_consistency = MSE(emb_row0, emb_row1) + MSE(emb_col0, emb_col1)

This prevents the model from memorizing "valid row visual patterns" and forces it
to extract a consistent abstract rule representation.
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

def train_epoch(model, dataloader, optimizer, scaler, device, epoch, total_epochs, 
                lambda_contrast, lambda_attr, lambda_consistency, use_amp):
    model.train()
    meters = {'total': 0.0, 'ce': 0.0, 'con': 0.0, 'attr': 0.0, 'scl': 0.0, 'acc': 0.0}
    total_samples = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    
    for batch in pbar:
        if len(batch) == 4:
            x, y, paths, meta = batch
        else:
            x, y, paths = batch
            meta = {}
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            context_features, choice_features = model.encoder(x)
            
            # Request extras (embeddings + attributes)
            logits, extras = model.reasoner(context_features, choice_features, return_extras=True)
            
            ce_loss = F.cross_entropy(logits, y)
            con_loss = contrastive_loss(logits, y, margin=1.0)
            
            # Attribute Loss
            attr_loss = torch.tensor(0.0, device=device)
            if lambda_attr > 0 and model.reasoner.compute_attribute_loss:
                attr_loss = model.reasoner.compute_attribute_loss(context_features, meta)
                
            # Consistency Loss (SCL)
            scl_loss = torch.tensor(0.0, device=device)
            if lambda_consistency > 0:
                # MSE(Row0, Row1) + MSE(Col0, Col1)
                # extras contains embeddings: emb_r0, emb_r1, emb_c0, emb_c1
                # They are (B, hidden_dim) or (B*N, hidden) - check
                # Our update used B as batch size in forward loop, but row_conv call inside forward?
                # Inside forward: row0 = ... (B, ...). row_conv returns (B, ...).
                # So inputs are aligned.
                
                loss_r = F.mse_loss(extras['emb_r0'], extras['emb_r1'])
                loss_c = F.mse_loss(extras['emb_c0'], extras['emb_c1'])
                scl_loss = loss_r + loss_c
                
            loss = ce_loss + lambda_contrast * con_loss + lambda_attr * attr_loss + lambda_consistency * scl_loss
        
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
        meters['attr'] += attr_loss.item() * bs
        meters['scl'] += scl_loss.item() * bs
        meters['acc'] += (logits.argmax(1) == y).float().sum().item()
        
        pbar.set_postfix({
            'loss': f"{meters['total']/total_samples:.3f}", 
            'acc': f"{meters['acc']/total_samples:.3f}",
            'scl': f"{meters['scl']/total_samples:.3f}"
        })
        
    return {k: v / total_samples for k, v in meters.items()}

def validate(model, dataloader, device, epoch, total_epochs, use_amp):
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
                x, y, _ = batch
            x, y = x.to(device), y.to(device)
            with autocast(enabled=use_amp):
                # Standard validation (classification only)
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
    parser.add_argument('--lambda_attr', type=float, default=1.0)
    parser.add_argument('--lambda_consistency', type=float, default=1.0) # SCL
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--no_amp', action='store_true')
    args = parser.parse_args()
    
    set_seed(42)
    device = DEVICE
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    use_amp = not args.no_amp and device == 'cuda'
    
    print("="*60)
    print(f"Experiment 15: SCL CONV REASONER (Attr={args.lambda_attr}, SCL={args.lambda_consistency})")
    print("="*60)
    
    train_dl, val_dl, test_dl = create_dataloaders_with_meta(Path(args.data_dir), args.batch_size, NUM_WORKERS)
    
    model = create_model(
        model_type='spatial_conv',     # Phase 5 Reasoner (modified for Phase 7)
        pretrained_encoder=True,
        freeze_encoder=False,
        use_simple_encoder=False,
        encoder_type='spatial',        # Spatial Encoder (Flatten=False)
        feature_dim=512,
        hidden_dim=256,
        dropout=0.5                    # High Dropout from Phase 6
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = GradScaler(enabled=use_amp)
    
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tm = train_epoch(model, train_dl, optimizer, scaler, device, epoch, args.epochs, 
                         args.lambda_contrast, args.lambda_attr, args.lambda_consistency, use_amp)
        val_loss, val_acc = validate(model, val_dl, device, epoch, args.epochs, use_amp)
        scheduler.step()
        print(f"Epoch {epoch} | Train: {tm['acc']:.1%} (L={tm['total']:.3f}, SCL={tm['scl']:.3f}) | Val: {val_acc:.1%}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / 'spatial_conv_scl_best.pth')
            print(f"  → New best: {val_acc:.1%}")

if __name__ == '__main__':
    main()
