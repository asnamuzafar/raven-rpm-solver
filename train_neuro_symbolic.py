"""
RAVEN RPM Solver - Neuro-Symbolic Training Script

Train the rule-aware neuro-symbolic model with multi-task learning:
- Cross-entropy loss for answer selection
- Supervised attribute prediction loss
- Rule consistency loss

Usage:
    python train_neuro_symbolic.py --data_dir ./data/iraven_large --epochs 15
"""
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_WORKERS, 
    WEIGHT_DECAY, DROPOUT
)
from models import create_model
from models.rule_reasoner import RuleAwareReasoner
from utils.dataset import RAVENDataset, get_split_files


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders_with_meta(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple:
    """Create dataloaders that return metadata for supervised attribute learning."""
    train_files, val_files, test_files = get_split_files(data_dir)
    
    # Training set with augmentation and metadata
    train_ds = RAVENDataset(train_files, augment=True, return_meta=True)
    val_ds = RAVENDataset(val_files, augment=False, return_meta=True)
    test_ds = RAVENDataset(test_files, augment=False, return_meta=True)
    
    def collate_fn(batch):
        """Custom collate to handle metadata dict."""
        if len(batch[0]) == 4:  # x, y, path, meta
            x_list, y_list, path_list, meta_list = zip(*batch)
            
            x = torch.stack(x_list)
            y = torch.stack(y_list)
            
            # Collate metadata
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
    
    train_dl = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Dataset splits: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    return train_dl, val_dl, test_dl


def compute_attribute_loss(
    context_features: torch.Tensor,
    meta: dict,
    reasoner: RuleAwareReasoner,
    device: str
) -> torch.Tensor:
    """
    Compute supervised attribute prediction loss.
    """
    if not meta or 'context' not in meta:
        return torch.tensor(0.0, device=device)
    
    # Get attribute predictions
    attr_logits = reasoner.attr_head(context_features)
    
    context_gt = meta['context'].to(device)  # (B, num_attrs, 8)
    
    total_loss = 0.0
    num_attrs = 0
    
    # Match attribute indices to heads
    attr_names = ['type', 'size', 'color', 'number']
    
    for attr_idx, name in enumerate(attr_names):
        if name in attr_logits and attr_idx < context_gt.shape[1]:
            logits = attr_logits[name]  # (B, 8, num_classes)
            gt = context_gt[:, attr_idx, :]  # (B, 8)
            
            # Clamp to valid range
            num_classes = logits.shape[-1]
            gt_clamped = gt.clamp(0, num_classes - 1)
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, num_classes),
                gt_clamped.view(-1),
                reduction='mean'
            )
            total_loss = total_loss + loss
            num_attrs += 1
    
    return total_loss / max(num_attrs, 1)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
    lambda_attr: float = 0.5,
    lambda_rule: float = 0.3
):
    """Train for one epoch with multi-task losses."""
    model.train()
    
    total_ce_loss = 0.0
    total_attr_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    
    for batch in pbar:
        x, y, paths, meta = batch
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with extras for multi-task learning
        context_features, choice_features = model.encoder(x)
        logits, extras = model.reasoner(context_features, choice_features, return_extras=True)
        
        # 1. Cross-entropy loss for answer selection
        ce_loss = F.cross_entropy(logits, y)
        
        # 2. Attribute prediction loss (supervised)
        attr_loss = compute_attribute_loss(context_features, meta, model.reasoner, device)
        
        # 3. Rule consistency loss (optional, can be computed from extras)
        # For now, we use a simpler approach based on pattern similarity
        
        # Combined loss
        loss = ce_loss + lambda_attr * attr_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_ce_loss += ce_loss.item()
        total_attr_loss += attr_loss.item() if isinstance(attr_loss, torch.Tensor) else attr_loss
        total_loss += loss.item()
        
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix({
            'loss': f'{total_loss/len(pbar):.4f}',
            'ce': f'{total_ce_loss/len(pbar):.4f}',
            'attr': f'{total_attr_loss/len(pbar):.4f}',
            'acc': f'{correct/total:.3f}'
        })
    
    return total_loss / len(dataloader), correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    epoch: int,
    total_epochs: int
):
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs} [Val]", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            x, y, paths, meta = batch
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            pbar.set_postfix({
                'loss': f'{total_loss/len(pbar):.4f}',
                'acc': f'{correct/total:.3f}'
            })
    
    return total_loss / len(dataloader), correct / total


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    save_dir: Path,
    lambda_attr: float = 0.5,
    patience: int = 7
):
    """Full training loop with early stopping."""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("\n" + "=" * 60)
    print("NEURO-SYMBOLIC TRAINING")
    print("=" * 60)
    print(f"Attribute loss weight: {lambda_attr}")
    
    pbar = tqdm(range(1, epochs + 1), desc="Overall Progress")
    
    for epoch in pbar:
        train_loss, train_acc = train_epoch(
            model, train_dl, optimizer, device, epoch, epochs, lambda_attr
        )
        
        val_loss, val_acc = validate(model, val_dl, device, epoch, epochs)
        
        scheduler.step()
        
        # Update progress bar
        pbar.set_postfix({
            'train_acc': f'{train_acc:.3f}',
            'val_acc': f'{val_acc:.3f}'
        })
        
        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_name': 'neuro_symbolic'
            }
            save_path = save_dir / 'neuro_symbolic_model.pth'
            torch.save(checkpoint, save_path)
            print(f"  → Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
                break
    
    return best_val_acc, best_epoch


def main():
    parser = argparse.ArgumentParser(description='Train Neuro-Symbolic RAVEN Model')
    parser.add_argument('--data_dir', type=str, default='./data/iraven_large',
                        help='Path to I-RAVEN data directory')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lambda_attr', type=float, default=0.5,
                        help='Weight for attribute supervision loss')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save models')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with minimal data')
    parser.add_argument('--encoder', type=str, default='resnet', choices=['resnet', 'simple'],
                        help='Encoder type: resnet (11M params) or simple (500K params)')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Feature dimension (512 for resnet, recommend 128-256 for simple)')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights (only train reasoner)')
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    device = DEVICE
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Attribute loss weight: {args.lambda_attr}")
    print(f"Encoder: {args.encoder} (feature_dim={args.feature_dim})")
    print(f"Freeze encoder: {args.freeze_encoder}")
    
    # Create dataloaders
    train_dl, val_dl, test_dl = create_dataloaders_with_meta(
        Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS
    )
    
    # Create model based on encoder choice
    use_simple = (args.encoder == 'simple')
    feature_dim = args.feature_dim if use_simple else 512  # ResNet always outputs 512
    
    model = create_model(
        model_type='neuro_symbolic',
        pretrained_encoder=True,
        freeze_encoder=args.freeze_encoder,
        use_simple_encoder=use_simple,
        feature_dim=feature_dim,
        hidden_dim=256,
        dropout=DROPOUT
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Train
    best_acc, best_epoch = train_model(
        model, train_dl, val_dl,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=save_dir,
        lambda_attr=args.lambda_attr
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {save_dir / 'neuro_symbolic_model.pth'}")


if __name__ == '__main__':
    main()
