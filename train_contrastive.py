"""
RAVEN RPM Solver - Contrastive Training Script

Train models with contrastive losses for improved accuracy.

Key features:
1. Multi-loss training (CE + Contrastive + Ranking + Consistency)
2. Hard negative mining
3. Support for ContrastiveReasoner and DualContrastReasoner

Usage:
    python train_contrastive.py --data_dir ./data/iraven_large --epochs 15
"""
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, 
    SEED, DEVICE, NUM_WORKERS,
    FEATURE_DIM, HIDDEN_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT,
    LABEL_SMOOTHING, FREEZE_ENCODER, PATIENCE, USE_SIMPLE_ENCODER
)
from models import create_model, CombinedContrastiveLoss, ContrastiveReasoner
from utils import create_dataloaders


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch_contrastive(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedContrastiveLoss,
    device: str,
    epoch: int,
    total_epochs: int,
    use_contrastive_extras: bool = True
) -> tuple:
    """
    Train for one epoch with contrastive losses.
    
    Uses the model's return_extras feature to get intermediate representations
    for computing contrastive and consistency losses.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track individual loss components
    loss_components = {
        'ce': 0, 'contrastive': 0, 'ranking': 0, 'consistency': 0, 'total': 0
    }
    
    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        leave=False,
        ncols=120
    )
    
    for batch in pbar:
        x, y, _ = batch
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Check if model supports return_extras (ContrastiveReasoner)
        if use_contrastive_extras and hasattr(model.reasoner, 'forward') and 'return_extras' in model.reasoner.forward.__code__.co_varnames:
            # Get features from encoder
            ctx_feat, choice_feat = model.encoder(x)
            
            # Get logits and extras from reasoner
            logits, extras = model.reasoner(ctx_feat, choice_feat, return_extras=True)
            
            # Compute combined loss with all components
            loss, loss_dict = criterion(
                logits=logits,
                targets=y,
                context_repr=extras.get('context_repr'),
                choice_reprs=extras.get('choice_reprs'),
                context_features=ctx_feat,
                choice_features=choice_feat,
                rule_predictions=extras.get('rule_predictions')
            )
            
            # Track individual components
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key] * x.size(0)
        else:
            # Fallback: standard training with ranking loss only
            logits = model(x)
            
            # Still use combined loss but with limited components
            loss, loss_dict = criterion(
                logits=logits,
                targets=y
            )
            
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key] * x.size(0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        
        # Update progress bar with current metrics
        curr_loss = total_loss / total
        curr_acc = correct / total
        pbar.set_postfix({
            'loss': f'{curr_loss:.4f}',
            'acc': f'{curr_acc:.3f}',
            'ce': f'{loss_components["ce"]/total:.3f}',
            'rank': f'{loss_components["ranking"]/total:.3f}'
        })
    
    # Normalize loss components
    for key in loss_components:
        loss_components[key] /= total
    
    return total_loss / total, correct / total, loss_components


def validate_contrastive(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    total_epochs: int
) -> tuple:
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch}/{total_epochs} [Val]  ",
        leave=False,
        ncols=100
    )
    
    # Use simple CE for validation (consistent comparison)
    val_criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in pbar:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = val_criterion(logits, y)
            
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
            
            # Update progress bar
            curr_loss = total_loss / total
            curr_acc = correct / total
            pbar.set_postfix({
                'loss': f'{curr_loss:.4f}',
                'acc': f'{curr_acc:.3f}'
            })
    
    return total_loss / total, correct / total


def train_model_contrastive(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    model_name: str,
    save_dir: Path,
    contrastive_weight: float = 0.5,
    ranking_weight: float = 0.3,
    consistency_weight: float = 0.2,
    temperature: float = 0.1,
    margin: float = 1.0,
    patience: int = 10
) -> dict:
    """
    Full training loop with contrastive losses.
    """
    model = model.to(device)
    
    # Get all trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({100*num_trainable/num_total:.1f}%)")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Warmup + Cosine decay scheduler
    warmup_epochs = min(3, epochs // 5)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Combined contrastive loss
    criterion = CombinedContrastiveLoss(
        ce_weight=1.0,
        contrastive_weight=contrastive_weight,
        ranking_weight=ranking_weight,
        consistency_weight=consistency_weight,
        temperature=temperature,
        margin=margin,
        label_smoothing=0.0
    )
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'loss_components': []
    }
    best_val_acc = 0
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    
    print(f"\n{'='*60}")
    print(f"Contrastive Training: {model_name}")
    print(f"{'='*60}")
    print(f"Loss weights: CE=1.0, Contrastive={contrastive_weight}, Ranking={ranking_weight}, Consistency={consistency_weight}")
    
    # Check if model supports contrastive extras
    use_contrastive_extras = isinstance(model.reasoner, ContrastiveReasoner)
    if use_contrastive_extras:
        print("Using full contrastive training with extras")
    else:
        print("Using ranking loss only (model doesn't support contrastive extras)")
    
    epoch_pbar = tqdm(range(epochs), desc="Overall Progress", ncols=100)
    
    for epoch in epoch_pbar:
        train_loss, train_acc, loss_components = train_epoch_contrastive(
            model, train_dl, optimizer, criterion, device,
            epoch=epoch+1, total_epochs=epochs,
            use_contrastive_extras=use_contrastive_extras
        )
        val_loss, val_acc = validate_contrastive(
            model, val_dl, criterion, device,
            epoch=epoch+1, total_epochs=epochs
        )
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['loss_components'].append(loss_components)
        
        epoch_pbar.set_postfix({
            'train_acc': f'{train_acc:.3f}',
            'val_acc': f'{val_acc:.3f}'
        })
        
        # Print detailed summary
        tqdm.write(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
              f"CE: {loss_components['ce']:.3f} Rank: {loss_components['ranking']:.3f}")
        
        # Track best model
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            improved = True
        
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                tqdm.write(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save best model
    save_path = save_dir / f"{model_name.lower().replace(' ', '_')}_contrastive.pth"
    torch.save({
        'model_state_dict': best_state,
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'history': history,
        'contrastive_config': {
            'contrastive_weight': contrastive_weight,
            'ranking_weight': ranking_weight,
            'consistency_weight': consistency_weight,
            'temperature': temperature,
            'margin': margin
        }
    }, save_path)
    print(f"Saved best model to {save_path} (val_acc: {best_val_acc:.4f})")
    
    if best_state:
        model.load_state_dict(best_state)
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train RAVEN models with contrastive losses')
    parser.add_argument('--data_dir', type=str, default='./data/iraven_large',
                        help='Path to data directory')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--model', type=str, default='contrastive',
                        choices=['contrastive', 'dual_contrast', 'transformer', 'relation_net', 'mlp'],
                        help='Model type to train')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze pretrained encoder weights')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Contrastive loss hyperparameters
    parser.add_argument('--contrastive_weight', type=float, default=0.5,
                        help='Weight for contrastive loss')
    parser.add_argument('--ranking_weight', type=float, default=0.3,
                        help='Weight for ranking loss')
    parser.add_argument('--consistency_weight', type=float, default=0.2,
                        help='Weight for consistency loss')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for contrastive loss')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for ranking loss')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = DEVICE
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Create dataloaders
    train_dl, val_dl, test_dl = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS
    )
    
    # Create model
    model = create_model(
        model_type=args.model,
        pretrained_encoder=True,
        freeze_encoder=args.freeze_encoder,
        use_simple_encoder=USE_SIMPLE_ENCODER,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    model_names = {
        'contrastive': 'Contrastive',
        'dual_contrast': 'DualContrast',
        'transformer': 'Transformer',
        'relation_net': 'RelationNet',
        'mlp': 'MLP-Relational'
    }
    name = model_names.get(args.model, args.model)
    
    # Train with contrastive losses
    history = train_model_contrastive(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        model_name=name,
        save_dir=save_dir,
        contrastive_weight=args.contrastive_weight,
        ranking_weight=args.ranking_weight,
        consistency_weight=args.consistency_weight,
        temperature=args.temperature,
        margin=args.margin,
        patience=args.patience
    )
    
    # Save training history
    history_path = save_dir / f'{name.lower()}_contrastive_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("CONTRASTIVE TRAINING COMPLETE")
    print("="*60)
    print(f"\nBest validation accuracy: {max(history['val_acc']):.4f}")
    print(f"Model saved to: {save_dir}")


if __name__ == '__main__':
    main()
