"""
RAVEN RPM Solver - Training Script

Train all reasoning models on the RAVEN dataset.

Usage:
    python train.py --data_dir ./data/raven_medium --epochs 30
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
# Import encoder LR if available, otherwise use same as main LR
try:
    from config import ENCODER_LR
except ImportError:
    ENCODER_LR = LEARNING_RATE
from models import create_model, FullRAVENModel
from utils import create_dataloaders


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
    total_epochs: int
) -> tuple:
    """Train for one epoch with progress bar"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress bar for batches within epoch
    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        leave=False,
        ncols=100
    )
    
    for batch in pbar:
        x, y, _ = batch
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
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
            'acc': f'{curr_acc:.3f}'
        })
    
    return total_loss / total, correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    total_epochs: int
) -> tuple:
    """Validate the model with progress bar"""
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
    
    with torch.no_grad():
        for batch in pbar:
            x, y, _ = batch
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
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


def train_model(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    model_name: str,
    save_dir: Path,
    label_smoothing: float = 0.0,
    patience: int = 7
) -> dict:
    """Full training loop for a model with early stopping"""
    model = model.to(device)
    
    # Get all trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({100*num_trainable/num_total:.1f}%)")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Warmup + Cosine decay scheduler for stable training
    warmup_epochs = min(3, epochs // 5)  # 3 epochs warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0
    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Overall progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Overall Progress", ncols=100)
    
    for epoch in epoch_pbar:
        train_loss, train_acc = train_epoch(
            model, train_dl, optimizer, criterion, device,
            epoch=epoch+1, total_epochs=epochs
        )
        val_loss, val_acc = validate(
            model, val_dl, criterion, device,
            epoch=epoch+1, total_epochs=epochs
        )
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_acc': f'{train_acc:.3f}',
            'val_acc': f'{val_acc:.3f}'
        })
        
        # Print summary for this epoch
        tqdm.write(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")
        
        # Track best model by BOTH validation loss and accuracy
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            improved = True
        
        # Early stopping based on validation loss
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                tqdm.write(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    # Save best model
    save_path = save_dir / f"{model_name.lower().replace(' ', '_')}_model.pth"
    torch.save({
        'model_state_dict': best_state,
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'history': history
    }, save_path)
    print(f"Saved best model to {save_path} (val_acc: {best_val_acc:.4f}, val_loss: {best_val_loss:.4f})")
    
    # Load best weights
    model.load_state_dict(best_state)
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train RAVEN models')
    parser.add_argument('--dataset', type=str, default='raven',
                        choices=['raven', 'iraven'],
                        help='Dataset type: raven (original) or iraven (bias-corrected)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (overrides --dataset if provided)')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['transformer', 'mlp', 'relation_net', 'cnn_direct'],
                        help='Models to train')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')
    parser.add_argument('--freeze_encoder', action='store_true', default=FREEZE_ENCODER,
                        help='Freeze pretrained encoder weights')
    parser.add_argument('--label_smoothing', type=float, default=LABEL_SMOOTHING,
                        help='Label smoothing factor')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help='Early stopping patience')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = DEVICE
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = f'./data/{args.dataset}_medium'
    
    print(f"Device: {device}")
    print(f"Dataset type: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Freeze encoder: {args.freeze_encoder}") 
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Early stopping patience: {args.patience}")
    
    # Create dataloaders
    train_dl, val_dl, test_dl = create_dataloaders(
        data_dir,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS
    )
    
    # Train each model
    all_histories = {}
    all_models = {}
    
    model_names = {
        'transformer': 'Transformer',
        'mlp': 'MLP-Relational',
        'cnn_direct': 'CNN-Direct',
        'relation_net': 'RelationNet',
        'hybrid': 'Hybrid'
    }
    
    for model_type in args.models:
        name = model_names.get(model_type, model_type)
        model = create_model(
            model_type=model_type, 
            pretrained_encoder=True,
            freeze_encoder=args.freeze_encoder,
            use_simple_encoder=USE_SIMPLE_ENCODER,
            feature_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
        
        history = train_model(
            model=model,
            train_dl=train_dl,
            val_dl=val_dl,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            model_name=name,
            save_dir=save_dir,
            label_smoothing=args.label_smoothing,
            patience=args.patience
        )
        
        all_histories[name] = history
        all_models[name] = model
    
    # Save training histories
    histories_path = save_dir / 'training_histories.json'
    with open(histories_path, 'w') as f:
        json.dump(all_histories, f, indent=2)
    print(f"\nTraining histories saved to {histories_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {save_dir}")
    print("\nNext steps:")
    print("  1. Run evaluation: python evaluate.py")
    print("  2. Launch simulator: streamlit run raven_simulator.py")


if __name__ == '__main__':
    main()

