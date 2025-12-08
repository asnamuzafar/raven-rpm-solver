"""
I-RAVEN Training Script
========================
Unified training script for all I-RAVEN experiments.

Usage:
    python train_iraven.py --model spatial_conv --epochs 20
    python train_iraven.py --model neuro_symbolic --epochs 20
    python train_iraven.py --model transformer --epochs 20
    python train_iraven.py --model relation_net --epochs 20  

Available Models:
    - spatial_conv   : Spatial Conv Reasoner - BEST (34.0% val accuracy)
    - neuro_symbolic : Rule-Aware Reasoner (32.9% val accuracy)  
    - transformer    : Transformer-based reasoner (26.9%)
    - relation_net   : Relation Network with row/col/diag (26.2%)
    - cnn_direct     : Simple CNN baseline

Available Encoders:
    - resnet       : ResNet-18 pretrained (recommended for most models)
    - spatial      : Spatial ResNet (required for spatial_conv)
    - simple       : Custom CNN from scratch
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

# Import from existing model modules
from config import DEVICE, NUM_WORKERS
from utils.dataset import create_dataloaders
from models.encoder import ResNetVisualEncoder, SimpleConvEncoder, RAVENFeatureExtractor
from models.encoder_spatial import SpatialFeatureExtractor
from models.baselines import RelationNetwork, CNNDirectBaseline
from models.reasoner import TransformerReasoner, MLPRelationalReasoner
from models.reasoner_spatial_conv import SpatialConvolutionalReasoner


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Combined Models
# ============================================================================

class NeuroSymbolicReasoner(nn.Module):
    """
    Rule-Aware Reasoner with consistency constraints.
    Encodes row/column triplets as rules and scores based on rule matching.
    Best performing model on I-RAVEN (32.9% val accuracy).
    """
    def __init__(self, input_dim=512, hidden_dim=256, rule_dim=128, dropout=0.5):
        super().__init__()
        
        self.rule_net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rule_dim)
        )
        
        self.validity = nn.Sequential(
            nn.Linear(rule_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def get_rule(self, p1, p2, p3):
        return self.rule_net(torch.cat([p1, p2, p3], dim=-1))
        
    def forward(self, context_features, choice_features):
        # context: (B, 8, D), choices: (B, 8, D)
        p = [context_features[:, i] for i in range(8)]
        
        # Context rules
        r0 = self.get_rule(p[0], p[1], p[2])
        r1 = self.get_rule(p[3], p[4], p[5])
        c0 = self.get_rule(p[0], p[3], p[6])
        c1 = self.get_rule(p[1], p[4], p[7])
        
        scores = []
        for i in range(8):
            choice = choice_features[:, i]
            r2 = self.get_rule(p[6], p[7], choice)
            c2 = self.get_rule(p[2], p[5], choice)
            
            v_score = self.validity(r2) + self.validity(c2)
            c_score = -F.mse_loss(r2, r1, reduction='none').mean(1, keepdim=True) \
                      -F.mse_loss(c2, c1, reduction='none').mean(1, keepdim=True)
            scores.append(v_score + c_score)
            
        return torch.cat(scores, dim=1)


class IRAVENModel(nn.Module):
    """
    Complete I-RAVEN model combining encoder + reasoner.
    
    For spatial_conv model, uses SpatialFeatureExtractor encoder which preserves
    the 5x5 spatial feature maps instead of flattening.
    """
    def __init__(self, model_type='transformer', encoder_type='resnet', 
                 freeze_encoder=False, dropout=0.4):
        super().__init__()
        
        self.model_type = model_type
        self.use_spatial = (model_type == 'spatial_conv')
        
        # Encoder
        if self.use_spatial or encoder_type == 'spatial':
            # Spatial encoder preserves 5x5 feature maps
            self.encoder = SpatialFeatureExtractor(pretrained=True, flatten_output=False)
            feature_dim = 512
            self.spatial_dim = (5, 5)
        elif encoder_type == 'resnet':
            self.encoder = ResNetVisualEncoder(pretrained=True)
            feature_dim = 512
        elif encoder_type == 'simple':
            self.encoder = SimpleConvEncoder(feature_dim=256)
            feature_dim = 256
        else:
            self.encoder = ResNetVisualEncoder(pretrained=True)
            feature_dim = 512
            
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Reasoner
        if model_type == 'spatial_conv':
            # BEST MODEL: 34% val accuracy
            self.reasoner = SpatialConvolutionalReasoner(feature_dim=feature_dim, dropout=dropout)
        elif model_type == 'transformer':
            self.reasoner = TransformerReasoner(feature_dim=feature_dim, dropout=dropout)
        elif model_type == 'relation_net':
            self.reasoner = RelationNetwork(feature_dim=feature_dim)
        elif model_type == 'neuro_symbolic':
            self.reasoner = NeuroSymbolicReasoner(input_dim=feature_dim, dropout=dropout)
        elif model_type == 'cnn_direct':
            self.reasoner = CNNDirectBaseline(feature_dim=feature_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.feature_dim = feature_dim
            
    def forward(self, x):
        # x: (B, 16, 160, 160) or (B, 16, 1, 160, 160)
        if x.dim() == 5:
            x = x.squeeze(2)  # Remove channel dim if present: (B, 16, 1, H, W) -> (B, 16, H, W)
            
        B = x.shape[0]
        
        if self.use_spatial:
            # SpatialFeatureExtractor takes full input and returns (context, choices)
            context, choices = self.encoder(x)
            # context: (B, 8, 512, 5, 5), choices: (B, 8, 512, 5, 5)
        else:
            # Standard encoders: process each panel individually
            if x.dim() == 4:
                x = x.unsqueeze(2)  # (B, 16, H, W) -> (B, 16, 1, H, W)
            
            B, N, C, H, W = x.shape
            x_flat = x.view(B * N, C, H, W)
            features = self.encoder(x_flat)
            features = features.view(B, N, -1)
            context = features[:, :8]
            choices = features[:, 8:]
        
        return self.reasoner(context, choices)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images, targets = batch[0].to(device), batch[1].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100*total_correct/total_samples:.1f}%'})
    
    return total_loss / total_samples, total_correct / total_samples


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            images, targets = batch[0].to(device), batch[1].to(device)
            
            with autocast():
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
            
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)
    
    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser(
        description='Train models on I-RAVEN dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_iraven.py --model transformer --epochs 20
  python train_iraven.py --model neuro_symbolic --epochs 30 --dropout 0.5
  python train_iraven.py --model relation_net --freeze_encoder
        """
    )
    
    # Dataset
    parser.add_argument('--data_dir', type=str, default='./data/iraven_large',
                        help='Path to I-RAVEN dataset')
    parser.add_argument('--save_dir', type=str, default='./saved_models',
                        help='Directory to save models')
    
    # Model
    parser.add_argument('--model', type=str, default='spatial_conv',
                        choices=['spatial_conv', 'neuro_symbolic', 'transformer', 'relation_net', 'cnn_direct'],
                        help='Model architecture (spatial_conv is best: 34%%)')
    parser.add_argument('--encoder', type=str, default='resnet',
                        choices=['resnet', 'spatial', 'simple'],
                        help='Visual encoder type (auto-set for spatial_conv)')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze pretrained encoder weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"I-RAVEN Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Encoder: {args.encoder}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    model = IRAVENModel(
        model_type=args.model,
        encoder_type=args.encoder,
        freeze_encoder=args.freeze_encoder,
        dropout=args.dropout
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # Training
    best_val_acc = 0
    patience_counter = 0
    history = []
    
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()}")
    print(f"{'='*60}")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {100*train_acc:.1f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {100*val_acc:.1f}%")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'iraven_{args.model}_best.pt'))
            print(f"  ✓ New best! Val Acc: {100*val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Test evaluation
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}")
    
    checkpoint = torch.load(os.path.join(args.save_dir, f'iraven_{args.model}_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, device)
    print(f"Test Accuracy: {100*test_acc:.1f}%")
    
    # Save history
    with open(os.path.join(args.save_dir, f'iraven_{args.model}_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
