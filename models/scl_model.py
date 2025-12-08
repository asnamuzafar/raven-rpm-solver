"""
SCL-Inspired Model for RAVEN Puzzles
=====================================
Implements key insights from "Scattering Compositional Learner" (SCL):
1. Scattering Transform: Divide image into patches, process each with SHARED weights
2. Compositional Reasoning: Learn row/column rules from triplets
3. Consistency Scoring: Match candidate answers to learned rules

Key Architectural Choices:
- Patch-based encoding (strong inductive bias for object-invariance)
- Shared weights across patches (forces generalization)
- Explicit row/column rule learning
- Aggressive regularization (dropout, weight decay)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEncoder(nn.Module):
    """
    Scattering-style patch encoder.
    Divides input image into patches and processes each with SHARED weights.
    
    Input: (B, 1, 160, 160)
    Output: (B, num_patches, embed_dim)
    """
    def __init__(self, patch_size=32, embed_dim=128, dropout=0.3):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 160 / 32 = 5 patches per dimension -> 25 patches total
        self.num_patches = (160 // patch_size) ** 2
        
        # Shared CNN for all patches
        self.patch_cnn = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 4x4 -> 2x2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Flatten: 128 * 2 * 2 = 512
            nn.Flatten(),
            nn.Linear(512, embed_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, 160, 160)
        Returns:
            patches: (B, 25, embed_dim)
        """
        B = x.shape[0]
        ps = self.patch_size
        
        # Extract patches using unfold
        # (B, 1, 160, 160) -> (B, 1, 5, 32, 5, 32) -> (B, 25, 1, 32, 32)
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.contiguous().view(B, -1, 1, ps, ps)  # (B, 25, 1, 32, 32)
        
        # Process each patch with shared CNN
        num_patches = patches.shape[1]
        patches = patches.view(B * num_patches, 1, ps, ps)  # (B*25, 1, 32, 32)
        patch_embeds = self.patch_cnn(patches)  # (B*25, embed_dim)
        patch_embeds = patch_embeds.view(B, num_patches, -1)  # (B, 25, embed_dim)
        
        return patch_embeds


class PanelEncoder(nn.Module):
    """
    Encodes a single panel from patch embeddings.
    Uses self-attention to aggregate patch information.
    
    Input: (B, 25, embed_dim)
    Output: (B, panel_dim)
    """
    def __init__(self, embed_dim=128, panel_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Positional encoding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, 25, embed_dim) * 0.02)
        
        # Self-attention over patches
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Aggregation
        self.aggregate = nn.Sequential(
            nn.Linear(embed_dim, panel_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, patches):
        """
        Args:
            patches: (B, 25, embed_dim)
        Returns:
            panel: (B, panel_dim)
        """
        # Add positional encoding
        x = patches + self.pos_embed
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Mean pooling over patches
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Project to panel dimension
        panel = self.aggregate(x)  # (B, panel_dim)
        
        return panel


class RuleEncoder(nn.Module):
    """
    Encodes a row/column triplet into a rule embedding.
    Uses an LSTM to capture sequential relationships.
    
    Input: 3 panel embeddings
    Output: rule embedding
    """
    def __init__(self, panel_dim=256, rule_dim=128, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=panel_dim,
            hidden_size=rule_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        self.project = nn.Sequential(
            nn.Linear(rule_dim, rule_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, p1, p2, p3):
        """
        Args:
            p1, p2, p3: (B, panel_dim) - three panels in a row/column
        Returns:
            rule: (B, rule_dim)
        """
        # Stack as sequence: (B, 3, panel_dim)
        seq = torch.stack([p1, p2, p3], dim=1)
        
        # LSTM encoding
        _, (h_n, _) = self.lstm(seq)
        rule = h_n.squeeze(0)  # (B, rule_dim)
        
        rule = self.project(rule)
        return rule


class SCLModel(nn.Module):
    """
    Complete SCL-inspired model for RAVEN puzzles.
    
    Architecture:
    1. Patch Encoder: Image -> Patch embeddings (scattering transform)
    2. Panel Encoder: Patch embeddings -> Panel embedding
    3. Rule Encoder: Panel triplets -> Rule embeddings
    4. Answer Scorer: Compare candidate rules with context rules
    """
    def __init__(self, patch_size=32, embed_dim=128, panel_dim=256, rule_dim=128, dropout=0.4):
        super().__init__()
        
        self.patch_encoder = PatchEncoder(patch_size, embed_dim, dropout)
        self.panel_encoder = PanelEncoder(embed_dim, panel_dim, dropout=dropout)
        self.rule_encoder = RuleEncoder(panel_dim, rule_dim, dropout)
        
        # Validity scorer (is this a valid rule?)
        self.validity_head = nn.Sequential(
            nn.Linear(rule_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def encode_panel(self, img):
        """
        Encode a single panel image.
        
        Args:
            img: (B, 1, 160, 160)
        Returns:
            panel: (B, panel_dim)
        """
        patches = self.patch_encoder(img)  # (B, 25, embed_dim)
        panel = self.panel_encoder(patches)  # (B, panel_dim)
        return panel
        
    def forward(self, x, return_extras=False):
        """
        Args:
            x: (B, 16, 160, 160) or (B, 16, 1, 160, 160)
               16 images: 8 context + 8 choices
        Returns:
            logits: (B, 8) - scores for each choice
        """
        # Handle different input shapes
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, 16, 1, 160, 160)
        
        B = x.shape[0]
        
        # Encode all 16 panels
        x_flat = x.view(B * 16, 1, 160, 160)
        panels = []
        
        # Encode in chunks to save memory
        chunk_size = 16
        for i in range(0, B * 16, chunk_size):
            chunk = x_flat[i:i+chunk_size]
            panel = self.encode_panel(chunk)
            panels.append(panel)
        
        panels = torch.cat(panels, dim=0)  # (B*16, panel_dim)
        panels = panels.view(B, 16, -1)  # (B, 16, panel_dim)
        
        # Split into context and choices
        ctx = panels[:, :8]  # (B, 8, panel_dim)
        choices = panels[:, 8:]  # (B, 8, panel_dim)
        
        # Get context panel embeddings
        p = [ctx[:, i] for i in range(8)]
        
        # Context rules (Row 0, Row 1, Col 0, Col 1)
        # Layout:
        # P0 P1 P2
        # P3 P4 P5
        # P6 P7 [?]
        r0 = self.rule_encoder(p[0], p[1], p[2])  # Row 0
        r1 = self.rule_encoder(p[3], p[4], p[5])  # Row 1
        c0 = self.rule_encoder(p[0], p[3], p[6])  # Col 0
        c1 = self.rule_encoder(p[1], p[4], p[7])  # Col 1
        
        # Score each choice
        scores = []
        extras_r2 = []
        extras_c2 = []
        
        for i in range(8):
            choice = choices[:, i]  # (B, panel_dim)
            
            # Form Row 2 and Col 2 with this choice
            r2 = self.rule_encoder(p[6], p[7], choice)
            c2 = self.rule_encoder(p[2], p[5], choice)
            
            # Validity score
            v_row = self.validity_head(r2)
            v_col = self.validity_head(c2)
            
            # Consistency score (how well does r2 match r0/r1?)
            # Use negative MSE as similarity
            cons_row = -F.mse_loss(r2, r1, reduction='none').mean(dim=1, keepdim=True)
            cons_col = -F.mse_loss(c2, c1, reduction='none').mean(dim=1, keepdim=True)
            
            # Also compare with r0 (should learn same rule)
            cons_row0 = -F.mse_loss(r2, r0, reduction='none').mean(dim=1, keepdim=True)
            cons_col0 = -F.mse_loss(c2, c0, reduction='none').mean(dim=1, keepdim=True)
            
            # Total score
            score = v_row + v_col + cons_row + cons_col + 0.5 * (cons_row0 + cons_col0)
            scores.append(score)
            
            if return_extras:
                extras_r2.append(r2)
                extras_c2.append(c2)
        
        logits = torch.cat(scores, dim=1)  # (B, 8)
        
        if return_extras:
            extras = {
                'r0': r0, 'r1': r1, 'c0': c0, 'c1': c1,
                'r2_stack': torch.stack(extras_r2, dim=1),
                'c2_stack': torch.stack(extras_c2, dim=1),
            }
            return logits, extras
        
        return logits


class SCLLoss(nn.Module):
    """
    Combined loss for SCL model.
    
    Components:
    1. Cross-entropy for answer prediction
    2. Contrastive loss for ranking
    3. Consistency loss (r0 ≈ r1 ≈ r2)
    """
    def __init__(self, lambda_contrast=1.0, lambda_consistency=0.5, margin=1.0):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_consistency = lambda_consistency
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets, extras=None):
        """
        Args:
            logits: (B, 8)
            targets: (B,)
            extras: dict with rule embeddings
        Returns:
            total_loss, loss_dict
        """
        # Cross-entropy
        ce = self.ce_loss(logits, targets)
        
        # Contrastive loss
        B = logits.shape[0]
        correct_scores = logits[torch.arange(B), targets]
        
        contrast_loss = 0
        for i in range(8):
            mask = (torch.arange(8, device=logits.device) != targets.unsqueeze(1)).float()
            wrong_scores = logits * mask[:, 0].unsqueeze(1)  # This needs fixing
        
        # Simple max-margin contrastive
        wrong_max, _ = logits.clone().scatter(1, targets.unsqueeze(1), float('-inf')).max(dim=1)
        contrast_loss = F.relu(self.margin - correct_scores + wrong_max).mean()
        
        # Consistency loss (if extras provided)
        consistency_loss = torch.tensor(0.0, device=logits.device)
        if extras is not None:
            r0, r1 = extras['r0'], extras['r1']
            c0, c1 = extras['c0'], extras['c1']
            
            # r0 should match r1 (same row rule)
            consistency_loss = F.mse_loss(r0, r1) + F.mse_loss(c0, c1)
        
        total = ce + self.lambda_contrast * contrast_loss + self.lambda_consistency * consistency_loss
        
        return total, {
            'ce': ce.item(),
            'contrast': contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else contrast_loss,
            'consistency': consistency_loss.item(),
            'total': total.item()
        }
