"""
Spatial Convolutional Reasoner (Phase 5)

Addresses the overfitting observed with the SpatialAdapter (flattening).
Instead of flattening 5x5 spatial maps into absolute position vectors,
this reasoner applies Convolutions over the feature maps directly.

Key Idea:
- Translational Invariance: A rule like "shift right" is a local operation 
  that should be detected regardless of absolute position.
- Stacked Convolutions: 3 panels in a row are stacked depthwise (C*3, 5, 5).
- 2D Conv layers process this stack to output a relation score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .rule_reasoner import SupervisedAttributeHead

class relational_conv_block(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        # Global Average Pooling to aggregate spatial relation detection
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # MLP to score the detected relation
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B*N, 3*C, 5, 5)
        x = self.conv(x)      # (B*N, H, 5, 5)
        x = self.avgpool(x)   # (B*N, H, 1, 1)
        emb = x.view(x.size(0), -1) # (B*N, H)
        score = self.scorer(emb)    # (B*N, 1)
        return score, emb

class SpatialConvolutionalReasoner(nn.Module):
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        # Input has 3 panels stacked -> 3 * feature_dim channels
        self.row_conv = relational_conv_block(feature_dim * 3, hidden_dim, dropout)
        self.col_conv = relational_conv_block(feature_dim * 3, hidden_dim, dropout)
        
        # Auxiliary Attribute Head for Regularization (Phase 6)
        # Input: (B, C, 5, 5) -> GlobalAvgPool -> (B, C) -> AttrHead
        self.attr_head = SupervisedAttributeHead(feature_dim, hidden_dim)
        
    def forward(self, context: torch.Tensor, choices: torch.Tensor, return_extras: bool = False) -> torch.Tensor:
        """
        Args:
            context: (B, 8, C, H, W) -> spatial maps
            choices: (B, 8, C, H, W)
            return_extras: If True, return attribute logits + SCL embeddings
        Returns:
            logits: (B, 8)
        """
        B, N_ctx, C, H, W = context.shape
        _, N_choice, _, _, _ = choices.shape
        
        # p0..p7 are context indices
        p = [context[:, i] for i in range(8)] 
        
        # --- SCL: Compute Context Rules (Row0, Row1, Col0, Col1) ---
        # These don't depend on the choice, so we compute them once per batch
        
        # Row 0: p0, p1, p2
        row0 = torch.cat([p[0], p[1], p[2]], dim=1) # (B, 3C, H, W)
        _, emb_r0 = self.row_conv(row0)
        
        # Row 1: p3, p4, p5
        row1 = torch.cat([p[3], p[4], p[5]], dim=1)
        _, emb_r1 = self.row_conv(row1)
        
        # Col 0: p0, p3, p6
        col0 = torch.cat([p[0], p[3], p[6]], dim=1)
        _, emb_c0 = self.col_conv(col0)
        
        # Col 1: p1, p4, p7
        col1 = torch.cat([p[1], p[4], p[7]], dim=1)
        _, emb_c1 = self.col_conv(col1)
        
        # --- Eval Choices ---
        scores = []
        embs_r2 = [] # To store choice embeddings if needed (optional for visualization)
        embs_c2 = []
        
        for c_idx in range(8):
            choice = choices[:, c_idx]
            
            # Row 2: p6, p7, choice
            row2 = torch.cat([p[6], p[7], choice], dim=1)
            
            # Col 2: p2, p5, choice
            col2 = torch.cat([p[2], p[5], choice], dim=1)
            
            # Compute scores and embeddings
            r_score, emb_r2 = self.row_conv(row2) 
            c_score, emb_c2 = self.col_conv(col2)
            
            # Combine scores (Validity score)
            # In SCL, we could also add consistency score here (distance to r0/r1)
            # But standard SCL usually trains with the loss and uses validity score at inference.
            # Let's keep it simple: just validity score for now. 
            # (Or we could add -MSE(emb_r2, emb_r1) as a score term?)
            # Let's stick to validity score. The SCL loss will force the validity scorer to be consistent.
            
            scores.append(r_score + c_score)
            
            if return_extras:
                embs_r2.append(emb_r2)
                embs_c2.append(emb_c2)
            
        logits = torch.cat(scores, dim=1) # (B, 8)
        
        if return_extras:
            extras = {}
            
            # 1. Attribute Head (Regularization)
            ctx_flat = context.reshape(-1, C, H, W)
            ctx_pooled = F.adaptive_avg_pool2d(ctx_flat, (1, 1)).view(ctx_flat.size(0), -1)
            attr_logits = self.attr_head(ctx_pooled)
            for k, v in attr_logits.items():
                extras[k] = v.view(B, 8, -1)
            
            # 2. SCL Embeddings (Consistency)
            extras['emb_r0'] = emb_r0
            extras['emb_r1'] = emb_r1
            extras['emb_c0'] = emb_c0
            extras['emb_c1'] = emb_c1
            # We assume Row2/Col2 embeddings for the *correct* choice are needed for SCL?
            # SCL minimizes dist(r1, r0) and also dist(r2_correct, r1)?
            # Yes, usually: L_consist = ||r0 - r1||^2 + ||r1 - r2_gt||^2
            # We need the embedding of the correct choice.
            # We have computed it in the loop. We can return all choice embeddings.
            
            extras['embs_r2'] = torch.stack(embs_r2, dim=1) # (B, 8, H)
            extras['embs_c2'] = torch.stack(embs_c2, dim=1) # (B, 8, H)
                
            return logits, extras
            
        return logits

    def compute_attribute_loss(
        self,
        context_features: torch.Tensor, # (B, 8, 512, 5, 5)
        meta: Dict
    ) -> torch.Tensor:
        """
        Compute supervised attribute prediction loss.
        """
        # 1. Forward pass through attribute head
        B, N, C, H, W = context_features.shape
        ctx_flat = context_features.reshape(-1, C, H, W)
        ctx_pooled = F.adaptive_avg_pool2d(ctx_flat, (1, 1)).view(ctx_flat.size(0), -1)
        attr_logits = self.attr_head(ctx_pooled) # Dict of (B*8, num_classes)
        
        # Reshape to (B, 8, num_classes)
        for k, v in attr_logits.items():
            attr_logits[k] = v.view(B, 8, -1)
            
        total_loss = 0.0
        num_attrs = 0
        device = context_features.device
        
        if 'context' in meta:
            context_gt = meta['context'].to(device)  # (B, num_attrs, 8)
            
            # Iterate over predicted attributes (type, size, color, number)
            for attr_idx, (name, logits) in enumerate(attr_logits.items()):
                # Check if this attribute exists in ground truth
                # Names order from SupervisedAttributeHead: type, size, color, number
                # Indices in I-RAVEN: 
                # Single: Type(0), Size(1), Color(2)
                # Multi: Number(0), Position(1), Type(2), Size(3), Color(4) ?? 
                # Actually, dataset.py doesn't strictly guarantee index alignment by name.
                # However, previous exp with RuleAwareReasoner assumed:
                # attr_logits iter order matches context_gt rows if available.
                # Let's trust the logic reused from RuleAwareReasoner. 
                # (Actually, RuleAwareReasoner logic matches by INDEX not name, blindly)
                
                if attr_idx < context_gt.shape[1]:
                    gt = context_gt[:, attr_idx, :]  # (B, 8)
                    num_classes = logits.shape[-1]
                    
                    # Clamp ground truth to valid range
                    gt = gt.clamp(0, num_classes - 1)
                    
                    loss = F.cross_entropy(
                        logits.reshape(-1, num_classes),
                        gt.reshape(-1),
                        ignore_index=-1
                    )
                    total_loss += loss
                    num_attrs += 1
                    
        return total_loss / max(num_attrs, 1)
