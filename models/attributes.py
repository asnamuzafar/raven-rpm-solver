"""
Shared Attribute Prediction Module

Contains:
1. SupervisedAttributeHead: predicting attributes from features
2. compute_attribute_loss: loss function for supervised attribute prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class SupervisedAttributeHead(nn.Module):
    """
    Multi-task attribute prediction heads.
    Uses ground-truth from I-RAVEN meta_matrix for supervision.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        # Shared processing for all attributes
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Per-attribute prediction heads
        # I-RAVEN attributes vary by configuration:
        # center_single: Type(5), Size(6), Color(10)
        # distribute_*: Number(9), Position(varies), Type(5), Size(6), Color(10)
        
        # We use a unified representation with max classes per attribute
        self.type_head = nn.Linear(hidden_dim, 5)      # 5 shape types
        self.size_head = nn.Linear(hidden_dim, 6)      # 6 size levels
        self.color_head = nn.Linear(hidden_dim, 10)    # 10 color levels
        self.number_head = nn.Linear(hidden_dim, 9)    # 1-9 objects
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, N, D) where N is number of panels
        Returns:
            Dict of attribute logits, each (B, N, num_classes)
        """
        B, N, D = features.shape
        features = features.reshape(B * N, D)  # Use reshape instead of view for non-contiguous
        
        shared = self.shared(features)  # (B*N, hidden)
        
        type_logits = self.type_head(shared).view(B, N, -1)
        size_logits = self.size_head(shared).view(B, N, -1)
        color_logits = self.color_head(shared).view(B, N, -1)
        number_logits = self.number_head(shared).view(B, N, -1)
        
        return {
            'type': type_logits,
            'size': size_logits,
            'color': color_logits,
            'number': number_logits
        }


def compute_attribute_loss(
    attr_head: nn.Module,
    context_features: torch.Tensor,
    meta: Dict,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Compute supervised attribute prediction loss using ground-truth from meta_matrix.
    
    Args:
        attr_head: SupervisedAttributeHead instance
        context_features: (B, 8, D)
        meta: dict with 'context' (B, num_attrs, 8)
        device: device string
    """
    if not meta or 'context' not in meta:
        return torch.tensor(0.0, device=device)

    # Predict attributes for context panels
    attr_logits = attr_head(context_features)  # Dict of (B, 8, num_classes)
    
    context_gt = meta['context'].to(device)  # (B, num_attrs, 8)
    
    total_loss = torch.tensor(0.0, device=device)
    num_attrs = 0
    
    # Match attribute indices to heads
    # I-RAVEN Attr order: Type(0), Size(1), Color(2), Number(3), Position(4)
    # Note: This mapping depends on the fixed order in I-RAVEN generator.
    # Assuming standard order: Type, Size, Color, Number. 
    # If config varies, we check what's available.
    
    # Mapping based on standard I-RAVEN Attribute.py
    # But wait, index mapping might vary by config type (center_single vs distribute).
    # Typically:
    # 0: Type
    # 1: Size
    # 2: Color
    # 3: Number
    # We will try to match based on available channels in GT
    
    attr_names_ordered = ['type', 'size', 'color', 'number']
    
    for attr_idx, name in enumerate(attr_names_ordered):
        if name in attr_logits and attr_idx < context_gt.shape[1]:
            logits = attr_logits[name]  # (B, 8, num_classes)
            gt = context_gt[:, attr_idx, :]  # (B, 8)
            
            # Clamp to valid range
            num_classes = logits.shape[-1]
            gt_clamped = gt.clamp(0, num_classes - 1)
            
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits.reshape(-1, num_classes),
                gt_clamped.reshape(-1),
                reduction='mean'
            )
            total_loss = total_loss + loss
            num_attrs += 1
    
    return total_loss / max(num_attrs, 1)
