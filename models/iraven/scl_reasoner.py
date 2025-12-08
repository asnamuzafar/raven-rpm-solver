"""
SCL-Inspired Reasoner for RAVEN

Structure-Consistency Learning (SCL) achieves 95%+ on I-RAVEN by:
1. Row/column structural attention
2. Contrastive learning between correct and incorrect candidates
3. Multi-scale feature aggregation

References:
- SCL paper: "Structure-Consistency Loss for Solving RPMs"
- NVSA: Neuro-Vector-Symbolic Architecture (98.8% I-RAVEN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class RowColumnEncoder(nn.Module):
    """
    Encodes row/column patterns using self-attention.
    Key insight: RPM rules apply along rows/columns independently.
    """
    def __init__(self, feature_dim: int = 512, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Self-attention for pattern recognition
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Position embedding for 3 positions in a row/col
        self.pos_embed = nn.Parameter(torch.randn(1, 3, feature_dim) * 0.02)
        
        # Output projection
        self.norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, D) - 3 panels in a row/column
        Returns:
            pattern: (B, D) - aggregated pattern representation
        """
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm(x + attn_out)
        
        # Aggregate using mean pooling
        pattern = x.mean(dim=1)  # (B, D)
        pattern = self.proj(pattern)
        
        return pattern


class ContrastiveScorer(nn.Module):
    """
    Scores candidates by comparing against row/column patterns.
    Uses contrastive approach: correct answer should match patterns better.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # Pattern comparison
        self.pattern_mlp = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim * 2),  # 4 patterns: 2 rows + 2 cols
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        
        # Candidate encoder
        self.candidate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),  # candidate + row2_partial + col2_partial
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Final scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self, 
        row_patterns: torch.Tensor,   # (B, 2, D) - patterns from row0, row1
        col_patterns: torch.Tensor,   # (B, 2, D) - patterns from col0, col1
        candidate: torch.Tensor,      # (B, D)
        row2_partial: torch.Tensor,   # (B, 2, D) - panels 6, 7
        col2_partial: torch.Tensor    # (B, 2, D) - panels 2, 5
    ) -> torch.Tensor:
        """
        Score a single candidate based on pattern consistency.
        """
        B = candidate.shape[0]
        
        # Flatten row/col patterns
        pattern_flat = torch.cat([
            row_patterns.view(B, -1),
            col_patterns.view(B, -1)
        ], dim=-1)  # (B, 4*D)
        
        pattern_enc = self.pattern_mlp(pattern_flat)  # (B, hidden)
        
        # Encode candidate context
        candidate_context = torch.cat([
            candidate,
            row2_partial.mean(dim=1),  # Average of row2 partial
            col2_partial.mean(dim=1),  # Average of col2 partial
        ], dim=-1)  # (B, 3*D)
        
        candidate_enc = self.candidate_mlp(candidate_context)  # (B, hidden)
        
        # Combine and score
        combined = torch.cat([pattern_enc, candidate_enc], dim=-1)
        score = self.scorer(combined)  # (B, 1)
        
        return score


class SCLReasoner(nn.Module):
    """
    Structure-Consistency Learning inspired reasoner.
    
    Architecture:
    1. Encode complete rows/columns to learn patterns
    2. For each candidate, complete the puzzle and check consistency
    3. Use contrastive loss during training
    """
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_choices: int = 8
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_choices = num_choices
        
        # Row/column pattern encoders
        self.row_encoder = RowColumnEncoder(feature_dim, num_heads, dropout)
        self.col_encoder = RowColumnEncoder(feature_dim, num_heads, dropout)
        
        # Pattern projection for consistency checking
        self.pattern_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Contrastive scorer
        self.scorer = ContrastiveScorer(feature_dim, hidden_dim, dropout)
        
        # Auxiliary: attribute prediction heads (for supervision)
        # Using 'shape' instead of 'type' to avoid conflict with Python builtin
        self.attr_heads = nn.ModuleDict({
            'shape': nn.Linear(feature_dim, 5),
            'obj_size': nn.Linear(feature_dim, 6),
            'obj_color': nn.Linear(feature_dim, 10),
            'obj_number': nn.Linear(feature_dim, 9),
        })
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def get_attr_predictions(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attribute predictions for auxiliary loss."""
        return {name: head(features) for name, head in self.attr_heads.items()}
    
    def forward(
        self,
        context_features: torch.Tensor,  # (B, 8, D)
        choice_features: torch.Tensor,   # (B, 8, D)
        return_extras: bool = False
    ) -> torch.Tensor:
        """
        Args:
            context_features: Features for 8 context panels (positions 0-7 in 3x3 grid minus bottom-right)
            choice_features: Features for 8 candidate answers
            
        Grid layout:
            0 1 2
            3 4 5
            6 7 X  <- X is the missing answer
        
        Returns:
            logits: (B, 8) score for each choice
        """
        B = context_features.shape[0]
        
        # Extract panels by position
        # Row 0: panels 0, 1, 2
        # Row 1: panels 3, 4, 5
        # Row 2 (partial): panels 6, 7
        # Col 0: panels 0, 3, 6
        # Col 1: panels 1, 4, 7
        # Col 2 (partial): panels 2, 5
        
        row0 = context_features[:, 0:3]  # (B, 3, D)
        row1 = context_features[:, 3:6]  # (B, 3, D)
        row2_partial = context_features[:, 6:8]  # (B, 2, D)
        
        col0 = torch.stack([context_features[:, i] for i in [0, 3, 6]], dim=1)  # (B, 3, D)
        col1 = torch.stack([context_features[:, i] for i in [1, 4, 7]], dim=1)  # (B, 3, D)
        col2_partial = torch.stack([context_features[:, i] for i in [2, 5]], dim=1)  # (B, 2, D)
        
        # Encode complete row/column patterns
        row0_pattern = self.row_encoder(row0)  # (B, D)
        row1_pattern = self.row_encoder(row1)  # (B, D)
        col0_pattern = self.col_encoder(col0)  # (B, D)
        col1_pattern = self.col_encoder(col1)  # (B, D)
        
        row_patterns = torch.stack([row0_pattern, row1_pattern], dim=1)  # (B, 2, D)
        col_patterns = torch.stack([col0_pattern, col1_pattern], dim=1)  # (B, 2, D)
        
        # Score each candidate
        scores = []
        for i in range(self.num_choices):
            candidate = choice_features[:, i]  # (B, D)
            
            score = self.scorer(
                row_patterns=row_patterns,
                col_patterns=col_patterns,
                candidate=candidate,
                row2_partial=row2_partial,
                col2_partial=col2_partial
            )
            scores.append(score)
        
        logits = torch.cat(scores, dim=1)  # (B, 8)
        
        if return_extras:
            return logits, {
                'row_patterns': row_patterns,
                'col_patterns': col_patterns,
                'attr_preds': self.get_attr_predictions(context_features)
            }
        
        return logits


class SCLModel(nn.Module):
    """Complete SCL model with encoder."""
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = encoder
        self.reasoner = SCLReasoner(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, return_extras: bool = False):
        """
        Args:
            x: (B, 16, H, W) - 16 panels per puzzle
        """
        context_features, choice_features = self.encoder(x)
        return self.reasoner(context_features, choice_features, return_extras)
    
    def get_features(self, x: torch.Tensor):
        return self.encoder(x)


def create_scl_model(
    pretrained_encoder: bool = True,
    freeze_encoder: bool = False,
    feature_dim: int = 512,
    hidden_dim: int = 256,
    dropout: float = 0.1
) -> SCLModel:
    """Factory function to create SCL model."""
    from models.encoder import RAVENFeatureExtractor
    
    encoder = RAVENFeatureExtractor(
        pretrained=pretrained_encoder,
        freeze=freeze_encoder,
        use_simple_encoder=False  # Always use ResNet
    )
    
    return SCLModel(
        encoder=encoder,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
