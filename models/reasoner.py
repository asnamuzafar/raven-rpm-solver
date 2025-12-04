"""
Stage C: Deep Learning Reasoner (Primary Reasoning Engine)

Analyzes relationships across all panels to infer visual rules.
Uses contrastive relational reasoning - models DIFFERENCES between panels.
"""
import torch
import torch.nn as nn
from typing import List, Optional


class TransformerReasoner(nn.Module):
    """
    Simple Context-Choice Scorer for RAVEN puzzles.
    
    Key insight: Concatenate all context features with each choice,
    and score how well the choice completes the pattern.
    
    This is simpler and more direct than complex relation networks.
    """
    def __init__(
        self, 
        feature_dim: int = 128, 
        num_heads: int = 8,  # unused
        num_layers: int = 3,  # unused
        hidden_dim: int = 256, 
        dropout: float = 0.1, 
        num_choices: int = 8
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_choices = num_choices
        
        # Context encoder: processes all 8 context panels together
        # 8 panels * feature_dim = input size
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim * 8, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Choice encoder: processes each candidate answer
        self.choice_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Final scorer: compares encoded context with encoded choice
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(
        self, 
        context_features: torch.Tensor, 
        choice_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            context_features: (B, 8, D) - features for 8 context panels
            choice_features: (B, 8, D) - features for 8 choice panels
            
        Returns:
            logits: (B, 8) - score for each choice
        """
        B = context_features.shape[0]
        
        # Flatten context: (B, 8, D) -> (B, 8*D)
        context_flat = context_features.view(B, -1)
        
        # Encode context
        context_encoded = self.context_encoder(context_flat)  # (B, hidden)
        
        # Score each choice
        scores = []
        for i in range(self.num_choices):
            choice = choice_features[:, i]  # (B, D)
            
            # Encode choice
            choice_encoded = self.choice_encoder(choice)  # (B, hidden)
            
            # Combine context and choice
            combined = torch.cat([context_encoded, choice_encoded], dim=-1)
            
            # Score
            score = self.scorer(combined)  # (B, 1)
            scores.append(score)
        
        logits = torch.cat(scores, dim=1)  # (B, 8)
        return logits
    
    def get_attention_weights(self, context_features, choice_features, choice_idx=0):
        """Dummy method for API compatibility"""
        return []


class MLPRelationalReasoner(nn.Module):
    """
    MLP-based relational reasoner.
    Combines features pairwise to infer patterns along rows and columns.
    """
    def __init__(
        self, 
        feature_dim: int = 512, 
        hidden_dim: int = 512, 
        num_choices: int = 8
    ):
        super().__init__()
        
        self.num_choices = num_choices
        
        # Row-wise relation module
        self.row_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Column-wise relation module  
        self.col_mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combine row and column patterns
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),  # 3 rows + 3 cols
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self, 
        context_features: torch.Tensor, 
        choice_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            context_features: (B, 8, 512) - 8 context panels (positions 0-7)
            choice_features: (B, 8, 512) - 8 choices
        Returns:
            logits: (B, 8)
        """
        B = context_features.shape[0]
        scores = []
        
        for i in range(self.num_choices):
            # Complete the 3x3 grid with this choice
            # Grid positions: 0,1,2 (row0), 3,4,5 (row1), 6,7,X (row2)
            choice = choice_features[:, i, :]  # (B, 512)
            
            # Extract rows
            row0 = torch.cat([
                context_features[:, 0], 
                context_features[:, 1], 
                context_features[:, 2]
            ], dim=1)  # (B, 1536)
            row1 = torch.cat([
                context_features[:, 3], 
                context_features[:, 4], 
                context_features[:, 5]
            ], dim=1)
            row2 = torch.cat([
                context_features[:, 6], 
                context_features[:, 7], 
                choice
            ], dim=1)
            
            # Extract columns
            col0 = torch.cat([
                context_features[:, 0], 
                context_features[:, 3], 
                context_features[:, 6]
            ], dim=1)
            col1 = torch.cat([
                context_features[:, 1], 
                context_features[:, 4], 
                context_features[:, 7]
            ], dim=1)
            col2 = torch.cat([
                context_features[:, 2], 
                context_features[:, 5], 
                choice
            ], dim=1)
            
            # Process rows and columns
            row_feats = torch.cat([
                self.row_mlp(row0),
                self.row_mlp(row1),
                self.row_mlp(row2)
            ], dim=1)  # (B, hidden*3)
            
            col_feats = torch.cat([
                self.col_mlp(col0),
                self.col_mlp(col1),
                self.col_mlp(col2)
            ], dim=1)  # (B, hidden*3)
            
            # Combine and score
            combined = torch.cat([row_feats, col_feats], dim=1)  # (B, hidden*6)
            score = self.combine(combined)  # (B, 1)
            scores.append(score)
        
        return torch.cat(scores, dim=1)  # (B, 8)

