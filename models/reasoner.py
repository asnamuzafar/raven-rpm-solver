"""
Stage C: Deep Learning Reasoner (Primary Reasoning Engine)

Analyzes relationships across all panels to infer visual rules.
Implements Transformer-based and MLP-based relational reasoning.
"""
import torch
import torch.nn as nn
from typing import List, Optional


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for panel positions in the 3x3 grid"""
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        return x + self.pos_embed(positions)


class TransformerReasoner(nn.Module):
    """
    Transformer-based reasoning engine.
    Uses self-attention to learn relationships between panels.
    """
    def __init__(
        self, 
        feature_dim: int = 512, 
        num_heads: int = 8, 
        num_layers: int = 4,
        hidden_dim: int = 1024, 
        dropout: float = 0.1, 
        num_choices: int = 8
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_choices = num_choices
        
        # Project features to model dimension
        self.input_proj = nn.Linear(feature_dim, feature_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(feature_dim, max_len=16)
        
        # Panel type embedding (context vs choice)
        self.type_embed = nn.Embedding(2, feature_dim)  # 0=context, 1=choice
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers - score each choice
        self.score_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(
        self, 
        context_features: torch.Tensor, 
        choice_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            context_features: (B, 8, 512) - features for 8 context panels
            choice_features: (B, 8, 512) - features for 8 choice panels
        Returns:
            logits: (B, 8) - score for each choice
        """
        B = context_features.shape[0]
        
        # Process each choice separately with context
        scores = []
        for i in range(self.num_choices):
            # Get the i-th choice for all batches
            choice = choice_features[:, i:i+1, :]  # (B, 1, 512)
            
            # Concatenate context + this choice
            seq = torch.cat([context_features, choice], dim=1)  # (B, 9, 512)
            
            # Project and add embeddings
            seq = self.input_proj(seq)
            seq = self.pos_encoding(seq)
            
            # Add type embeddings
            type_ids = torch.cat([
                torch.zeros(B, 8, dtype=torch.long, device=seq.device),
                torch.ones(B, 1, dtype=torch.long, device=seq.device)
            ], dim=1)
            seq = seq + self.type_embed(type_ids)
            
            # Apply transformer
            seq = self.norm(seq)
            seq = self.transformer(seq)  # (B, 9, 512)
            
            # Get score from the choice position (last token)
            choice_repr = seq[:, -1, :]  # (B, 512)
            score = self.score_head(choice_repr)  # (B, 1)
            scores.append(score)
        
        # Stack scores
        logits = torch.cat(scores, dim=1)  # (B, 8)
        return logits
    
    def get_attention_weights(
        self, 
        context_features: torch.Tensor, 
        choice_features: torch.Tensor, 
        choice_idx: int = 0
    ) -> List[torch.Tensor]:
        """Get attention weights for visualization"""
        B = context_features.shape[0]
        choice = choice_features[:, choice_idx:choice_idx+1, :]
        seq = torch.cat([context_features, choice], dim=1)
        seq = self.input_proj(seq)
        seq = self.pos_encoding(seq)
        
        type_ids = torch.cat([
            torch.zeros(B, 8, dtype=torch.long, device=seq.device),
            torch.ones(B, 1, dtype=torch.long, device=seq.device)
        ], dim=1)
        seq = seq + self.type_embed(type_ids)
        seq = self.norm(seq)
        
        # Get attention from each layer
        attn_weights = []
        for layer in self.transformer.layers:
            attn_output, weights = layer.self_attn(seq, seq, seq, need_weights=True)
            attn_weights.append(weights)
            seq = layer(seq)
        
        return attn_weights


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

