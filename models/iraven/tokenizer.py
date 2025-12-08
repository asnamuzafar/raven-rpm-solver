"""
Stage B: Tokenizer (Symbolic Abstraction Layer)

Converts visual features to structured symbolic attributes.
Predicts: shape, size, color, count, position
"""
import torch
import torch.nn as nn
from typing import Dict, List


class SymbolicTokenizer(nn.Module):
    """
    Tokenizer that converts visual features to symbolic attributes.
    Predicts discrete labels for each panel's visual properties.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Attribute prediction heads
        self.shape_head = nn.Linear(hidden_dim, 5)      # triangle, square, pentagon, hexagon, circle
        self.size_head = nn.Linear(hidden_dim, 3)       # small, medium, large
        self.color_head = nn.Linear(hidden_dim, 4)      # dark, gray, light, white
        self.count_head = nn.Linear(hidden_dim, 9)      # 1-9 objects
        self.position_head = nn.Linear(hidden_dim, 9)   # 9 positions in grid
        
        # Attribute names for interpretability
        self.shape_names = ['triangle', 'square', 'pentagon', 'hexagon', 'circle']
        self.size_names = ['small', 'medium', 'large']
        self.color_names = ['dark', 'gray', 'light', 'white']
        self.count_names = [str(i) for i in range(1, 10)]
        self.position_names = ['top-left', 'top', 'top-right', 'left', 'center', 
                              'right', 'bottom-left', 'bottom', 'bottom-right']
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, N, 512) visual features for N panels
        Returns:
            dict of attribute logits, each (B, N, num_classes)
        """
        B, N, D = features.shape
        features = features.view(B * N, D)
        
        shared_feat = self.shared(features)  # (B*N, hidden_dim)
        
        shape_logits = self.shape_head(shared_feat).view(B, N, -1)
        size_logits = self.size_head(shared_feat).view(B, N, -1)
        color_logits = self.color_head(shared_feat).view(B, N, -1)
        count_logits = self.count_head(shared_feat).view(B, N, -1)
        position_logits = self.position_head(shared_feat).view(B, N, -1)
        
        return {
            'shape': shape_logits,
            'size': size_logits,
            'color': color_logits,
            'count': count_logits,
            'position': position_logits
        }
    
    def predict_attributes(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predicted attribute classes (not logits)"""
        logits = self.forward(features)
        return {
            'shape': logits['shape'].argmax(-1),
            'size': logits['size'].argmax(-1),
            'color': logits['color'].argmax(-1),
            'count': logits['count'].argmax(-1),
            'position': logits['position'].argmax(-1)
        }
    
    def to_symbolic(self, features: torch.Tensor) -> List[List[Dict[str, str]]]:
        """Convert features to human-readable symbolic representation"""
        preds = self.predict_attributes(features)
        B, N = preds['shape'].shape
        
        symbolic = []
        for b in range(B):
            puzzle_symbols = []
            for n in range(N):
                attrs = {
                    'shape': self.shape_names[preds['shape'][b, n].item()],
                    'size': self.size_names[preds['size'][b, n].item()],
                    'color': self.color_names[preds['color'][b, n].item()],
                    'count': self.count_names[preds['count'][b, n].item()],
                    'position': self.position_names[preds['position'][b, n].item()]
                }
                puzzle_symbols.append(attrs)
            symbolic.append(puzzle_symbols)
        return symbolic


class SymbolicEmbedding(nn.Module):
    """
    Converts symbolic attributes back to embeddings for hybrid reasoning.
    """
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.shape_embed = nn.Embedding(5, embed_dim)
        self.size_embed = nn.Embedding(3, embed_dim)
        self.color_embed = nn.Embedding(4, embed_dim)
        self.count_embed = nn.Embedding(9, embed_dim)
        self.position_embed = nn.Embedding(9, embed_dim)
        
        self.output_proj = nn.Linear(embed_dim * 5, embed_dim * 4)
        
    def forward(self, attrs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            attrs: dict of attribute indices
        Returns:
            embeddings: (B, N, embed_dim*4)
        """
        shape_emb = self.shape_embed(attrs['shape'])
        size_emb = self.size_embed(attrs['size'])
        color_emb = self.color_embed(attrs['color'])
        count_emb = self.count_embed(attrs['count'])
        position_emb = self.position_embed(attrs['position'])
        
        combined = torch.cat([shape_emb, size_emb, color_emb, count_emb, position_emb], dim=-1)
        return self.output_proj(combined)

