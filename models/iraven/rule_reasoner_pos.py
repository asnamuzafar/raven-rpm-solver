"""
Rule-Aware Neuro-Symbolic Reasoner for RAVEN (With Position Supervision)

Extended version of RuleAwareReasoner that includes a 'position' head
to explicitly supervise spatial understanding. This is critical for
overcoming the spatial bottleneck when using the SpatialAdapter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from .rule_reasoner import RulePredictor

class SupervisedAttributeHeadWithPos(nn.Module):
    """
    Multi-task classification head for supervised attribute extraction.
    Predicts: Type (5), Size (6), Color (10), Number (9), Position (9)
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.type_head = nn.Linear(hidden_dim, 5)
        self.size_head = nn.Linear(hidden_dim, 6)
        self.color_head = nn.Linear(hidden_dim, 10)
        self.number_head = nn.Linear(hidden_dim, 9)
        self.pos_head = nn.Linear(hidden_dim, 9)  # 3x3 grid positions
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(features)
        return {
            'type': self.type_head(shared),
            'size': self.size_head(shared),
            'color': self.color_head(shared),
            'number': self.number_head(shared),
            'position': self.pos_head(shared)
        }

class RuleAwareReasonerPos(nn.Module):
    """
    Neuro-symbolic reasoner with Position supervision.
    """
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_choices: int = 8
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_choices = num_choices
        
        # Supervised attribute extraction with Position
        self.attr_head = SupervisedAttributeHeadWithPos(feature_dim, hidden_dim)
        
        # Rule prediction for rows and columns
        self.row_rule_predictor = RulePredictor(feature_dim, hidden_dim)
        self.col_rule_predictor = RulePredictor(feature_dim, hidden_dim)
        
        # Context pattern encoder (for learning from complete rows)
        self.pattern_encoder = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Pattern alignment scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        context_features: torch.Tensor,
        choice_features: torch.Tensor,
        return_extras: bool = False
    ) -> torch.Tensor:
        # Re-use logic from base class would be cleaner, but inheritance is tricky 
        # because we replaced attr_head. We'll duplicate the forward pass logic 
        # to ensure it works correctly with our new structure.
        
        B, _, D = context_features.shape
        p = [context_features[:, i] for i in range(8)]
        
        row0 = torch.cat([p[0], p[1], p[2]], dim=1)
        row1 = torch.cat([p[3], p[4], p[5]], dim=1)
        
        row0_pattern = self.pattern_encoder(row0)
        row1_pattern = self.pattern_encoder(row1)
        
        # Rule prediction (auxiliary only for loss, or for scoring)
        # We simplify and use just pattern scoring + rudimentary rule consistency if needed
        # For now, let's stick to Pattern Score + Contrastive Loss as main driver
        # The base RuleAwareReasoner added rule consistency. We can add it back.
        
        # But wait, Experiment 10 used RuleAwareReasoner which relies on Rule Consistency Scorer.
        # I should really inherit or copy fully.
        # Let's import RuleAwareReasoner and just replace attr_head.
        # But `forward` uses `self.scorer`.
        
        # Actually, best way: Copy forward method from RuleAwareReasoner but we can't easily 
        # unless we subclass.
        pass 
        # Wait, I'm writing this file now. I'll implement full forward.
        
        # Predict rules (needed for consistency score)
        row0_rule, _ = self.row_rule_predictor(context_features[:, :3])
        row1_rule, _ = self.row_rule_predictor(context_features[:, 3:6])
        
        col0_feats = torch.stack([p[0], p[3], p[6]], dim=1)
        col1_feats = torch.stack([p[1], p[4], p[7]], dim=1)
        col0_rule, _ = self.col_rule_predictor(col0_feats)
        col1_rule, _ = self.col_rule_predictor(col1_feats)
        
        scores = []
        for i in range(self.num_choices):
            choice = choice_features[:, i]
            row2 = torch.cat([p[6], p[7], choice], dim=1)
            col2 = torch.cat([p[2], p[5], choice], dim=1)
            col2_feats = torch.stack([p[2], p[5], choice], dim=1)
            
            row2_pattern = self.pattern_encoder(row2)
            row2_rule, _ = self.row_rule_predictor(torch.stack([p[6], p[7], choice], dim=1))
            col2_rule, _ = self.col_rule_predictor(col2_feats)
            
            # Pattern score
            combined_patterns = torch.cat([
                row0_pattern, row1_pattern, row2_pattern,
                self.pattern_encoder(col2) # col2 used as pseudo-row pattern? 
                # Original used pattern_encoder(col2).
            ], dim=-1)
            pattern_score = self.scorer(combined_patterns)
            
            # Consistency score
            row_consist = F.kl_div(F.log_softmax(row2_rule, dim=-1), F.softmax((row0_rule + row1_rule)/2, dim=-1), reduction='none').sum(-1, keepdim=True)
            col_consist = F.kl_div(F.log_softmax(col2_rule, dim=-1), F.softmax((col0_rule + col1_rule)/2, dim=-1), reduction='none').sum(-1, keepdim=True)
            consistency_score = -row_consist - col_consist
            
            scores.append(pattern_score + 0.5 * consistency_score)
            
        logits = torch.cat(scores, dim=1)
        
        if return_extras:
            return logits, {}
            
        return logits
