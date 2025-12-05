"""
Rule-Aware Neuro-Symbolic Reasoner for RAVEN

Implements SOTA techniques inspired by:
- NVSA: Explicit attribute extraction + rule reasoning
- PrAE: Probabilistic program execution on attributes
- CoPINet: Contrastive perceptual inference

Key innovations:
1. Supervised attribute extraction using ground-truth from meta_matrix
2. Rule prediction heads for each attribute (Constant/Progression/Distribute/Arithmetic)
3. Rule-aware scoring that checks if choices satisfy predicted rules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class SupervisedAttributeHead(nn.Module):
    """
    Multi-task classification head for supervised attribute extraction.
    Predicts: Type (5), Size (6), Color (10), Number (9)
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
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(features)
        return {
            'type': self.type_head(shared),
            'size': self.size_head(shared),
            'color': self.color_head(shared),
            'number': self.number_head(shared)
        }

class RulePredictor(nn.Module):
    """
    Predicts which rule applies to each attribute for each row/column.
    
    Rules in I-RAVEN:
    - Constant: Same value across row (0)
    - Progression: Arithmetic sequence +/- delta (1)
    - Arithmetic: Panel3 = Panel1 ± Panel2 (2)
    - Distribute_Three: Permutation of 3 values (3)
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.num_rules = 4  # Constant, Progression, Arithmetic, Distribute
        
        # Process a row/column of 3 panels to predict rule
        self.row_encoder = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Rule classification head
        self.rule_head = nn.Linear(hidden_dim // 2, self.num_rules)
        
        # Auxiliary: predict the rule parameter (e.g., progression step)
        self.param_head = nn.Linear(hidden_dim // 2, 5)  # 5 possible params
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, row_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            row_features: (B, 3, D) - 3 panels in a row
        Returns:
            rule_logits: (B, num_rules)
            param_logits: (B, 5)
        """
        B, _, D = row_features.shape
        row_flat = row_features.view(B, -1)  # (B, 3*D)
        
        encoded = self.row_encoder(row_flat)  # (B, hidden//2)
        rule_logits = self.rule_head(encoded)  # (B, 4)
        param_logits = self.param_head(encoded)  # (B, 5)
        
        return rule_logits, param_logits


class RuleAwareReasoner(nn.Module):
    """
    Main neuro-symbolic reasoner that:
    1. Extracts supervised attributes from features
    2. Predicts rules for each row/column
    3. Scores choices based on rule satisfaction
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
        
        # Supervised attribute extraction
        self.attr_head = SupervisedAttributeHead(feature_dim, hidden_dim)
        
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
        
        # Choice scorer that combines structural consistency
        # 4 patterns each of hidden_dim // 2 = hidden_dim * 2 total
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 4 patterns × (hidden_dim/2)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Rule consistency scorer
        self.rule_consistency = nn.Sequential(
            nn.Linear(hidden_dim // 2 * 6, hidden_dim),  # 6 patterns (3 rows + 3 cols)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
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
        context_features: torch.Tensor,  # (B, 8, D)
        choice_features: torch.Tensor,   # (B, 8, D)
        return_extras: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with rule-aware scoring.
        
        Args:
            context_features: (B, 8, D) - features for 8 context panels
            choice_features: (B, 8, D) - features for 8 choice panels
            return_extras: whether to return intermediate outputs for loss computation
        
        Returns:
            logits: (B, 8) - score for each choice
        """
        B, _, D = context_features.shape
        
        # Get panel features - context is indexed 0-7
        # Grid: 0,1,2 (row0), 3,4,5 (row1), 6,7,X (row2 incomplete)
        p = [context_features[:, i] for i in range(8)]
        
        # Encode complete row patterns (row0 and row1)
        row0 = torch.cat([p[0], p[1], p[2]], dim=1)
        row1 = torch.cat([p[3], p[4], p[5]], dim=1)
        
        row0_pattern = self.pattern_encoder(row0)  # (B, H/2)
        row1_pattern = self.pattern_encoder(row1)  # (B, H/2)
        
        # Predict rules from complete rows
        row0_rule, _ = self.row_rule_predictor(context_features[:, :3])   # positions 0,1,2
        row1_rule, _ = self.row_rule_predictor(context_features[:, 3:6])  # positions 3,4,5
        
        # Predict rules from complete columns (0,3,6), (1,4,7), incomplete (2,5,X)
        col0_feats = torch.stack([p[0], p[3], p[6]], dim=1)
        col1_feats = torch.stack([p[1], p[4], p[7]], dim=1)
        col0_rule, _ = self.col_rule_predictor(col0_feats)
        col1_rule, _ = self.col_rule_predictor(col1_feats)
        
        scores = []
        for i in range(self.num_choices):
            choice = choice_features[:, i]  # (B, D)
            
            # Complete row2 and col2 with this choice
            row2 = torch.cat([p[6], p[7], choice], dim=1)
            col2 = torch.cat([p[2], p[5], choice], dim=1)
            
            row2_pattern = self.pattern_encoder(row2)
            col2_feats = torch.stack([p[2], p[5], choice], dim=1)
            
            # Get rule predictions for completed structures
            row2_rule, _ = self.row_rule_predictor(torch.stack([p[6], p[7], choice], dim=1))
            col2_rule, _ = self.col_rule_predictor(col2_feats)
            
            # Compute rule consistency: how well do row2/col2 rules match row0,1/col0,1?
            # KL divergence between rule distributions
            row_consist = F.kl_div(
                F.log_softmax(row2_rule, dim=-1), 
                F.softmax((row0_rule + row1_rule) / 2, dim=-1),
                reduction='none'
            ).sum(dim=-1, keepdim=True)
            
            col_consist = F.kl_div(
                F.log_softmax(col2_rule, dim=-1),
                F.softmax((col0_rule + col1_rule) / 2, dim=-1),
                reduction='none'
            ).sum(dim=-1, keepdim=True)
            
            # Pattern-based scoring
            combined_patterns = torch.cat([
                row0_pattern, row1_pattern, row2_pattern,
                self.pattern_encoder(col2)
            ], dim=-1)  # (B, H*2)
            
            pattern_score = self.scorer(combined_patterns)  # (B, 1)
            
            # Rule consistency score (lower KL = better match = higher score)
            consistency_score = -row_consist - col_consist  # (B, 1)
            
            # Final score combines pattern matching and rule consistency
            score = pattern_score + 0.5 * consistency_score
            scores.append(score)
        
        logits = torch.cat(scores, dim=1)  # (B, 8)
        
        if return_extras:
            return logits, {
                'row0_rule': row0_rule,
                'row1_rule': row1_rule,
                'col0_rule': col0_rule,
                'col1_rule': col1_rule,
            }
        
        return logits
    
    def compute_attribute_loss(
        self,
        context_features: torch.Tensor,
        choice_features: torch.Tensor,
        meta: Dict,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised attribute prediction loss using ground-truth from meta_matrix.
        
        Args:
            context_features: (B, 8, D)
            choice_features: (B, 8, D)
            meta: dict with 'context' (B, num_attrs, 8) and 'target_attrs' (B, num_attrs)
            target: (B,) correct choice indices
        """
        # Predict attributes for context panels
        attr_logits = self.attr_head(context_features)  # Dict of (B, 8, num_classes)
        
        total_loss = 0.0
        num_attrs = 0
        
        if 'context' in meta:
            context_gt = meta['context']  # (B, num_attrs, 8)
            
            # Map attribute indices to our heads
            # For center_single: typically 3 attributes (Type=0, Size=1, Color=2)
            # We compute loss for available attributes
            
            for attr_idx, (name, logits) in enumerate(attr_logits.items()):
                if attr_idx < context_gt.shape[1]:
                    gt = context_gt[:, attr_idx, :]  # (B, 8)
                    # Clamp gt to valid range for this attribute head
                    num_classes = logits.shape[-1]
                    gt_clamped = gt.clamp(0, num_classes - 1)
                    
                    # Cross-entropy loss over all 8 context panels
                    loss = F.cross_entropy(
                        logits.view(-1, num_classes),
                        gt_clamped.view(-1),
                        ignore_index=-1  # Skip invalid entries
                    )
                    total_loss = total_loss + loss
                    num_attrs += 1
        
        return total_loss / max(num_attrs, 1)


class NeuroSymbolicModel(nn.Module):
    """
    Complete neuro-symbolic model combining encoder and rule-aware reasoner.
    """
    def __init__(
        self, 
        encoder: nn.Module,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = encoder
        self.reasoner = RuleAwareReasoner(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor, return_extras: bool = False):
        """
        Args:
            x: (B, 16, H, W) - 16 panels per puzzle
        Returns:
            logits: (B, 8) - score for each choice
        """
        context_features, choice_features = self.encoder(x)
        return self.reasoner(context_features, choice_features, return_extras)
    
    def get_features(self, x: torch.Tensor):
        """Extract features without reasoning"""
        return self.encoder(x)
