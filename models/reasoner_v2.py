"""
Advanced Reasoner V2 for RAVEN Neuro-Symbolic Reasoning

Implements SOTA reasoning architectures inspired by:
- CoPINet: Contrastive perceptual inference
- DCNet: Dual-contrast network with rule/choice contrast
- SRAN: Stratified rule-aware reasoning

Key improvements over baseline reasoners:
1. Contrastive scoring - compare choices against context pattern
2. Multi-head attention for structural reasoning
3. Pairwise difference features for transformation modeling
4. Auxiliary rule prediction head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class MultiHeadComparison(nn.Module):
    """
    Multi-head comparison module for comparing context pattern with choices.
    
    Uses multiple attention heads to capture different aspects of comparison:
    - Shape relationships
    - Positional patterns
    - Transformation rules
    """
    def __init__(self, feature_dim: int = 512, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Projection layers for each attention head
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self, 
        query: torch.Tensor,  # (B, D) - context representation
        key: torch.Tensor,    # (B, 8, D) - choice representations
        value: torch.Tensor   # (B, 8, D) - choice representations
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention comparison.
        
        Returns:
            output: (B, 8, D) - attended choice representations
            attention: (B, num_heads, 8) - attention weights per choice
        """
        B = query.shape[0]
        
        # Project and reshape for multi-head attention
        Q = self.query_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D/H)
        K = self.key_proj(key).view(B, 8, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 8, D/H)
        V = self.value_proj(value).view(B, 8, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 8, D/H)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, 1, 8)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.matmul(attn_weights, V)  # (B, H, 1, D/H)
        output = output.transpose(1, 2).contiguous().view(B, -1)  # (B, D)
        output = self.out_proj(output)
        
        return output, attn_weights.squeeze(2)  # (B, D), (B, H, 8)


class ContrastiveReasoner(nn.Module):
    """
    Contrastive Reasoner for RAVEN puzzles.
    
    Key innovations:
    1. Encodes context panels into a single "pattern representation"
    2. Compares each choice against the pattern using contrastive scoring
    3. Uses structural reasoning (row/col/diag) to capture transformations
    4. Provides auxiliary outputs for interpretability
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
        self.hidden_dim = hidden_dim
        self.num_choices = num_choices
        
        # ========== Context Pattern Encoder ==========
        # Encodes all 8 context panels into a single pattern representation
        
        # Row encoder: processes each row to capture horizontal patterns
        self.row_encoder = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Column encoder: processes each column to capture vertical patterns
        self.col_encoder = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Pattern aggregator: combines row and column patterns
        # 3 rows (but row 2 is partial) + 3 cols (but col 2 is partial) = 2+2 = 4 patterns
        # Plus global context
        self.pattern_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 4 + feature_dim * 8, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # ========== Choice Encoder ==========
        # Projects each choice to same space as pattern
        self.choice_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # ========== Contrastive Comparison ==========
        # Multi-head comparison between pattern and choices
        self.comparison = MultiHeadComparison(
            feature_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # ========== Row/Column Completion Scorer ==========
        # Scores how well each choice completes the missing row/column
        self.row_completion_scorer = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.col_completion_scorer = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.diag_completion_scorer = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # ========== Final Scorer ==========
        # Combines contrastive comparison with completion scores
        self.final_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim // 2),  # pattern match + 3 completion scores
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # ========== Auxiliary Rule Predictor (optional) ==========
        # Predicts which rule type governs each attribute
        self.rule_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4 * 4)  # 4 attributes x 4 rule types
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
                
    def encode_context_pattern(
        self, 
        context_features: torch.Tensor  # (B, 8, D)
    ) -> torch.Tensor:
        """
        Encode 8 context panels into a single pattern representation.
        
        Uses structural encoding:
        - Row 0 (panels 0,1,2): complete horizontal pattern
        - Row 1 (panels 3,4,5): complete horizontal pattern
        - Col 0 (panels 0,3,6): complete vertical pattern
        - Col 1 (panels 1,4,7): complete vertical pattern
        """
        B, _, D = context_features.shape
        
        # Extract panels
        p0 = context_features[:, 0]  # (B, D)
        p1 = context_features[:, 1]
        p2 = context_features[:, 2]
        p3 = context_features[:, 3]
        p4 = context_features[:, 4]
        p5 = context_features[:, 5]
        p6 = context_features[:, 6]
        p7 = context_features[:, 7]
        
        # Encode complete rows (row 0 and row 1)
        row0 = self.row_encoder(torch.cat([p0, p1, p2], dim=1))  # (B, H)
        row1 = self.row_encoder(torch.cat([p3, p4, p5], dim=1))  # (B, H)
        
        # Encode complete columns (col 0 and col 1)
        col0 = self.col_encoder(torch.cat([p0, p3, p6], dim=1))  # (B, H)
        col1 = self.col_encoder(torch.cat([p1, p4, p7], dim=1))  # (B, H)
        
        # Flatten context for global information
        context_flat = context_features.view(B, -1)  # (B, 8*D)
        
        # Aggregate all patterns into single representation
        pattern = self.pattern_aggregator(
            torch.cat([row0, row1, col0, col1, context_flat], dim=1)
        )  # (B, H)
        
        return pattern
        
    def forward(
        self,
        context_features: torch.Tensor,  # (B, 8, D)
        choice_features: torch.Tensor,   # (B, 8, D)
        return_extras: bool = False
    ) -> 'torch.Tensor | tuple[torch.Tensor, Dict]':
        """
        Forward pass with contrastive scoring.
        
        Args:
            context_features: Features of 8 context panels
            choice_features: Features of 8 candidate answers
            return_extras: If True, return auxiliary outputs for loss computation
            
        Returns:
            logits: (B, 8) scores for each choice
            extras (optional): dict with context_repr, choice_reprs, rule_preds
        """
        B = context_features.shape[0]
        D = self.feature_dim
        
        # Extract context panels for structural scoring
        p6 = context_features[:, 6]  # (B, D)
        p7 = context_features[:, 7]  # (B, D)
        p2 = context_features[:, 2]
        p5 = context_features[:, 5]
        p0 = context_features[:, 0]
        p4 = context_features[:, 4]
        
        # Encode context pattern
        pattern = self.encode_context_pattern(context_features)  # (B, H)
        
        # Encode all choices
        choice_encoded = self.choice_encoder(choice_features)  # (B, 8, H)
        
        # Contrastive comparison between pattern and choices
        compared, attn_weights = self.comparison(
            pattern, choice_encoded, choice_encoded
        )  # (B, H), (B, num_heads, 8)
        
        # Score each choice
        scores = []
        for i in range(self.num_choices):
            choice = choice_features[:, i]  # (B, D)
            choice_enc = choice_encoded[:, i]  # (B, H)
            
            # Row 2 completion score (panels 6, 7, choice)
            row2 = torch.cat([p6, p7, choice], dim=1)  # (B, 3D)
            row_score = self.row_completion_scorer(row2)  # (B, 1)
            
            # Column 2 completion score (panels 2, 5, choice)
            col2 = torch.cat([p2, p5, choice], dim=1)
            col_score = self.col_completion_scorer(col2)  # (B, 1)
            
            # Main diagonal completion score (panels 0, 4, choice)
            diag = torch.cat([p0, p4, choice], dim=1)
            diag_score = self.diag_completion_scorer(diag)  # (B, 1)
            
            # Pattern matching score (cosine similarity between pattern and choice)
            pattern_match = F.cosine_similarity(pattern, choice_enc, dim=-1, eps=1e-8).unsqueeze(1)  # (B, 1)
            
            # Final score combines all components
            combined = torch.cat([
                choice_enc * pattern,  # element-wise match (B, H)
                row_score,
                col_score,
                diag_score
            ], dim=1)  # (B, H+3)
            
            # Note: combined has H+3 dims but final_scorer expects H+3
            score = self.final_scorer(combined)  # (B, 1)
            scores.append(score)
        
        logits = torch.cat(scores, dim=1)  # (B, 8)
        
        if return_extras:
            # Predict rules for interpretability
            rule_preds = self.rule_predictor(pattern)  # (B, 16)
            rule_preds = rule_preds.view(B, 4, 4)  # (B, 4 attributes, 4 rules)
            
            extras = {
                'context_repr': pattern,  # (B, H)
                'choice_reprs': choice_encoded,  # (B, 8, H)
                'attention_weights': attn_weights,  # (B, num_heads, 8)
                'rule_predictions': rule_preds  # (B, 4, 4)
            }
            return logits, extras
        
        return logits


class DualContrastReasoner(nn.Module):
    """
    Dual-Contrast Reasoner inspired by DCNet.
    
    Uses two types of contrast:
    1. Rule contrast: Compare latent rules between complete and incomplete rows
    2. Choice contrast: Enhance differences among candidate answers
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
        
        # Rule encoder: extracts latent rule from complete row/column
        self.rule_encoder = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Incomplete row encoder: encodes partial row (2 panels)
        self.partial_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Rule contrast module: compares complete vs incomplete patterns
        self.rule_contrast = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Choice contrast module: differentiates between choices
        self.choice_contrast = nn.Sequential(
            nn.Linear(feature_dim * 8, hidden_dim),  # All 8 choices
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Final scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + feature_dim, hidden_dim),
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
        choice_features: torch.Tensor    # (B, 8, D)
    ) -> torch.Tensor:
        """
        Forward pass with dual contrast.
        """
        B = context_features.shape[0]
        
        # Extract panels
        p0, p1, p2 = context_features[:, 0], context_features[:, 1], context_features[:, 2]
        p3, p4, p5 = context_features[:, 3], context_features[:, 4], context_features[:, 5]
        p6, p7 = context_features[:, 6], context_features[:, 7]
        
        # Encode complete rows (rule extraction)
        row0_rule = self.rule_encoder(torch.cat([p0, p1, p2], dim=1))
        row1_rule = self.rule_encoder(torch.cat([p3, p4, p5], dim=1))
        
        # Encode partial row 2
        row2_partial = self.partial_encoder(torch.cat([p6, p7], dim=1))
        
        # Choice contrast: encode all choices together for differentiation
        choices_flat = choice_features.view(B, -1)  # (B, 8*D)
        choice_contrast_repr = self.choice_contrast(choices_flat)  # (B, H)
        
        # Score each choice
        scores = []
        for i in range(self.num_choices):
            choice = choice_features[:, i]  # (B, D)
            
            # Complete row 2 with this choice
            row2_complete = self.rule_encoder(torch.cat([p6, p7, choice], dim=1))
            
            # Rule contrast: compare row2 rule with row0/row1 rules
            rule_diff = row2_complete - (row0_rule + row1_rule) / 2
            rule_contrast_repr = self.rule_contrast(
                torch.cat([rule_diff, row2_partial], dim=1)
            )
            
            # Combine all signals
            combined = torch.cat([
                rule_contrast_repr,
                choice_contrast_repr,
                choice
            ], dim=1)
            
            score = self.scorer(combined)
            scores.append(score)
        
        return torch.cat(scores, dim=1)
